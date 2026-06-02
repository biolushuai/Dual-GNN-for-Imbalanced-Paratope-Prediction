import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import torch.nn.functional as F

# DataProcessor 用于将蛋白质数据转换为 PyTorch Geometric 图对象
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_graph(self, protein_data):
        """
        将蛋白质数据转换为图结构(可以直接被GNN应用），支持边特征
        蛋白质数据集中的key words
                features = np.array(data_item.get('ab_feature', []))
                labels = np.array(data_item.get('antibody_labels', []))
                seq_adj = np.array(data_item.get('antibody_adjacency_labels', [])) # 序列邻接矩阵
                spatial_adj = np.array(data_item.get('antibody_adjacency_labels_onsurface', [])) # 表面残基的空间邻接矩阵。
                surface_indices = np.array(data_item.get('antibody_surface_index', [])) # 获取表面残基索引
        """
        try:
            # 获取节点特征和标签
            x = torch.tensor(protein_data['features'], dtype=torch.float)
            y = torch.tensor(protein_data['labels'], dtype=torch.float)

            # 构建边索引和边特征
            seq_adj = np.array(protein_data['seq_adj'])
            spatial_adj = np.array(protein_data['spatial_adj'])
            
            
            # 处理序列边特征
            seq_edge_index = []
            seq_edge_attr = []
            for i in range(len(seq_adj)):
                for j in range(len(seq_adj[i])):
                    if seq_adj[i][j] > 0:   # 即i和j之间有序列边
                        seq_edge_index.append([i, j]) # 序列图的边索引
                        # 添加序列边特征：残基间距离
                        seq_edge_attr.append([
                            abs(i - j),  # 序列距离
                            protein_data.get('seq_distances', {}).get((i, j), 0.0),  # 物理距离
                        ])
            
            # 处理空间边特征
            spatial_edge_index = []
            spatial_edge_attr = []
            for i in range(len(spatial_adj)):
                for j in range(len(spatial_adj[i])):
                    if spatial_adj[i][j] > 0:
                        spatial_edge_index.append([i, j])
                        # 添加空间边特征：距离和角度
                        spatial_edge_attr.append([
                            protein_data.get('distances', {}).get((i, j), 0.0),  # 空间距离
                            protein_data.get('angles', {}).get((i, j), 0.0),     # 二面角
                            protein_data.get('dihedrals', {}).get((i, j), 0.0)   # 二面角
                        ])

            # 转换为张量
            seq_edge_index = torch.tensor(seq_edge_index, dtype=torch.long).t()
            spatial_edge_index = torch.tensor(spatial_edge_index, dtype=torch.long).t()
            seq_edge_attr = torch.tensor(seq_edge_attr, dtype=torch.float)
            spatial_edge_attr = torch.tensor(spatial_edge_attr, dtype=torch.float)

            # 创建带边特征的图
            return Data(
                x=x, # 节点特征
                y=y, # 标签
                seq_edge_index=seq_edge_index,# 序列图的边索引
                spatial_edge_index=spatial_edge_index, # 序列图的边特征
                seq_edge_attr=seq_edge_attr,
                spatial_edge_attr=spatial_edge_attr
            )

        except Exception as e:
            print(f"创建图结构时出错: {str(e)}")
            raise e

    def process_protein_data(self, proteins_data):
        """处理蛋白质数据列表"""
        print(f"处理 {len(proteins_data)} 个蛋白质数据...")
        graphs = []
        for i, protein in enumerate(proteins_data):
            try:
                graph = self.create_graph(protein)
                graphs.append(graph)
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(proteins_data)} 个蛋白质")
            except Exception as e:
                print(f"处理第 {i + 1} 个蛋白质时出错: {str(e)}")
                continue

        print(f"成功处理 {len(graphs)}/{len(proteins_data)} 个蛋白质")
        return graphs

# 过采样技术，用于缓解类别不平衡问题。
class AdvancedGraphOversampler:
    def __init__(self, k_neighbors=5, sample_ratio=2.0):
        self.k_neighbors = k_neighbors # 在特征空间中查找最近邻的数量（默认 5）
        self.sample_ratio = sample_ratio # 过采样的比例（默认 2.0），表示希望少数类样本增加到原来的多少倍
        print(f"初始化过采样器: k_neighbors={k_neighbors}, sample_ratio={sample_ratio}")

    def oversample_graphs(self, graph_list):
        """高级图过采样方法"""
        oversampled_graphs = []

        # 遍历一个包含 PyTorch Geometric Data 对象的列表 graph_list
        for idx, graph in enumerate(graph_list):
            try:
                print(f"\n处理第 {idx + 1}/{len(graph_list)} 个图:")
                print(f"x shape: {graph.x.shape}")
                print(f"y shape: {graph.y.shape}")
                print(f"seq_edge_index shape: {graph.seq_edge_index.shape}")
                print(f"spatial_edge_index shape: {graph.spatial_edge_index.shape}")

                pos_mask = graph.y == 1 # 检查途中正样本（graph.y == 1）的比例
                num_pos = int(pos_mask.sum())
                total_nodes = len(graph.y)
                pos_ratio = num_pos / total_nodes

                print(f"正样本数量: {num_pos}")
                print(f"总节点数量: {total_nodes}")
                print(f"正样本比例: {pos_ratio:.3f}")

                if pos_ratio < 0.3:
                    # 如果正样本比例低于某个阈值（这里是 0.3），则执行过采样
                    pos_indices = torch.where(pos_mask)[0] # 获取所有正样本节点的索引
                    num_new_samples = int(num_pos * (self.sample_ratio - 1))
                    print(f"需要生成的新样本数量: {num_new_samples}")

                    new_features = []
                    new_labels = []

                    for pos_idx in pos_indices:
                        # 对每个正样本节点 pos_idx
                        pos_feature = graph.x[pos_idx] # 获取其特征 pos_feature

                        # 计算与所有节点的欧氏距离
                        dists = torch.norm(graph.x - pos_feature, dim=1)
                        _, nn_indices = torch.topk(-dists, self.k_neighbors + 1)
                        nn_indices = nn_indices[1:]  # 排除自身

                        # 为每个正样本生成新样本
                        samples_per_node = max(1, num_new_samples // num_pos)
                        for _ in range(samples_per_node):
                            # 随机选择两个近邻
                            selected_nn = nn_indices[torch.randperm(len(nn_indices))[:2]]
                            # 生成随机插值权重
                            alpha = torch.rand(1)
                            # 插值生成新特征
                            new_feature = alpha * graph.x[selected_nn[0]] + (1 - alpha) * graph.x[selected_nn[1]]

                            new_features.append(new_feature)
                            new_labels.append(torch.tensor(1.0))

                    if new_features:
                        # 堆叠新特征和标签
                        new_features = torch.stack(new_features)
                        new_labels = torch.stack(new_labels)

                        print(f"生成的新特征形状: {new_features.shape}")

                        # 合并原始和新生成的数据
                        x = torch.cat([graph.x, new_features], dim=0)
                        y = torch.cat([graph.y, new_labels])

                        # 更新边索引和边特征
                        seq_edge_index, seq_edge_attr = self._update_edge_indices(
                            graph.seq_edge_index,
                            graph.x.size(0),
                            len(new_features),
                            x,
                            graph
                        )
                        spatial_edge_index, spatial_edge_attr = self._update_edge_indices(
                            graph.spatial_edge_index,
                            graph.x.size(0),
                            len(new_features),
                            x,
                            graph
                        )

                        new_graph = Data(
                            x=x,
                            y=y,
                            seq_edge_index=seq_edge_index,
                            spatial_edge_index=spatial_edge_index,
                            seq_edge_attr=seq_edge_attr,
                            spatial_edge_attr=spatial_edge_attr
                        )
                        print(f"过采样后的图大小: x={x.shape}, y={y.shape}")
                        oversampled_graphs.append(new_graph)
                    else:
                        oversampled_graphs.append(graph)
                else:
                    oversampled_graphs.append(graph)

            except Exception as e:
                print(f"处理图时出错: {str(e)}")
                oversampled_graphs.append(graph)

        return oversampled_graphs

    def _update_edge_indices(self, edge_index, num_orig_nodes, num_new_nodes, features, graph):
        """更新边索引和边特征,基于特征相似度连接到最近邻节点"""
        if edge_index is None:
            return None, None

        new_edges_src = []
        new_edges_dst = []
        new_edge_attrs = []  # 新增：存储边特征

        # 获取原始节点的特征
        orig_features = features[:num_orig_nodes]
        
        # 对每个新节点
        for i in range(num_new_nodes):
            new_node = num_orig_nodes + i
            new_feat = features[new_node]
            
            # 计算与所有原始节点的距离
            distances = torch.norm(orig_features - new_feat, dim=1)
            
            # 找到k个最近邻
            _, nearest_indices = torch.topk(distances, self.k_neighbors, largest=False)
            
            # 添加边连接和边特征
            for neighbor_idx in nearest_indices:
                new_edges_src.extend([new_node, neighbor_idx])
                new_edges_dst.extend([neighbor_idx, new_node])
                
                # 计算边特征
                # 1. 欧氏距离
                dist = torch.norm(features[new_node] - features[neighbor_idx])
                # 2. 余弦相似度
                cos_sim = F.cosine_similarity(
                    features[new_node].unsqueeze(0), 
                    features[neighbor_idx].unsqueeze(0)
                )
                # 3. 特征差异
                feat_diff = torch.abs(features[new_node] - features[neighbor_idx]).mean()
                
                # 为每条边添加双向的边特征
                edge_attr = torch.tensor([dist, cos_sim, feat_diff], dtype=torch.float)
                new_edge_attrs.extend([edge_attr, edge_attr])  # 双向边具有相同的特征

        if new_edges_src:
            new_edges = torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)
            new_edge_attrs = torch.stack(new_edge_attrs)
            
            # 如果已有边特征，则合并
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_attr = torch.cat([graph.edge_attr, new_edge_attrs], dim=0)
            else:
                # 为原有边创建默认特征
                num_existing_edges = edge_index.size(1)
                default_edge_attr = torch.ones(num_existing_edges, 3)  # 3个特征维度
                edge_attr = torch.cat([default_edge_attr, new_edge_attrs], dim=0)
                
            edge_index = torch.cat([edge_index, new_edges], dim=1)
            
            return edge_index, edge_attr
        
        return edge_index, None