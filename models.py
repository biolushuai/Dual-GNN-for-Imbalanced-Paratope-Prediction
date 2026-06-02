import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GraphNorm, BatchNorm
from torch_geometric.nn import JumpingKnowledge

'''
这段代码的目标是利用图神经网络 (GNN) 来处理蛋白质结构数据，并预测哪些氨基酸残基是结合位点（通常是与另一个分子相互作用的位置）。
它定义了两种主要的 GNN 架构 (ProteinBindingSiteGNN 和 HAGNN)，以及一个自定义的损失函数 (SAFocalLoss) 来处理可能存在的类别不平衡问题。
'''
# 文件: models.py

# 定义一个用于预测蛋白质结合位点的图神经网络模型。这个模型采用了双通道架构，分别处理基于序列邻近性和空间邻近性的图信息，并进行交互和融合。
class ProteinBindingSiteGNN(torch.nn.Module):
    def __init__(self, num_node_features=1024, hidden_channels=128, num_layers=3, dropout=0.4):
        super(ProteinBindingSiteGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 特征降维和规范化
        self.input_bn = nn.BatchNorm1d(num_node_features) # 4.1
            # 一个多层感知机 (MLP)，将输入的 1024 维特征降维到 hidden_channels (128 维)，中间包含 BatchNorm（批次归一化）和 ReLU 激活函数。
        self.feature_proj = nn.Sequential(                # 4.2
            nn.Linear(num_node_features, hidden_channels * 4),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        # 序列特征处理层  对应论文中的全局特征处理  包含 num_layers 个 GCNConv (图卷积网络) 层 这个通道处理基于序列距离构建的图（seq_edge_index）。
        self.seq_convs = nn.ModuleList()
        self.seq_bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0: # 第一层是 hidden_channels，后续层是 hidden_channels * 2（因为会拼接来自空间通道的特征）。
                self.seq_convs.append(GCNConv(hidden_channels, hidden_channels))
            else:
                self.seq_convs.append(GCNConv(hidden_channels * 2, hidden_channels))
            self.seq_bns.append(nn.BatchNorm1d(hidden_channels))

        # 空间特征处理层                    包含 num_layers 个 GATConv (图注意力网络) 层，使用了 4 个注意力头 (heads=4)。
        self.spatial_convs = nn.ModuleList()# 这个通道处理基于三维空间距离构建的图（spatial_edge_index）。
        self.spatial_bns = nn.ModuleList()

        for i in range(num_layers):
            if i == 0: # 第一层是 hidden_channels，后续层是 hidden_channels * 2（因为会拼接来自空间通道的特征）。
                self.spatial_convs.append(GATConv(hidden_channels, hidden_channels // 4, heads=4))
            else:
                self.spatial_convs.append(GATConv(hidden_channels * 2, hidden_channels // 4, heads=4))
            self.spatial_bns.append(nn.BatchNorm1d(hidden_channels))


        # 特征交互层
        # 对每一层，将序列通道和空间通道的输出拼接 (hidden_channels * 2)，然后通过一个简单的 MLP 将其融合回 hidden_channels 维度。
        self.interaction_layers = nn.ModuleList()  # 4.7
        for _ in range(num_layers):
            self.interaction_layers.append(nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # 全局上下文
        # 对每一层的交互特征进行图级别的平均池化 (global_mean_pool)，得到一个代表整个图（蛋白质）的全局特征，然后通过一个 MLP 处理。
        self.global_context = nn.ModuleList()
        for _ in range(num_layers):
            self.global_context.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        # 预测头
        '''
        首先，将所有 GNN 层输出的最终交互特征（seq_features 列表中的 h）沿着特征维度拼接起来，形成一个维度为 hidden_channels * num_layers 的特征向量。
        然后，将这个聚合后的特征输入一个多层 MLP，逐步降维 (hidden_channels*2 -> hidden_channels -> hidden_channels//2)。
        最后通过一个线性层输出一个标量值（维度为 1），代表每个节点（残基）是结合位点的概率（经过 Sigmoid 激活后）。
        '''
        prediction_layers = []
        current_dim = hidden_channels * num_layers
        dims = [hidden_channels * 2, hidden_channels, hidden_channels // 2]

        for dim in dims:
            prediction_layers.extend([
                nn.Linear(current_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = dim

        prediction_layers.append(nn.Linear(dims[-1], 1))

        self.node_pred = nn.Sequential(*prediction_layers) # 4.9

    def forward(self, x, seq_edge_index, spatial_edge_index, batch):
        '''
        x:输入的节点特征矩阵 (形状: [num_nodes, num_node_features])。           H(0)
        seq_edge_index: 序列图的边索引 (形状: [2, num_seq_edges])。           A
        spatial_edge_index: 空间图的边索引 (形状: [2, num_spatial_edges])。   Asurf
        batch: 批次信息，指示哪些节点属于哪个图 (形状: [num_nodes])。
        '''
        # 输入特征归一化和投影
        x = self.input_bn(x)    # 4.1
        x = self.feature_proj(x)# 4.2     X现在是H(0)

        seq_features = []
        spatial_features = []

        seq_x = x               # X_global
        spatial_x = x           # X_surf

        batch_size = int(batch.max()) + 1

        # 准备序列和空间通道的输入：第一层直接使用投影后的 x，后续层使用上一层各自通道的输出与对方通道输出的拼接（体现了通道间的交互）。
        for i in range(self.num_layers):   # 对应4.4
            # 序列特征处理 第一层是 hidden_channels，后续层是 hidden_channels * 2（因为会拼接来自空间通道的特征）。
            # # 对应公式 (4.4) 中的 Xin_global 准备
            if i > 0:
                # Xin_global = X(l-1)_global + X(l-1)_surf (论文用⊕表示矩阵加法)
                seq_input = torch.cat([seq_x, spatial_x], dim=-1)
            else:
                seq_input = seq_x

            # 对应公式4.3    X(l)_global = ReLU(BatchNorm(GCNConv(Xin_global, A)))
            seq_x = self.seq_convs[i](seq_input, seq_edge_index) # 全局处理接触图
            seq_x = self.seq_bns[i](seq_x) # 输出维度：hidden_channels
            seq_x = F.elu(seq_x)
            seq_x = F.dropout(seq_x, p=self.dropout, training=self.training)

            # 空间特征处理
            # 对应公式 (4.4) 中的 Xin_surf 准备:
            if i > 0: # 第一层是 hidden_channels，后续层是 hidden_channels * 2（因为会拼接来自空间通道的特征）。
                # Xin_surf = X(l-1)_surf + X(l-1)_global (论文用⊕表示矩阵加法)
                spatial_input = torch.cat([spatial_x, seq_x], dim=-1)
            else:
                spatial_input = spatial_x

            spatial_x = self.spatial_convs[i](spatial_input, spatial_edge_index) # 4.6
            spatial_x = self.spatial_bns[i](spatial_x) # 输出维度：hidden_channels
            spatial_x = F.elu(spatial_x)
            spatial_x = F.dropout(spatial_x, p=self.dropout, training=self.training)

            # 特征交互和全局上下文
            # 对每一层，将序列通道和空间通道的输出拼接 (hidden_channels * 2)，然后通过一个简单的 MLP 将其融合回 hidden_channels 维度。
            # 对应公式 (4.7): H(l)_fuse = ReLU(BatchNorm(Wf[X(l)_global || X(l)_surf] + bf))
            h = torch.cat([seq_x, spatial_x], dim=-1)
            h = self.interaction_layers[i](h)

            # 添加全局上下文信息。用 global_context 计算 h 的全局平均特征  属于模型的额外加强
            global_h = global_mean_pool(h, batch)
            global_h = self.global_context[i](global_h)
            global_h = global_h[batch]

            h = h + global_h

            # 保存每层特征，用于最终聚合
            seq_features.append(h)

            # 残差连接
            if i > 0:
                seq_x = seq_x + seq_features[i - 1]
                spatial_x = spatial_x + spatial_features[i - 1]

            spatial_features.append(h)

        # 聚合所有层的特征   4.8
        x = torch.cat(seq_features, dim=-1)

        # 节点预测          4.9
        pred = self.node_pred(x)

        return pred.squeeze(-1)

# 定义一个增强版的自适应 Focal Loss。Focal Loss 本身用于解决类别不平衡问题
# 这个版本在此基础上增加了自适应调整 alpha 和 gamma 参数以及其他增强项。
class SAFocalLoss(nn.Module):
    """增强版Self-Adaptive Focal Loss"""

    def __init__(self, alpha_init=0.25, gamma_init=2.0, adaptive_rate=0.005, class_weights=None):
        super(SAFocalLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.adaptive_rate = adaptive_rate
        self.class_weights = class_weights
        self.smoothing = 0.1  # 标签平滑参数

    def forward(self, inputs, targets, epoch=None):
        """
        计算增强版Self-Adaptive Focal Loss
        inputs: 模型预测值
        targets: 真实标签
        epoch: 当前训练轮次
        """
        # 标签平滑
        if self.training:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)

        # 计算预测准确度
        pred_correct = ((probs >= 0.5).float() == targets).float()

        # 自适应更新alpha和gamma
        if self.training:
            with torch.no_grad():
                acc_rate = pred_correct.mean()
                # 根据准确率动态调整alpha
                target_acc = 0.75  # 目标准确率
                self.alpha.data -= self.adaptive_rate * (acc_rate - target_acc)
                self.alpha.data.clamp_(0.1, 0.9)

                # 根据损失值动态调整gamma
                mean_loss = bce_loss.mean()
                target_loss = 0.3  # 目标损失值
                self.gamma.data += self.adaptive_rate * (mean_loss - target_loss)
                self.gamma.data.clamp_(1.0, 4.0)

        # 计算Focal项
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # 处理正负样本权重
        if self.class_weights is not None:
            alpha_weight = torch.where(targets == 1,
                                       self.alpha * self.class_weights[1],
                                       (1 - self.alpha) * self.class_weights[0])
        else:
            alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # 添加困难样本挖掘
        with torch.no_grad():
            hard_examples = bce_loss > bce_loss.mean()
            hard_weight = torch.where(hard_examples,
                                      torch.ones_like(bce_loss) * 1.5,
                                      torch.ones_like(bce_loss))

        # 最终损失计算
        loss = alpha_weight * focal_weight * bce_loss * hard_weight

        # 添加L2正则化
        l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in self.parameters())

        return loss.mean() + l2_reg


# 在 models.py 文件中添加

class HAGNNLayer(nn.Module):
    """层次化注意力图神经网络层"""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.4):
        super(HAGNNLayer, self).__init__()
        
        # 序列注意力层，添加edge_dim参数
        self.seq_attention = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=2  # 序列边特征维度：[序列距离, 物理距离]
        )
        
        # 空间注意力层，添加edge_dim参数
        self.spatial_attention = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=3  # 空间边特征维度：[距离, 角度, 二面角]
        )
        
        # 其他层保持不变
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2)
        )
        
        self.output_transform = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, x, seq_edge_index, spatial_edge_index, seq_edge_attr=None, spatial_edge_attr=None):
        # 序列注意力，使用边特征
        seq_out = self.seq_attention(x, seq_edge_index, seq_edge_attr)
        
        # 空间注意力，使用边特征
        spatial_out = self.spatial_attention(x, spatial_edge_index, spatial_edge_attr)
        
        # 计算层次化注意力权重
        combined = torch.cat([seq_out, spatial_out], dim=-1)
        attention_weights = F.softmax(self.hierarchical_attention(combined), dim=-1)
        
        # 加权融合
        weighted_seq = seq_out * attention_weights[:, 0].unsqueeze(-1)
        weighted_spatial = spatial_out * attention_weights[:, 1].unsqueeze(-1)
        
        # 特征融合
        output = self.output_transform(torch.cat([weighted_seq, weighted_spatial], dim=-1))
        
        return output


class HAGNN(torch.nn.Module):
    """层次化注意力图神经网络"""

    def __init__(self, num_node_features=1024, hidden_channels=128, num_layers=1, dropout=0.4):
        super(HAGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 特征投影层 降低输入特征维度
        self.feature_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        # HAGNN层    包含 num_layers 个 HAGNNLayer。
        self.hagnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.hagnn_layers.append(
                HAGNNLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dropout=dropout
                )
            )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, seq_edge_index, spatial_edge_index, batch):
        """
        和第一个函数的输入参数类似
        修改后的forward方法，接收分离的参数而不是Data对象
        """
        # 特征投影
        x = self.feature_proj(x)
        
        # 通过HAGNN层
        layer_outputs = []
        for layer in self.hagnn_layers:
            # 创建默认的边特征
            seq_edge_attr = torch.ones(seq_edge_index.size(1), 2, device=x.device)
            spatial_edge_attr = torch.ones(spatial_edge_index.size(1), 3, device=x.device)
            
            out = layer(
                x, 
                seq_edge_index, 
                spatial_edge_index,
                seq_edge_attr,
                spatial_edge_attr
            )
            layer_outputs.append(out)
            x = out
        
        # 最终预测
        final_output = torch.cat(layer_outputs, dim=-1)
        pred = self.output_layer(final_output)
        
        return pred.squeeze(-1)