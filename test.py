import os
import pickle

import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from models import *
from train import *


class Tester:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config


    def evaluate(self, data_loader):
        self.model.eval()

        epoch_loss = 0.0
        n = 0
        valid_pred = []
        valid_true = []

        # 使用tqdm创建进度条，
        progress_bar = tqdm(data_loader, desc="Training", leave=False)
        # 迭代 train_loader 中的每个批次 (batch)
        for batch in progress_bar:
            with (torch.no_grad()):
                # Inside Tester.evaluate, before calling self.model
                batch = batch.to(self.device)
                # print("Device check inside evaluate:")
                # print("batch.x device:", batch.x.device)
                # print("batch.seq_edge_index device:", batch.seq_edge_index.device)
                # print("batch.spatial_edge_index device:", batch.spatial_edge_index.device)
                # if hasattr(batch, 'seq_edge_attr') and batch.seq_edge_attr is not None:
                #     print("batch.seq_edge_attr device:", batch.seq_edge_attr.device)
                # if hasattr(batch, 'spatial_edge_attr') and batch.spatial_edge_attr is not None:
                #     print("batch.spatial_edge_attr device:", batch.spatial_edge_attr.device)

                # 前向传播 - 使用分离的参数调用模型
                y_pred = self.model(
                    batch.x,
                    batch.seq_edge_index,
                    batch.spatial_edge_index,
                    batch.batch  # 批次信息
                )
                # 计算损失
                weights = torch.ones_like(batch.y, device=self.device)
                weights[batch.y == 1] = 5.0  # 给正样本更高的权重

                loss = F.binary_cross_entropy_with_logits(  # 计算加权二元交叉熵损失
                    y_pred, batch.y,
                    weight=weights,
                    reduction='mean'
                )

                y_pred = torch.sigmoid(y_pred)
                y_pred = y_pred.cpu().detach().numpy()
                y_true = batch.y.cpu().detach().numpy()
                valid_pred.extend(y_pred.tolist())  # 将概率添加到列表
                valid_true.extend(y_true.tolist())  # 将真实标签添加到列表

                epoch_loss += loss.item()
                n += 1
        epoch_loss_avg = epoch_loss / n

        return epoch_loss_avg, valid_true, valid_pred

    def analysis(self,y_true, y_pred, best_threshold=None):
        if best_threshold == None:
            best_f1 = 0
            best_threshold = 0
            for threshold in range(0, 100):
                threshold = threshold / 100
                binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
                binary_true = y_true
                f1 = metrics.f1_score(binary_true, binary_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
        binary_true = y_true

        # binary evaluate
        binary_acc = metrics.accuracy_score(binary_true, binary_pred)
        precision = metrics.precision_score(binary_true, binary_pred)
        recall = metrics.recall_score(binary_true, binary_pred)
        f1 = metrics.f1_score(binary_true, binary_pred)
        AUC = metrics.roc_auc_score(binary_true, y_pred)
        precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
        AUPRC = metrics.auc(recalls, precisions)
        AUROC = metrics.roc_auc_score(binary_true,binary_pred)
        mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

        results = {
            'binary_acc': binary_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'AUC': AUC,
            'AUPRC': AUPRC,
            'AUROC': AUROC,
            'mcc': mcc,
            'threshold': best_threshold
        }
        return results


    def test(self,test_loader):

        model_name = 'best_model.pth'

        model_path = os.path.join(self.config['Model_Path'], model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        epoch_loss_test_avg, test_true, test_pred = self.evaluate(test_loader)

        result_test = self.analysis(test_true, test_pred)

        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test AUROC: ", result_test['AUROC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])
        print()

        # Export prediction
        # with open(model_name.split(".")[0] + "_pred.pkl", "wb") as f:
        # pickle.dump(pred_dict, f)



def load_protein_data(data_dir):
    """加载蛋白质数据"""

    try:
        print(f"正在从目录加载数据: {data_dir}")

        def process_protein_data(data_item):
            """处理单个蛋白质数据"""
            try:
                features = np.array(data_item.get('ab_feature', []))
                labels = np.array(data_item.get('antibody_labels', []))
                seq_adj = np.array(data_item.get('antibody_adjacency_labels', [])) # 序列邻接矩阵
                spatial_adj = np.array(data_item.get('antibody_adjacency_labels_onsurface', [])) # 表面残基的空间邻接矩阵。
                surface_indices = np.array(data_item.get('antibody_surface_index', [])) # 获取表面残基索引
                # print(surface_indices)
                if len(features) > 0 and len(labels) > 0:

                    return {
                        'features': features,
                        'labels': labels,
                        'seq_adj': seq_adj,
                        'spatial_adj': spatial_adj if len(spatial_adj) > 0 else seq_adj,
                        'surface_indices': surface_indices
                    }
            except Exception as e:
                print(f"处理数据时出错: {str(e)}")
                print(f"数据字段: {data_item.keys()}")
            return None

        # 加载测试集
        with open(os.path.join(data_dir, 'pecan-paratope-test-all-paragraph.pkl'), 'rb') as f:
            test_data = pickle.load(f)
            print(f"测试数据数量: {len(test_data)}")
            test_processed = []
            for item in test_data:
                processed = process_protein_data(item)
                if processed is not None:
                    test_processed.append(processed)
            print(f"处理后的测试数据数量: {len(test_processed)}")

        # 验证数据
        if not test_processed:
            raise ValueError("处理后的数据集为空")

        # 打印示例数据信息
        print("\n示例数据:")
        sample = test_processed[0]
        print(f"特征维度: {sample['features'].shape}")
        print(f"标签维度: {sample['labels'].shape}")
        print(f"序列邻接矩阵维度: {sample['seq_adj'].shape}")
        print(f"空间邻接矩阵维度: {sample['spatial_adj'].shape}")

        return test_processed

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        raise e


def main():
    # 更新配置参数
    config = {
        'data_dir': r'D:\wudi_datasets\datasets',
        'font_path': r'D:\wudi_datasets\仿宋_GB2312.ttf',
        'save_dir': r'D:\wudi_datasets\results',
        'Model_Path': r'D:\wudi_datasets\results',
        'save_dir2':r'D:\wudi_datasets\results2',
        'learning_rate': 2e-3,
        'weight_decay': 1e-4,
        'batch_size': 8,
        'num_epochs': 50,
        'num_node_features': 1024,
        'hidden_channels': 256,
        'num_workers': 0,
        'pos_weight': 5.0,
        'dropout': 0.5,
        'early_stopping_patience': 15,
        'lr_scheduler_patience': 5,
        'lr_scheduler_factor': 0.5,
        'min_lr': 1e-6
    }
    # 创建保存目录 (如果需要保存测试结果)
    os.makedirs(config['save_dir2'], exist_ok=True)


    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    test_data = load_protein_data(config['data_dir'])

    print("开始处理数据...")
    data_processor = DataProcessor()

    # 保存原始的验证和测试数据
    print("处理验证和测试数据...")
    test_graph_list = data_processor.process_protein_data(test_data)
    print(f"原始测试图数量: {len(test_graph_list)}")

    # !!! 创建 DataLoader !!!
    from torch_geometric.loader import DataLoader # 确保导入
    test_loader = DataLoader(
        test_graph_list,
        batch_size=config['batch_size'],
        shuffle=False, # 测试集通常不需要打乱
        num_workers=config['num_workers']
    )

    model = HAGNN(
        num_node_features=config['num_node_features'],
        hidden_channels=config['hidden_channels']
    ).to(device)
    tester = Tester(model, device, config)

    print("Evaluate HAGNN on test_data")
    tester.test(test_loader)



if __name__ == "__main__":
    main()
