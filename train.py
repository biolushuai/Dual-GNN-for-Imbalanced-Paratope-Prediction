import os
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, \
    recall_score,precision_recall_curve,auc
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.utils import resample
from sklearn.metrics import f1_score
from models import ProteinBindingSiteGNN, SAFocalLoss, HAGNN
from data_processing import DataProcessor, AdvancedGraphOversampler

def compute_auc_pr(labels, preds):
    p, r, _ = precision_recall_curve(labels, preds)
    auc_pr = auc(r, p)
    return auc_pr
class Trainer: # 封装了模型训练、验证、评估、模型保存、早停、学习率调整和结果记录的整个流程。
    # 初始化: 设置模型、设备、配置、损失函数、优化器、学习率调度器和用于跟踪训练状态的变量
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config

        # 使用已有的 WeightedLoss 替换 SAFocalLoss
        self.criterion = WeightedLoss(
            pos_weight=torch.tensor(5.0).to(device),  # 使用默认权重5.0
            reduction='mean'
        )

        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'], # 学习率和权重衰减从配置中读取
            weight_decay=config['weight_decay']
        )

        # 设置学习率调度器   使用 ReduceLROnPlateau，当验证集的某个指标在一定轮数 (patience=5) 内不再提升时，将学习率乘以 factor (0.5)。
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 初始化最佳指标   初始化 best_val_score (未使用) 和 best_model_path。
        self.best_val_score = 0
        self.best_model_path = os.path.join(config['save_dir'], 'best_model.pth')

        # 设置中文字体，用于绘图
        self.font = FontProperties(fname=config['font_path'])

        # 添加移动平均损失跟踪
        self.loss_window_size = 10  # 移动窗口大小
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_avg_val_loss = float('inf')
        self.patience_counter = 0

    def get_moving_average(self, loss_history):
        """计算移动平均损失"""
        if len(loss_history) < self.loss_window_size:
            return sum(loss_history) / (len(loss_history)+1)
        return sum(loss_history[-self.loss_window_size:]) / self.loss_window_size

    def train_epoch(self, train_loader):
        # 训练周期 (train_epoch): 在训练数据上执行一个完整的训练轮次（epoch），
        # 包括前向传播、损失计算、反向传播、优化器更新、梯度裁剪，并计算该轮次的训练指标。
        self.model.train() # 训练模式
        total_loss = 0
        node_predictions = []
        node_labels = []

        # 使用tqdm创建进度条，
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        # 迭代 train_loader 中的每个批次 (batch)
        for batch in progress_bar:
            try:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)    # 将数据移动到设备

                # 前向传播 - 使用分离的参数调用模型
                out = self.model(
                    batch.x, 
                    batch.seq_edge_index,
                    batch.spatial_edge_index, 
                    batch.batch # 批次信息
                )

                # 计算损失
                weights = torch.ones_like(batch.y, device=self.device)
                weights[batch.y == 1] = 5.0  # 给正样本更高的权重

                loss = F.binary_cross_entropy_with_logits(  # 计算加权二元交叉熵损失
                    out, batch.y,
                    weight=weights,
                    reduction='mean'
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                total_loss += loss.item()

                # 收集预测和标签
                node_predictions.extend(torch.sigmoid(out).detach().cpu().numpy())
                node_labels.extend(batch.y.cpu().numpy())
  
            except Exception as e:
                print(f"训练批次时出错: {str(e)}")
                print(f"批次数据形状:")
                print(f"x: {batch.x.shape}")
                print(f"seq_edge_index: {batch.seq_edge_index.shape}")
                print(f"spatial_edge_index: {batch.spatial_edge_index.shape}")
                print(f"y: {batch.y.shape}")
                continue
      
        metrics = self.calculate_metrics(  #计算整个 epoch 的训练指标。 y_true,y_pred
            np.array(node_labels),
            np.array(node_predictions)
        )

        return metrics, total_loss / len(train_loader) # 返回训练指标字典和平均训练损失

    def validate(self, val_loader):
        # 验证周期 (validate): 在验证数据上评估模型性能，计算验证损失和指标，并根据验证损失执行早停逻辑（判断是否保存最佳模型、是否停止训练）
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                out = self.model(
                    batch.x, 
                    batch.seq_edge_index,
                    batch.spatial_edge_index, 
                    batch.batch # 在HAGNN中反而没用到
                )

                # 计算验证损失
                loss = self.criterion(out, batch.y)
                total_loss += loss.item()

                predictions.extend(out.cpu().numpy())
                labels.extend(batch.y.cpu().numpy())

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)

        # 计算其他指标
        metrics = self.calculate_metrics(np.array(labels), np.array(predictions)) # y_true,y_pred
        
        # 添加验证损失到指标字典中
        metrics['val_loss'] = avg_loss
        
        # 记录验证损失
        self.val_loss_history.append(avg_loss)

        # 计算移动平均验证损失
        avg_val_loss = self.get_moving_average(self.val_loss_history)

        # 基于移动平均损失进行早停和模型选择
        '''
        早停逻辑: 计算移动平均验证损失，如果它没有改善（低于历史最佳），则增加 patience_counter；
        如果改善，则重置计数器并保存当前模型为最佳模型 (best_model.path)。
        '''
        if avg_val_loss < self.best_avg_val_loss:
            self.best_avg_val_loss = avg_val_loss
            self.patience_counter = 0
            # 保存最佳模型
            torch.save(self.model.state_dict(), self.best_model_path)
        else:
            self.patience_counter += 1
            
        # 检查是否应该早停
        should_stop = self.patience_counter >= self.config['early_stopping_patience']
        
        return metrics, avg_loss, should_stop # 返回验证指标、平均验证损失和是否应停止训练的标志。


    @staticmethod # 接收真实标签 y_true 和模型原始输出 y_pred
    def calculate_metrics(y_true, y_pred, threshold=0.5):
        """计算各种评估指标"""
        # 指标计算 (calculate_metrics): 计算一系列常用的二分类评估指标（AUC-ROC, AUC-PR, F1, MCC, Precision, Recall），
        # 并采用动态阈值选择策略来优化 F1 分数。
        # 打印输入数据信息
        print(f"\nCalculating metrics:")
        print(f"y_true shape: {y_true.shape}")
        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

        # 确保输入数据为numpy数组
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # 将模型输出通过 Sigmoid 函数转换为概率 y_pred_proba
        y_pred_proba = 1 / (1 + np.exp(-y_pred))  # sigmoid

        # 动态阈值选择：遍历一系列阈值 (0.1 到 0.9)，计算每个阈值下的 F1 分数，选择使 F1 分数最高的阈值作为最佳阈值 best_threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = threshold

        for t in thresholds:
            y_pred_binary = (y_pred_proba >= t).astype(int) # 使用最佳阈值将概率二值化为 0 或 1 (y_pred_binary)
            f1 = f1_score(y_true, y_pred_binary, zero_division=1)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        # 使用最佳阈值进行二值化
        y_pred_binary = (y_pred_proba >= best_threshold).astype(int)

        # 计算各项指标，添加zero_division参数，防止出现除零错误
        return {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': compute_auc_pr(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred_binary, zero_division=1),
            'mcc': matthews_corrcoef(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=1),
            'recall': recall_score(y_true, y_pred_binary, zero_division=1),
            'threshold': best_threshold
        }

    def _write_metrics_to_file(self, metrics, mode='a'):
        """将训练指标写入文件"""
        log_file = os.path.join(self.config['save_dir'], 'training_metrics2.txt')
        
        with open(log_file, mode) as f:
            f.write(f"\nEpoch {metrics['epoch']}, Stage {metrics['stage']}\n")
            if 'train_loss' in metrics:
                f.write(f"Training Loss: {metrics['train_loss']:.4f}\n")
            
            if 'train_metrics' in metrics:
                f.write("\nTraining Metrics:\n")
                for k, v in metrics['train_metrics'].items():
                    f.write(f"{k}: {v:.4f}\n")
                
            if 'val_metrics' in metrics:
                f.write("\nValidation Metrics:\n")
                for k, v in metrics['val_metrics'].items():
                    f.write(f"{k}: {v:.4f}\n")
                
            if 'test_metrics' in metrics:
                f.write("\nTest Metrics:\n")
                for k, v in metrics['test_metrics'].items():
                    f.write(f"{k}: {v:.4f}\n")
            
            f.write("\n" + "="*50 + "\n")

    def train(self, train_loader, val_loader, test_loader=None):
        """训练模型"""
        # 完整训练流程 (train): 协调执行多个训练周期，包括调用训练和验证周期、更新学习率、记录日志、处理早停，并在训练结束后加载最佳模型、绘制训练历史图。
        print("开始训练...")
        train_metrics_history = []
        val_metrics_history = []
        for epoch in range(self.config['num_epochs']):
            # 训练一个epoch
            train_metrics, train_loss = self.train_epoch(train_loader)
            train_metrics_history.append(train_metrics)
            
            # 验证
            val_metrics, val_loss, should_stop = self.validate(val_loader)
            val_metrics_history.append(val_metrics)
            
            # 更新学习率调度器(使用移动平均验证损失)  使用验证集的移动平均损失更新学习率调度器 self.scheduler.step()
            avg_val_loss = self.get_moving_average(self.val_loss_history)
            self.scheduler.step(avg_val_loss)
            
            # 记录训练指标
            self._write_metrics_to_file({
                'epoch': epoch + 1,
                'stage': 'training',
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
            # 打印当前epoch的移动平均损失
            train_avg_loss = self.get_moving_average(self.train_loss_history)
            print(f"\nEpoch {epoch+1}")
            print(f"Moving Average Train Loss: {train_avg_loss:.4f}")
            print(f"Moving Average Val Loss: {avg_val_loss:.4f}")
            
            # # 检查是否需要早停
            if should_stop:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
                
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.plot_training_history(train_metrics_history, val_metrics_history)
    @staticmethod
    def print_metrics(phase, metrics):
        """打印评估指标"""
        print(f"\n{phase} Metrics:")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {metrics['auc_pr']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

    def plot_training_history(self, train_history, val_history):
        """绘制训练历史图"""
        # 结果可视化 (plot_training_history): 绘制训练和验证过程中各项指标随训练周期变化的曲线图。
        metrics = ['auc_roc', 'auc_pr', 'f1', 'mcc', 'precision', 'recall']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('训练过程指标变化', fontproperties=self.font, fontsize=16)

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            train_values = [x[metric] for x in train_history]
            val_values = [x[metric] for x in val_history]

            ax.plot(train_values, label='训练')
            ax.plot(val_values, label='验证')
            ax.set_title(metric.upper(), fontproperties=self.font)
            ax.set_xlabel('Epoch', fontproperties=self.font)
            ax.set_ylabel('Score', fontproperties=self.font)
            ax.legend(prop=self.font)
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'training_history.png'))
        plt.close()


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience    # 耐心值
        self.min_delta = min_delta  # “最小变化阈值”。定义了多大的性能提升才算作“显著改善”。
        self.restore_best_weights = restore_best_weights    #  是否在训练停止后，自动将模型的权重恢复到验证集性能最佳时的状态。
        self.best_weights = None
        self.best_score = None
        self.counter = 0    # 计数器，记录性能连续多少个周期没有改善，初始为 0。
        self.best_epoch = 0 # 记录最佳得分出现时的相关信息（当前实现记录的是 counter 被重置前的值）

    def __call__(self, score, model=None):
        # 通常在每个训练周期结束后，传入当前在验证集上计算出的性能得分 score 和当前模型 model 来调用。
        # 例如 early_stopper(current_val_score, current_model)
        stop = False
        if self.best_score is None: # 即第一个epoch
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                stop = True
        else:
            self.best_score = score
            self.best_epoch = self.counter
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()

        return stop

    def get_best_weights(self):
        return self.best_weights


class WeightedLoss(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super(WeightedLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, pred, target):
        # 动态计算正样本权重
        if self.pos_weight is None:
            pos_count = target.sum()
            neg_count = target.size(0) - pos_count
            self.pos_weight = neg_count / (pos_count + 1e-8)

        # 计算加权交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            pred, target,
            pos_weight=self.pos_weight * torch.ones_like(target),
            reduction='none'
        )

        # 添加正则化项
        l1_lambda = 0.01
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss = loss + l1_lambda * l1_norm

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

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
                    # 如果有表面残基索引,则只使用表面残基的特征和标签
                    # if len(surface_indices) > 0:
                    #     surface_features = features[surface_indices]
                    #     surface_labels = labels[surface_indices]
                    #     # 重新构建表面残基的邻接矩阵
                    #     if len(spatial_adj) > 0:
                    #         surface_adj = np.zeros((len(surface_indices), len(surface_indices)))
                    #         for i, si in enumerate(surface_indices):
                    #             for j, sj in enumerate(surface_indices):
                    #                 surface_adj[i,j] = spatial_adj[si,sj]
                    #         spatial_adj = surface_adj
                    # 如果数据有效，返回一个包含提取和转换后的 NumPy 数组的新字典。这个字典包含了后续 DataProcessor 类构建图所需的关键信息
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

        # 加载训练集
        with open(os.path.join(data_dir, 'pecan-paratope-train-all-paragraph.pkl'), 'rb') as f:
            train_data = pickle.load(f)
            print(f"训练数据数量: {len(train_data)}")
            train_processed = []
            for item in train_data:
                processed = process_protein_data(item)
                # 如果 process_protein_data 返回了有效的处理后字典 processed，则将其添加到对应的 _processed 列表中（如 train_processed）
                if processed is not None:
                    train_processed.append(processed)
            print(f"处理后的训练数据数量: {len(train_processed)}")

        # 加载验证集
        with open(os.path.join(data_dir, 'pecan-paratope-val-all-paragraph.pkl'), 'rb') as f:
            val_data = pickle.load(f)
            print(f"验证数据数量: {len(val_data)}")
            val_processed = []
            for item in val_data:
                processed = process_protein_data(item)
                if processed is not None:
                    val_processed.append(processed)
            print(f"处理后的验证数据数量: {len(val_processed)}")

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
        if not train_processed or not val_processed or not test_processed:
            raise ValueError("处理后的数据集为空")

        # 打印示例数据信息
        print("\n示例数据:")
        sample = train_processed[0]
        print(f"特征维度: {sample['features'].shape}")
        print(f"标签维度: {sample['labels'].shape}")
        print(f"序列邻接矩阵维度: {sample['seq_adj'].shape}")
        print(f"空间邻接矩阵维度: {sample['spatial_adj'].shape}")

        return train_processed, val_processed, test_processed

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

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据处理
    data_processor = DataProcessor()
    
    # 加载数据
    print("开始加载数据...")
    train_data, val_data, test_data = load_protein_data(config['data_dir'])

    print("开始处理数据...")
    data_processor = DataProcessor()
    
    # 保存原始的验证和测试数据
    print("处理验证和测试数据...")
    val_graphs_original = data_processor.process_protein_data(val_data)
    test_graphs_original = data_processor.process_protein_data(test_data)
    print(f"原始验证图数量: {len(val_graphs_original)}")
    print(f"原始测试图数量: {len(test_graphs_original)}")

    # 只对训练数据进行过采样
    print("处理训练数据...")
    train_graphs = data_processor.process_protein_data(train_data)
    print(f"初始训练图数量: {len(train_graphs)}")
    
    advanced_oversampler = AdvancedGraphOversampler(
        k_neighbors=7,
        sample_ratio=2.0
    )
    train_graphs_oversampled = advanced_oversampler.oversample_graphs(train_graphs)
    print(f"过采样后训练图数量: {len(train_graphs_oversampled)}")

    model = HAGNN(
        num_node_features=config['num_node_features'],
        hidden_channels=config['hidden_channels']
    ).to(device)
    

    trainer = Trainer(model, device, config)
    
    trainer.train(
        train_graphs,
        # train_graphs_oversampled,
        val_graphs_original,
        test_graphs_original
    )
    trainer.plot_training_history

if __name__ == '__main__':
    main()