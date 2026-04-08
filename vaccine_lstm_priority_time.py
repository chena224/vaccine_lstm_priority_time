import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# 中文字体配置
from matplotlib import font_manager
chinese_fonts = [f.name for f in font_manager.fontManager.ttflist if any(k in f.name.lower() for k in ['hei', 'song', 'yahei'])]
plt.rcParams["font.family"] = chinese_fonts[0] if chinese_fonts else "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# 1. 全局配置（已基于数据验证优化：路径、维度、参数）
# ==============================================================================
config = {
    'experiment_name': 'vaccine_lstm_priority_time_final',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    'data': {
        # 已修正为/mnt目录下的正确路径
        'x_path': r"D:\360MoveData\Users\不遵\Desktop\vaccine_lstm_priority_time_final\x.csv",
        'h0_path': "D:\360MoveData\Users\不遵\Desktop\vaccine_lstm_priority_time_final\h0.csv",
        'y_path': r"D:\360MoveData\Users\不遵\Desktop\vaccine_lstm_priority_time_final\y.csv",
        'seq_len': 4,          # 3优先级（X0/X1/X2）+1时间（Y0）
        'input_dim': 32,       # 每个特征的维度（X0/X1/X2/H0均为32维）
        'h0_dim': 32,          
        'output_dim': 2,       # 输出：Y1（第2月）+ Y2（第6月）
        'val_split': 0.2,
        'test_split': 0.1,
        'batch_size': 16,      # 适配1488个样本的合理batch
        'num_workers': 0       # 云端环境禁用多线程
    },
    
    'model': {
        'rnn_type': 'lstm',
        'hidden_dim': 32,      # 与输入维度匹配，避免过拟合
        'num_layers': 1,       # 小样本适配，减少复杂度
        'bidirectional': False,
        'attention_type': 'priority_time',  # 优先级+时间双注意力
        'dropout': 0.2,        # 小样本适配，降低过拟合
        'use_layer_norm': True,
        'priority_weights_init': [0.6, 0.3, 0.1]  # X0>X1>X2的初始优先级
    },
    
    'training': {
        'epochs': 50,          # 足够的训练轮次，配合早停
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,        # 适配小样本的学习率
            'weight_decay': 5e-5  # 轻度正则化
        },
        'scheduler': {
            'type': 'CosineAnnealingWarmRestarts',
            'T_0': 15,         # 学习率调整周期
            'eta_min': 1e-6     # 最小学习率
        },
        'grad_clip': 1.0,      # 梯度裁剪，防止梯度爆炸
        'early_stopping': {
            'patience': 8,      # 适配小样本，避免过早停止
            'mode': 'min'
        }
    },
    
    'logging': {
        'log_dir': './vaccine_runs',
        'save_dir': './vaccine_models'
    }
}

# ==============================================================================
# 2. 工具函数（已优化数据加载逻辑，适配验证后的列名）
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"创建目录: {path}")

class VaccineTimeDataset(Dataset):
    def __init__(self, X_spatial, h0, y_time):
        """
        适配验证后的数据结构：
        X_spatial: [样本数, 3, 32] → X0/X1/X2（32维 each）
        h0: [样本数, 32] → H0词嵌入
        y_time: [样本数, 3] → Y0/Y1/Y2（第1/2/6月阳转率）
        """
        self.X_spatial = X_spatial
        self.h0 = h0
        self.y0 = y_time[:, 0:1]  # 历史时间特征：Y0
        self.y_target = y_time[:, 1:3]  # 预测目标：Y1+Y2
    
    def __getitem__(self, idx):
        # Y0（1维）扩维至32维，与空间特征维度一致
        y0_expand = torch.tensor(self.y0[idx]).repeat(32, 1).T  # [1, 32]
        # 时空特征拼接：[3,32] + [1,32] → [4, 32]
        X_spatio_temporal = torch.cat([
            torch.tensor(self.X_spatial[idx], dtype=torch.float32),
            y0_expand.float()
        ], dim=0)
        return (
            X_spatio_temporal,  # [4, 32] 输入特征
            torch.tensor(self.h0[idx], dtype=torch.float32),  # [32] H0嵌入
            torch.tensor(self.y_target[idx], dtype=torch.float32)  # [2] 目标值
        )
    
    def __len__(self):
        return len(self.X_spatial)

def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 5))
    # MSE曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_mse'], label='训练MSE', linewidth=1.5, color='#1f77b4')
    plt.plot(history['val_mse'], label='验证MSE', linewidth=1.5, color='#ff7f0e')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('均方误差 (MSE)')
    plt.title('疫苗效果预测 - MSE损失曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    # MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='训练MAE', linewidth=1.5, color='#1f77b4')
    plt.plot(history['val_mae'], label='验证MAE', linewidth=1.5, color='#ff7f0e')
    plt.xlabel('训练轮次 (Epoch)')
    plt.ylabel('平均绝对误差 (MAE)')
    plt.title('疫苗效果预测 - MAE曲线')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_pred_vs_true(y_true, y_pred, save_path):
    time_points = ['第2月（Y1）', '第6月（Y2）']  # 匹配输出维度
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (ax, time) in enumerate(zip(axes, time_points)):
        true = y_true[:, i]
        pred = y_pred[:, i]
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        # 散点图
        ax.scatter(true, pred, alpha=0.6, color='#2ca02c', s=50)
        # 理想预测线
        ax.plot([true.min(), true.max()], [true.min(), true.max()], 'r--', linewidth=2, label='理想预测线')
        # 指标文本
        ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('真实阳转率')
        ax.set_ylabel('预测阳转率')
        ax.set_title(f'{time}疫苗效果：预测值vs真实值')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_priority_time_attn(attn_weights, save_path):
    """可视化优先级+时间注意力权重"""
    sample_attn = attn_weights[:10] if len(attn_weights)>=10 else attn_weights
    # 拆分权重：前3列=优先级（X0/X1/X2），后4列=时间（X0/X1/X2/Y0）
    priority_attn = sample_attn[:, :3]
    time_attn = sample_attn[:, 3:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 优先级权重热力图
    sns.heatmap(
        priority_attn, ax=ax1, annot=True, fmt='.4f', cmap='YlOrRd',
        xticklabels=['X0（疫苗成分）', 'X1（疾病类型）', 'X2（人群特征）'],
        yticklabels=[f'样本{i+1}' for i in range(len(sample_attn))],
        cbar_kws={'label': '优先级权重（值越大越重要）'}
    )
    ax1.set_xlabel('优先级特征')
    ax1.set_ylabel('样本')
    ax1.set_title('疫苗效果预测 - 优先级注意力权重')
    
    # 时间权重热力图
    sns.heatmap(
        time_attn, ax=ax2, annot=True, fmt='.4f', cmap='Blues',
        xticklabels=['X0', 'X1', 'X2', 'Y0（第1月效果）'],
        yticklabels=[f'样本{i+1}' for i in range(len(sample_attn))],
        cbar_kws={'label': '时间权重（值越大越重要）'}
    )
    ax2.set_xlabel('时空特征')
    ax2.set_ylabel('样本')
    ax2.set_title('疫苗效果预测 - 时间注意力权重')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def export_sample_results(y_true, y_pred, save_path):
    """导出样本级预测结果"""
    sample_df = pd.DataFrame()
    sample_df['真实值_第2月（Y1）'] = y_true[:, 0]
    sample_df['预测值_第2月（Y1）'] = y_pred[:, 0]
    sample_df['绝对误差_第2月（Y1）'] = np.abs(y_pred[:, 0] - y_true[:, 0])
    sample_df['真实值_第6月（Y2）'] = y_true[:, 1]
    sample_df['预测值_第6月（Y2）'] = y_pred[:, 1]
    sample_df['绝对误差_第6月（Y2）'] = np.abs(y_pred[:, 1] - y_true[:, 1])
    sample_df['平均绝对误差'] = sample_df[['绝对误差_第2月（Y1）', '绝对误差_第6月（Y2）']].mean(axis=1)
    sample_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 样本预测结果已保存至：{save_path}")
    return sample_df

# ==============================================================================
# 3. 核心模型（双注意力机制，适配32维输入）
# ==============================================================================
class PriorityTimeAttention(nn.Module):
    def __init__(self, hidden_dim, priority_init):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 优先级注意力权重（可学习）
        self.priority_weights = nn.Parameter(
            torch.tensor(priority_init, dtype=torch.float32),
            requires_grad=True
        )
        # 时间注意力权重（可学习）
        self.time_weights = nn.Parameter(
            torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32),
            requires_grad=True
        )
        # 特征融合层
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, lstm_out):
        batch_size = lstm_out.shape[0]
        
        # 1. 优先级权重计算（X0/X1/X2）
        priority_weights = torch.softmax(self.priority_weights, dim=0)  # [3]
        priority_expand = priority_weights.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        spatial_weighted = lstm_out[:, :3, :] * priority_expand  # [batch, 3, 32]
        
        # 2. 时间权重计算（X0/X1/X2/Y0）
        time_weights = torch.softmax(self.time_weights, dim=0)  # [4]
        time_expand = time_weights.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
        # 拼接空间加权和时间特征（Y0）
        spatio_temporal_weighted = torch.cat([
            spatial_weighted,
            lstm_out[:, 3:, :] * time_expand[:, 3:, :]
        ], dim=1)  # [batch, 4, 32]
        
        # 3. 全局融合
        global_feature = torch.sum(spatio_temporal_weighted, dim=1)  # [batch, 32]
        fused_feature = torch.tanh(self.fusion(global_feature))
        
        # 4. 整理权重用于可视化
        attn_weights = torch.cat([
            priority_weights.unsqueeze(0).repeat(batch_size, 1),  # [batch, 3]
            time_weights.unsqueeze(0).repeat(batch_size, 1)       # [batch, 4]
        ], dim=1)  # [batch, 7]
        
        return fused_feature, attn_weights

class VaccineLSTMWithPriorityTime(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        data_config = config['data']
        model_config = config['model']
        
        # H0维度适配层（32维→32维，与hidden_dim一致）
        self.h0_proj = nn.Linear(data_config['h0_dim'], model_config['hidden_dim'])
        
        # LSTM层（处理[batch, 4, 32]的时空特征）
        self.lstm = nn.LSTM(
            input_size=data_config['input_dim'],
            hidden_size=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            batch_first=True,
            bidirectional=model_config['bidirectional'],
            dropout=model_config['dropout'] if model_config['num_layers']>1 else 0.0
        )
        
        # 双注意力层
        self.attention = PriorityTimeAttention(
            hidden_dim=model_config['hidden_dim'],
            priority_init=model_config['priority_weights_init']
        )
        
        # 输出层（预测Y1/Y2）
        self.layer_norm = nn.LayerNorm(model_config['hidden_dim']) if model_config['use_layer_norm'] else nn.Identity()
        self.dropout = nn.Dropout(model_config['dropout'])
        self.fc = nn.Linear(model_config['hidden_dim'], data_config['output_dim'])
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def forward(self, X_spatio_temporal, h0, return_attn=False):
        batch_size = X_spatio_temporal.size(0)
        model_config = self.config['model']
        num_layers = model_config['num_layers']
        num_directions = 2 if model_config['bidirectional'] else 1
        
        # H0投影：[batch, 32] → [batch, 32]
        h0_projected = self.h0_proj(h0)
        h0_lstm = h0_projected.unsqueeze(0).repeat(num_layers * num_directions, 1, 1)
        c0_lstm = torch.zeros_like(h0_lstm)
        
        # LSTM前向传播
        lstm_out, (hidden, _) = self.lstm(X_spatio_temporal, (h0_lstm, c0_lstm))
        
        # 双注意力融合
        fused_feature, attn_weights = self.attention(lstm_out)
        
        # 输出预测
        out = self.layer_norm(fused_feature)
        out = self.dropout(out)
        out = self.fc(out)
        
        if return_attn:
            return out, attn_weights
        return out

# ==============================================================================
# 4. 数据加载（已基于数据验证优化列名匹配）
# ==============================================================================
def filter_numeric_columns(df, expected_dim, col_prefix):
    """筛选指定前缀的数值列，确保维度正确"""
    numeric_cols = [col for col in df.columns if col.startswith(col_prefix)]
    # 验证列数量
    assert len(numeric_cols) == expected_dim, f"预期{expected_dim}列，实际{len(numeric_cols)}列（前缀：{col_prefix}）"
    # 验证数值类型
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"列{col}不是数值类型（前缀：{col_prefix}）"
    return df[numeric_cols].values.astype(np.float32)

def load_and_prepare_data(config):
    data_config = config['data']
    # 读取数据
    X_df = pd.read_csv(data_config['x_path'], encoding="utf-8-sig")
    h0_df = pd.read_csv(data_config['h0_path'], encoding="utf-8-sig")
    y_df = pd.read_csv(data_config['y_path'], encoding="utf-8-sig")
    
    # 验证样本ID一致性
    x_ids = set(X_df["样本ID"].tolist())
    h0_ids = set(h0_df["样本ID"].tolist())
    y_ids = set(y_df["样本ID"].tolist())
    assert x_ids == h0_ids == y_ids, "三个文件的样本ID不一致"
    print(f"✅ 样本ID完全一致，共{len(X_df)}个样本")
    
    # 提取X0/X1/X2（32维 each，前缀：X0_vec_, X1_vec_, X2_vec_）
    X0 = filter_numeric_columns(X_df, 32, 'X0_vec_')
    X1 = filter_numeric_columns(X_df, 32, 'X1_vec_')
    X2 = filter_numeric_columns(X_df, 32, 'X2_vec_')
    X_spatial = np.stack([X0, X1, X2], axis=1)  # [样本数, 3, 32]
    print(f"✅ X空间特征形状：{X_spatial.shape}（样本数, 优先级数, 维度）")
    
    # 提取H0（32维，前缀：H0_vec_）
    h0 = filter_numeric_columns(h0_df, 32, 'H0_vec_')
    print(f"✅ H0嵌入形状：{h0.shape}（样本数, 维度）")
    
    # 提取Y时间序列（3个阳转率列，前缀：Y0_, Y1_, Y2_）
    y_cols = [col for col in y_df.columns if "第" in col and "月阳转率" in col]
    assert len(y_cols) == 3, f"预期3个阳转率列，实际{len(y_cols)}列"
    y_time = y_df[y_cols].values.astype(np.float32)
    # 验证Y值范围（0~1）
    assert (y_time >= 0).all() and (y_time <= 1).all(), "Y值超出0~1范围"
    print(f"✅ Y时间序列形状：{y_time.shape}（样本数, 时间点数）")
    print(f"   Y列名：{y_cols}")
    
    # 划分数据集
    total_size = len(X_spatial)
    test_size = int(total_size * data_config['test_split'])
    val_size = int(total_size * data_config['val_split'])
    train_size = total_size - val_size - test_size
    
    # 固定随机种子划分
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        VaccineTimeDataset(X_spatial, h0, y_time), 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=data_config['batch_size'], shuffle=True, 
        num_workers=data_config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=data_config['batch_size'], shuffle=False, 
        num_workers=data_config['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=data_config['batch_size'], shuffle=False, 
        num_workers=data_config['num_workers']
    )
    
    print(f"✅ 数据集划分完成：训练{len(train_dataset)}个，验证{len(val_dataset)}个，测试{len(test_dataset)}个")
    return train_loader, val_loader, test_loader, X_spatial, h0, y_time

# ==============================================================================
# 5. 训练与评估函数
# ==============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_mse, total_mae, total = 0.0, 0.0, 0
    for X, h0_batch, y in train_loader:
        X, h0_batch, y = X.to(device), h0_batch.to(device), y.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        y_pred, _ = model(X, h0_batch, return_attn=True)
        loss = criterion(y_pred, y)
        
        # 反向传播
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # 计算指标
        y_cpu = y.cpu().numpy()
        y_pred_cpu = y_pred.detach().cpu().numpy()
        total_mse += mean_squared_error(y_cpu, y_pred_cpu) * len(y)
        total_mae += mean_absolute_error(y_cpu, y_pred_cpu) * len(y)
        total += len(y)
    
    return total_mse/total, total_mae/total

def validate(model, val_loader, criterion, device, return_details=False):
    model.eval()
    total_mse, total_mae, total = 0.0, 0.0, 0
    all_y, all_pred, all_attn = [], [], []
    
    with torch.no_grad():
        for X, h0_batch, y in val_loader:
            X, h0_batch, y = X.to(device), h0_batch.to(device), y.to(device)
            y_pred, attn = model(X, h0_batch, return_attn=True)
            
            # 计算损失和指标
            loss = criterion(y_pred, y)
            y_cpu = y.cpu().numpy()
            y_pred_cpu = y_pred.cpu().numpy()
            attn_cpu = attn.cpu().numpy()
            
            total_mse += mean_squared_error(y_cpu, y_pred_cpu) * len(y)
            total_mae += mean_absolute_error(y_cpu, y_pred_cpu) * len(y)
            total += len(y)
            
            # 保存详细结果
            all_y.append(y_cpu)
            all_pred.append(y_pred_cpu)
            all_attn.append(attn_cpu)
    
    # 拼接结果
    all_y = np.concatenate(all_y, axis=0) if all_y else np.array([])
    all_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    all_attn = np.concatenate(all_attn, axis=0) if all_attn else np.array([])
    
    avg_mse = total_mse / total
    avg_mae = total_mae / total
    
    if return_details:
        return avg_mse, avg_mae, all_y, all_pred, all_attn
    return avg_mse, avg_mae, None, None, None

# ==============================================================================
# 6. 主函数（完整运行流程）
# ==============================================================================
def main():
    # 初始化
    set_seed(config['seed'])
    device = torch.device(config['device'])
    print(f"="*60)
    print(f"开始训练：疫苗效果时空融合预测模型")
    print(f"使用设备：{device}")
    print(f"实验名称：{config['experiment_name']}")
    print(f"="*60)
    
    # 创建保存目录
    save_dir = os.path.join(config['logging']['save_dir'], config['experiment_name'])
    ensure_dir(save_dir)
    best_model_path = os.path.join(save_dir, 'best_vaccine_model_final.pth')
    print(f"模型保存路径：{best_model_path}")
    
    # 加载数据（已验证过格式和维度）
    train_loader, val_loader, test_loader, X_spatial, h0, y_time = load_and_prepare_data(config)
    
    # 创建模型
    model = VaccineLSTMWithPriorityTime(config).to(device)
    print(f"✅ 模型创建完成，结构如下：")
    print(model)
    
    # 定义训练组件
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['training']['scheduler']['T_0'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # 训练循环
    history = {'train_mse': [], 'train_mae': [], 'val_mse': [], 'val_mae': []}
    best_val_mse, patience = float('inf'), 0
    
    print(f"开始训练（共{config['training']['epochs']}轮）：")

    for epoch in range(config['training']['epochs']):
        # 训练一轮
        train_mse, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config['training']['grad_clip']
        )
        
        # 验证一轮
        val_mse, val_mae, _, _, _ = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史
        history['train_mse'].append(train_mse)
        history['train_mae'].append(train_mae)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        
        # 打印日志
        print(f"轮次 {epoch+1:2d}/{config['training']['epochs']} | "
              f"训练MSE: {train_mse:.4f} | 训练MAE: {train_mae:.4f} | "
              f"验证MSE: {val_mse:.4f} | 验证MAE: {val_mae:.4f}")
        
        # 早停逻辑
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mse': best_val_mse
            }, best_model_path)
            patience = 0
            print(f"   ✅ 保存最佳模型（验证MSE: {best_val_mse:.4f}）")
        else:
            patience += 1
            print(f"   ⚠️  验证MSE未下降，耐心剩余: {config['training']['early_stopping']['patience'] - patience}")
            if patience >= config['training']['early_stopping']['patience']:
                print(f"   ⏹️  早停触发，停止训练")
                break
    
    # 测试集评估
    print(f"" + "="*60)
    print(f"测试集评估（加载最佳模型）")
    print(f"="*60)
    # 加载最佳模型
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 加载轮次{checkpoint['epoch']}的最佳模型（验证MSE: {checkpoint['best_val_mse']:.4f}）")
    
    # 测试评估
    test_mse, test_mae, test_y, test_pred, test_attn = validate(
        model, test_loader, criterion, device, return_details=True
    )
    
    # 输出测试指标
    print(f"测试集整体结果：")
    print(f"  MSE: {test_mse:.4f} | MAE: {test_mae:.4f} | R²: {r2_score(test_y.flatten(), test_pred.flatten()):.4f}")
    print(f"第2月（Y1）结果：")
    print(f"  MSE: {mean_squared_error(test_y[:,0], test_pred[:,0]):.4f} | R²: {r2_score(test_y[:,0], test_pred[:,0]):.4f}")
    print(f"第6月（Y2）结果：")
    print(f"  MSE: {mean_squared_error(test_y[:,1], test_pred[:,1]):.4f} | R²: {r2_score(test_y[:,1], test_pred[:,1]):.4f}")
    
    # 注意力权重统计
    priority_avg = test_attn[:, :3].mean(axis=0)
    time_avg = test_attn[:, 3:].mean(axis=0)
    print(f"注意力权重统计（平均值）：")
    print(f"  优先级权重（X0/X1/X2）: {priority_avg.round(4)} → 重要性排序：X{np.argmax(priority_avg)+0} > X{np.argsort(priority_avg)[1]+0} > X{np.argmin(priority_avg)+0}")
    print(f"  时间权重（X0/X1/X2/Y0）: {time_avg.round(4)} → Y0（第1月）权重：{time_avg[3]:.4f}")
    
    # 生成可视化结果
    print(f"生成可视化结果：")
    plot_training_history(history, os.path.join(save_dir, '训练曲线_final.png'))
    print(f"  ✅ 训练曲线已保存")
    plot_pred_vs_true(test_y, test_pred, os.path.join(save_dir, '预测vs真实_final.png'))
    print(f"  ✅ 预测vs真实图已保存")
    plot_priority_time_attn(test_attn, os.path.join(save_dir, '注意力权重_final.png'))
    print(f"  ✅ 注意力权重图已保存")
    
    # 导出样本结果
    export_sample_results(test_y, test_pred, os.path.join(save_dir, '样本预测结果_final.csv'))
    

    print(f"✅ 模型训练与评估完全完成！")
    print(f"所有结果保存在：{save_dir}")
    print(f"="*60)

if __name__ == '__main__':
    main()
