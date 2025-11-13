import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class BinaryLogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(BinaryLogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

def train_binary_logistic_regression_pytorch(X_train, y_train, X_test, y_test, num_epochs=1000, learning_rate=0.01):
    # 转换数据为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)  # 转换为列向量
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # 获取输入维度
    input_dim = X_train.shape[1]
    
    # 创建模型并移动到设备
    model = BinaryLogisticRegressionPyTorch(input_dim).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练模型
    losses = []
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 测试模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted_probs = test_outputs.data.cpu().numpy()
        predicted = (predicted_probs > 0.5).astype(int).flatten()
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        f1 = f1_score(y_test, predicted)
        
        # 计算AUC指标（对不平衡数据更敏感）
        auc_roc = roc_auc_score(y_test, predicted_probs.flatten())
        auc_pr = average_precision_score(y_test, predicted_probs.flatten())
        
        print('\n========== PyTorch Binary Logistic Regression 模型评估 ==========')
        print('\n混淆矩阵：')
        print(confusion_matrix(y_test, predicted))
        print(f"\n准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    return model, predicted, predicted_probs.flatten()

# 如果您想在现有的供应链欺诈检测项目中使用这个PyTorch二分类LR模型，
# 可以取消下面的注释并根据需要调整数据加载部分

"""
# 加载数据（请根据您的实际数据路径进行调整）
# dataset = pd.read_csv('/kaggle/input/source/SupplyChain.csv', encoding='unicode_escape')
# data = dataset.copy()

# 数据预处理（根据您的实际需求进行调整）
# ... 数据预处理代码 ...

# 将多分类问题转换为二分类问题（例如：将标签8视为正类，其他为负类）
# y_binary = y.apply(lambda x: 1 if x == 8 else 0)

# 使用SMOTE平衡数据
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 调用PyTorch二分类LR模型
# model, predictions, probabilities = train_binary_logistic_regression_pytorch(
#     X_train_resampled, y_train_resampled, X_test, y_test, 
#     num_epochs=1000, learning_rate=0.01
# )
"""

if __name__ == "__main__":
    # 示例：生成一些示例数据进行测试
    print("生成示例数据进行测试...")
    
    # 生成示例数据（二分类）
    np.random.seed(42)
    X_sample = np.random.randn(1000, 10)
    y_sample = np.random.randint(0, 2, 1000)  # 二分类
    
    # 分割数据
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_sample = scaler.fit_transform(X_train_sample)
    X_test_sample = scaler.transform(X_test_sample)
    
    # 训练模型
    print("开始训练PyTorch二分类逻辑回归模型...")
    model, predictions, probabilities = train_binary_logistic_regression_pytorch(
        X_train_sample, y_train_sample, X_test_sample, y_test_sample,
        num_epochs=500, learning_rate=0.01
    )
    
    print("PyTorch二分类逻辑回归模型训练完成！")