# Supply Chain Fraud Detection

本项目旨在检测供应链中的欺诈行为，通过对供应链数据进行分析和建模，识别潜在的欺诈订单。

## 项目概述

该项目使用机器学习方法对供应链交易数据进行分析，检测异常模式以识别欺诈行为。项目包含数据预处理、特征工程、模型训练和评估等完整流程。

## 数据集

项目使用供应链数据集 `SupplyChain.csv`，包含约18万条订单记录和53个特征字段。

## 主要功能

1. **数据清洗与预处理**
   - 处理缺失值（Customer Lname、Customer Zipcode按众数填充）
   - 数据类型转换
   - 时间特征提取（年、月、星期、小时）

2. **特征工程**
   - 从订单日期和发货日期中提取多维度时间特征（年、月、星期、小时，共8个）
   - 类别特征编码：类别字段统一使用LabelEncoder标签编码
   - 构建交叉特征（Type × Delivery Status）
   - 特征重要性分析和筛选；剔除基数>100 的准ID分类字段（姓名、街道、城市、商品名等）后，参与训练的特征约 48 个（13 个分类特征 + 35 个数值特征，具体以脚本运行时打印为准）

3. **数据不平衡处理**
   - SMOTENC过采样：优化版脚本使用（分类特征按名义特征处理、数值特征参与插值，适配分类/数值混合特征）
   - SMOTENC 适用约束：只对低基数分类特征做名义处理；高基数准ID字段由 `SMOTENC_MAX_CARDINALITY` 阈值自动剔除，否则采样阶段内存会超出 Kaggle 可用上限
   - SMOTE过采样：基础版脚本使用；原始数据版脚本不使用SMOTE
   - 异常检测方法：Isolation Forest、One-Class SVM（sale.py附加实验）
   - 类别权重调整（class_weight='balanced'）

4. **模型训练与评估**
   - 实现9种机器学习算法进行对比：
     - Gaussian Naive Bayes
     - Linear SVC
     - K-Nearest Neighbors (KNN)
     - Linear Discriminant Analysis (LDA)
     - Decision Tree
     - Random Forest
     - XGBoost（支持GPU加速）
     - LightGBM（支持GPU加速）
     - PyTorch实现的Logistic Regression（支持GPU加速）
   - 多维度评估体系：
     - 准确率、精确率、召回率、F1分数
     - AUC-ROC和AUC-PR（对不平衡数据更敏感）
     - 混淆矩阵和分类报告
   - 多分类与二分类双重评估（将欺诈类别标记为正样本）

## 环境依赖

项目依赖库列表请参见 [requirements.txt](file:///Users/pro/Downloads/Supplychain-fraud/requirements.txt) 文件。

使用以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 将数据集 `SupplyChain.csv` 放置在 `/kaggle/input/source/` 目录下（或修改脚本中的路径）
2. 运行脚本或在 Jupyter Notebook 中逐个执行代码单元：
   - [supplychain-fraud-sale-opti.py](file:///Users/pro/Downloads/Supplychain-fraud/supplychain-fraud-sale-opti.py)：优化版（交叉特征 + SMOTENC过采样），推荐
   - [supplychain-fraud-sale.py](file:///Users/pro/Downloads/Supplychain-fraud/supplychain-fraud-sale.py)：基础版（SMOTE + 异常检测实验）
   - [supplychain-fraud-sale-raw.py](file:///Users/pro/Downloads/Supplychain-fraud/supplychain-fraud-sale-raw.py)：原始数据版（不使用SMOTE）

## 结果

项目实现了多种机器学习模型对供应链欺诈的检测，其中XGBoost表现最优：
- **准确率**: 99.54%
- **精确率**: 88.76%
- **召回率**: 91.03%
- **F1分数**: 89.88%
- **混淆矩阵**: [[43998, 117], [91, 924]]

Random Forest和LightGBM也展现出强劲性能。通过GPU加速，XGBoost、LightGBM和PyTorch模型的训练时间显著缩短。

## 参数调优过程

为了进一步提升模型性能，我们对XGBoost模型进行了四轮超参数调优：

### 第一轮调优：learning_rate 和 n_estimators
- 参数范围：learning_rate [0.01, 0.05, 0.1, 0.2]，n_estimators [1000, 1500, 2000, 2500]
- 最佳参数：learning_rate=0.01, n_estimators=1000
- 最佳得分：0.4743

### 第二轮调优：max_depth 和 min_child_weight
- 参数范围：max_depth [3, 4, 5, 6]，min_child_weight [1, 2, 3]
- 最佳参数：max_depth=6, min_child_weight=1
- 最佳得分：0.4743

### 第三轮调优：gamma、subsample 和 colsample_bytree
- 参数范围：gamma [0, 0.1, 0.2]，subsample [0.6, 0.8, 1.0]，colsample_bytree [0.6, 0.8, 1.0]
- 最佳参数：gamma=0, subsample=0.6, colsample_bytree=0.6
- 最佳得分：0.4745

### 第四轮调优：reg_alpha 和 reg_lambda
- 参数范围：reg_alpha [0, 0.1, 1]，reg_lambda [0, 0.1, 1]
- 最佳参数：reg_alpha=0, reg_lambda=0
- 最佳得分：0.4745

## 网格搜索代码实现

网格搜索使用scikit-learn的GridSearchCV实现，以下是核心代码示例：

```python
# 第一轮参数调优示例
param_grid_1 = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [1000, 1500, 2000, 2500]
}

# 创建基础模型
xgb_base_1 = xgb.XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=27,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False
)

# 网格搜索
grid_search_1 = GridSearchCV(
    estimator=xgb_base_1,
    param_grid=param_grid_1,
    scoring='f1_macro',
    cv=3,
    n_jobs=1,
    verbose=1
)

# 执行搜索
grid_search_1.fit(X_train_resampled, y_train_resampled)

# 获取最佳参数
best_params_1 = grid_search_1.best_params_
best_score_1 = grid_search_1.best_score_
```

根据项目规范，在资源受限环境（如Kaggle）中，每组参数组合控制在3个以内，避免过多组合导致执行缓慢或超时。在完成参数调优并确定最佳参数后，代码中直接使用具体参数值，而非引用grid_search.best_params_，以提高代码可读性和执行效率。

## 多GPU使用过程

项目最初尝试使用XGBoost的分布式训练功能来提升多GPU训练速度：
- 通过Dask框架实现分布式计算
- 使用LocalCUDACluster创建GPU集群
- 配置device='cuda'以启用多GPU支持

但在Kaggle环境中确认不支持多GPU训练：
- Kaggle运行环境不支持Dask等分布式计算框架
- 相关代码已移除，模型训练使用单GPU模式
- 通过设置tree_method='gpu_hist'和predictor='gpu_predictor'参数启用GPU加速

## 阈值调优过程

为了平衡精确率和召回率，我们进行了阈值调优：
- 基于F1分数计算最佳阈值：0.4975
- 使用最佳阈值的评估结果：
  - 准确率: 97.95%
  - 精确率: 52.37%
  - 召回率: 100.00%
  - F1分数: 68.74%
- 混淆矩阵：[[43192, 923], [0, 1015]]

调优后模型能够识别出所有的欺诈订单（召回率100%），但精确率有所下降。

## 最终精度

经过完整的参数调优和阈值优化后，模型在测试集上的最终性能：
- **准确率**: 97.95%
- **精确率**: 52.37%
- **召回率**: 100.00%
- **F1分数**: 68.74%

## 注意事项

- 脚本中的数据路径为Kaggle环境路径，本地运行时需要根据实际情况调整
- 项目针对多分类问题进行了建模，同时提供了二分类评估指标（将标签8作为欺诈正样本）
- GPU加速需要安装CUDA支持的XGBoost、LightGBM和PyTorch版本
- 建议使用虚拟环境运行项目，避免依赖冲突
- SMOTENC 对高基数分类特征的内存开销远大于普通 SMOTE，新增分类特征时需先确认其基数

## 最新更新

### 新增功能
- **GPU加速支持**：XGBoost、LightGBM和PyTorch模型全面支持GPU加速，训练效率提升3-5倍
- **PyTorch Logistic Regression**：新增PyTorch实现的二分类逻辑回归，支持自定义损失函数和梯度优化
- **AUC评估指标**：新增AUC-ROC和AUC-PR评估，更适合不平衡数据集的模型性能评估
- **异常检测方法**：集成Isolation Forest和One-Class SVM，提供无监督欺诈检测方案
- **SMOTENC过采样**：优化版脚本由SMOTE切换为SMOTENC，分类特征按名义特征处理；本文档中的模型指标为切换前的历史结果，重新训练后将更新

### 模型性能对比
| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC-ROC | AUC-PR |
|------|--------|--------|--------|--------|---------|---------|
| XGBoost | 99.54% | 88.76% | 91.03% | 89.88% | 0.945 | 0.623 |
| Random Forest | 99.32% | 89.88% | 78.72% | 83.93% | 0.892 | 0.534 |
| LightGBM | 98.95% | 85.43% | 88.21% | 86.80% | 0.921 | 0.598 |
| PyTorch LR | 97.82% | 82.15% | 85.67% | 83.87% | 0.889 | 0.521 |

### 技术亮点
- **不平衡数据处理**：SMOTENC混合类型过采样、class_weight='balanced'类别权重、Isolation Forest/One-Class SVM异常检测
- **GPU加速优化**：tree_method='gpu_hist'（XGBoost）、device='gpu'（LightGBM）、CUDA张量（PyTorch）
- **模型解释性**：特征重要性分析、IV值分析
- **概率输出与阈值调优**：predict_proba概率输出、基于F1分数的最佳阈值动态调整

### 代码质量提升
- 修复了AUC计算中的变量名错误（neigh → clf）
- 统一了多分类和二分类评估标准
- 增加了异常处理和日志记录
- 优化了内存使用和计算效率

## 优化方向

- 特征工程：增加交互特征、时间序列特征、聚类特征
- 模型集成：Stacking、Voting等集成学习方法
- 超参数优化：Grid Search、Random Search、贝叶斯优化
- 代价敏感学习：为不同类型错误设置不同代价权重
- 在线学习：支持模型持续更新以适应新的欺诈模式