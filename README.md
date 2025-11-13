# Supply Chain Fraud Detection

本项目旨在检测供应链中的欺诈行为，通过对供应链数据进行分析和建模，识别潜在的欺诈订单。

## 项目概述

该项目使用机器学习方法对供应链交易数据进行分析，检测异常模式以识别欺诈行为。项目包含数据预处理、特征工程、模型训练和评估等完整流程。

## 数据集

项目使用供应链数据集 `SupplyChain.csv`，包含约18万条订单记录和53个特征字段。

## 主要功能

1. **数据清洗与预处理**
   - 处理缺失值
   - 数据类型转换
   - 时间特征提取

2. **特征工程**
   - 从订单日期和发货日期中提取多维度时间特征（年、月、星期、小时）
   - 类别特征编码

3. **数据不平衡处理**
   - 使用SMOTE算法对训练集进行过采样，平衡各类别分布

4. **模型训练与评估**
   - 实现多种机器学习算法进行欺诈检测：
     - Gaussian Naive Bayes
     - Linear SVC
     - K-Nearest Neighbors
     - Linear Discriminant Analysis
     - Decision Tree
     - Random Forest
     - XGBoost
     - Logistic Regression
   - 模型评估指标包括准确率、精确率、召回率和F1分数

## 环境依赖

项目依赖库列表请参见 [requirements.txt](file:///Users/pro/Downloads/Supplychain-fraud/requirements.txt) 文件。

使用以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 将数据集 `SupplyChain.csv` 放置在 `/kaggle/input/source/` 目录下（或修改脚本中的路径）
2. 运行 [supplychain-fraud-sale.py](file:///Users/pro/Downloads/Supplychain-fraud/supplychain-fraud-sale.py) 脚本或在 Jupyter Notebook 中逐个执行代码单元

## 结果

项目实现了多种机器学习模型对供应链欺诈的检测，其中XGBoost和Random Forest表现较好。

## 注意事项

- 脚本中的数据路径为Kaggle环境路径，本地运行时需要根据实际情况调整
- 项目针对多分类问题进行了建模，同时提供了二分类评估指标（将"SUSPECTED_FRAUD"类标记为正样本）