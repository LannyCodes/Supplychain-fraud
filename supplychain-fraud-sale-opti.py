#==================================================
# Code cell 1
#==================================================

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#==================================================
# Code cell 2
#==================================================



#==================================================
# Code cell 3
#==================================================

input = '/kaggle/input/source/'  # 修改为正确的数据集路径
output = '/kaggle/working/'  # 输出路径保持不变

#==================================================
# Code cell 4
#==================================================

## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif']=[u'simHei']

# 在普通Python脚本中启用内联绘图
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

## 数据处理的
# 处理pandas-profiling的兼容性问题
try:
    import pandas_profiling as pp
except ImportError:
    try:
        from ydata_profiling import ProfileReport
        # 创建一个兼容的别名
        pp = ProfileReport
    except ImportError:
        print("未找到pandas-profiling或ydata-profiling库")
        pp = None

#缺失值可视化工具
import missingno as msno

## 模型预测的
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

## 处理数据不平衡
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
# 删除错误的导入
# from imblearn.combine import 

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score  # 添加AUC相关指标
from sklearn.pipeline import Pipeline

#==================================================
# Code cell 5
#==================================================

#数据加载
dataset=pd.read_csv(input+'SupplyChain.csv', encoding='unicode_escape')

# 打印前4个高价值特征的取值
print("========== 前4个高价值特征取值分析 ==========")
print("\nType字段取值:")
print(dataset['Type'].value_counts())

print("\nDelivery Status字段取值:")
print(dataset['Delivery Status'].value_counts())

if 'Late delivery risk' in dataset.columns:
    print("\nLate delivery risk字段取值:")
    print(dataset['Late delivery risk'].value_counts())
elif 'Late_delivery_risk' in dataset.columns:
    print("\nLate_delivery_risk字段取值:")
    print(dataset['Late_delivery_risk'].value_counts())

#==================================================
# Code cell 6
#==================================================

data = dataset.copy()

#==================================================
# Code cell 7
#==================================================

# 保留前4个高价值特征，删除其他不相关的特征
# 前4个高价值特征: Delivery Status, Type_Delivery_Cross, Type, Late_delivery_risk
# 先删除明显不相关的特征
unrelated_columns = [col for col in data.columns if col not in [
    'Type', 'Delivery Status', 'Late delivery risk', 'Late_delivery_risk', 'Order Status'
]]
data.drop(unrelated_columns, axis=1, inplace=True)

print("保留高价值特征后的数据形状:")
print(data.shape)

#==================================================
# Code cell 8
#==================================================

# 对缺失值进行简单处理
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype == 'object':
            data[column].fillna(data[column].mode()[0], inplace=True)
        else:
            data[column].fillna(data[column].median(), inplace=True)

#==================================================
# Code cell 9
#==================================================

# 转换格式 data['Type'].astype('category')
astype_columns = ['Type', 'Delivery Status', 'Order Status']

# 如果Late_delivery_risk列存在，也加入到类别列中
if 'Late delivery risk' in data.columns:
    astype_columns.append('Late delivery risk')
elif 'Late_delivery_risk' in data.columns:
    astype_columns.append('Late_delivery_risk')

# 添加Type和Delivery Status的交叉特征到astype_columns列表中
if 'Type_Delivery_Cross' not in astype_columns:
    astype_columns.append('Type_Delivery_Cross')

#==================================================
# Code cell 10
#==================================================

# y值
y_column = ['Order Status']

#==================================================
# Code cell 11
#==================================================

# 先创建交叉特征，再进行类别转换
# 添加Type和Delivery Status的交叉特征
print("构建Type和Delivery Status的交叉特征...")
# 保存原始的Type和Delivery Status用于交叉特征构建
data['Type_Delivery_Cross'] = data['Type'].astype(str) + '_' + data['Delivery Status'].astype(str)

# 将交叉特征也加入到类别列中
if 'Type_Delivery_Cross' not in astype_columns:
    astype_columns.append('Type_Delivery_Cross')

# 对所有类别特征进行转换
for column in astype_columns:
    data[column] = data[column].astype('category')

#==================================================
# Code cell 12
#==================================================

print("处理后的数据形状:")
print(data.shape)
temp = data.isnull().sum()
temp[temp>0]

#==================================================
# Code cell 13
#==================================================

data.info()

#==================================================
# Code cell 14
#==================================================

# 跳过时间特征处理，因为我们只保留前4个高价值特征

#==================================================
# Code cell 15
#==================================================

# 跳过时间特征处理，因为我们只保留前4个高价值特征

#==================================================
# Code cell 16
#==================================================

# 跳过时间特征处理，因为我们只保留前4个高价值特征

#==================================================
# Code cell 17
#==================================================

# 跳过时间特征处理，因为我们只保留前4个高价值特征

#==================================================
# Code cell 18
#==================================================

# 跳过时间特征处理，因为我们只保留前4个高价值特征

#==================================================
# Code cell 19
#==================================================

# 先创建交叉特征，再进行类别转换
# 添加Type和Delivery Status的交叉特征
print("构建Type和Delivery Status的交叉特征...")
# 保存原始的Type和Delivery Status用于交叉特征构建
data['Type_Delivery_Cross'] = data['Type'].astype(str) + '_' + data['Delivery Status'].astype(str)

# 添加TRANSFER+Shipping canceled精细化特征
print("构建TRANSFER+Shipping canceled精细化特征...")
# 创建高风险组合标识
data['High_Risk_TRANSFER_Cancel'] = ((data['Type'] == 'TRANSFER') & (data['Delivery Status'] == 'Shipping canceled')).astype(int)

# 添加Late_delivery_risk的概率化处理
if 'Late_delivery_risk' in data.columns:
    print("处理Late_delivery_risk特征...")
    # 如果该特征已经是0/1值，可以保持不变
    # 如果需要更精细的处理，可以基于其他特征计算动态风险值
    pass
elif 'Late delivery risk' in data.columns:
    # 重命名列以保持一致性
    data.rename(columns={'Late delivery risk': 'Late_delivery_risk'}, inplace=True)

# 将交叉特征也加入到类别列中
if 'Type_Delivery_Cross' not in astype_columns:
    astype_columns.append('Type_Delivery_Cross')

# 添加新的精细化特征到类别列中
if 'High_Risk_TRANSFER_Cancel' not in astype_columns:
    astype_columns.append('High_Risk_TRANSFER_Cancel')

# 对所有类别特征进行转换
for column in astype_columns:
    data[column] = data[column].astype('category')

#==================================================
# Code cell 20
#==================================================

print("处理后的数据形状:")
print(data.shape)
temp = data.isnull().sum()
temp[temp>0]

#==================================================
# Code cell 21
#==================================================

# data.select_dtypes(include=[object]).columns  
# data.select_dtypes(exclude=[object]).columns  
data.info()
data.select_dtypes(include=[object]).columns  

#==================================================
# Code cell 22
#==================================================

#  批量Labels encoding:

# preprocessing.LabelBinarizer
# preprocessing.LabelEncoder

data2 = data.copy()

str_cols = data2.select_dtypes(include=['category']).columns
clfs = {c:preprocessing.LabelEncoder() for c in str_cols}

for col, clf in clfs.items():
    data2[col] = clfs[col].fit_transform(data2[col])

# 标签反转演示
# for col, clf in clfs.items():
#     display(col, clfs[col].inverse_transform([0]))

#==================================================
# Code cell 23
#==================================================

# 标签反转演示
display("Order Status", clfs["Order Status"].inverse_transform([0,1,2,3,4,5,6,7,8]))

display("Order Status", clfs["Order Status"].inverse_transform([8]))

#==================================================
# Code cell 25
#==================================================

display(data['Order Status'].value_counts())
data['Order Status'].value_counts().plot.bar()

#==================================================
# Code cell 26
#==================================================

#  切分训练集、测试集

# # ### 删除不适用的特征
drop_features = ['Order Status']

# 重新对包含新特征的data2进行LabelEncoder编码
print("重新进行LabelEncoder编码，包含交叉特征...")
data2 = data.copy()
str_cols = data2.select_dtypes(include=['category']).columns
clfs = {c:preprocessing.LabelEncoder() for c in str_cols}

for col, clf in clfs.items():
    data2[col] = clfs[col].fit_transform(data2[col])

# 调试信息：检查原始数据和处理后数据的形状
print(f"原始数据形状: {data.shape}")
print(f"处理后数据形状: {data2.shape}")

# 更新特征列列表 - 只保留前4个高价值特征
# 前4个高价值特征: Delivery Status, Type_Delivery_Cross, Type, Late_delivery_risk
high_value_features = ['Delivery Status', 'Type_Delivery_Cross', 'Type', 'Late_delivery_risk']

# 检查这些特征是否在数据中存在
available_high_value_features = [col for col in high_value_features if col in data2.columns]
print(f"可用的高价值特征: {available_high_value_features}")

# 更新feature_cols只包含高价值特征
feature_cols = [column for column in data2.columns if column in available_high_value_features]

# 如果Late_delivery_risk列名不一致，检查其他可能的名称
if 'Late_delivery_risk' not in data2.columns and 'Late delivery risk' in data2.columns:
    # 替换特征列表中的名称
    feature_cols = [column for column in data2.columns if column in ['Delivery Status', 'Type_Delivery_Cross', 'Type', 'Late delivery risk']]
    print("使用 'Late delivery risk' 作为特征")

print(f"最终特征数量: {len(feature_cols)}")
print("特征列表:")
for i, col in enumerate(feature_cols):
    print(f"{i+1:2d}. {col}")

# 特别检查交叉特征是否被包含
if 'Type_Delivery_Cross' in feature_cols:
    print("\n✓ 交叉特征 'Type_Delivery_Cross' 已成功添加到特征集中")
else:
    print("\n✗ 交叉特征 'Type_Delivery_Cross' 未找到，请检查代码")
    # 调试信息：检查data中是否存在该列
    if 'Type_Delivery_Cross' in data.columns:
        print(f"  - 'Type_Delivery_Cross' 在data中存在")
    else:
        print(f"  - 'Type_Delivery_Cross' 在data中不存在")

# 全量数据 - 只使用高价值特征
X = data2[feature_cols]
y = data2['Order Status']

# 添加WOE和IV计算函数
def calculate_woe_iv(dataset, feature, target):
    """
    计算特征的WOE和IV值
    
    Parameters:
    dataset: 数据集
    feature: 特征名称
    target: 目标变量名称
    
    Returns:
    woe_df: 包含WOE值的DataFrame
    iv: 信息值
    """
    print(f"\n计算特征 '{feature}' 的WOE和IV值...")
    
    # 创建交叉表
    df = pd.crosstab(dataset[feature], dataset[target], margins=True)
    
    # 重命名列
    df.rename(columns={0: 'Non-Event', 1: 'Event'}, inplace=True)
    df.rename(index={'All': 'Total'}, inplace=True)
    
    # 计算分布比例
    df['Event_Rate'] = df['Event'] / df.loc['Total', 'Event']
    df['Non_Event_Rate'] = df['Non-Event'] / df.loc['Total', 'Non-Event']
    
    # 计算WOE
    df['WOE'] = np.log(df['Event_Rate'] / df['Non_Event_Rate'])
    
    # 处理无穷大值
    df['WOE'] = df['WOE'].replace([np.inf, -np.inf], 0)
    
    # 计算IV贡献
    df['IV'] = (df['Event_Rate'] - df['Non_Event_Rate']) * df['WOE']
    
    # 计算总IV值
    iv = df['IV'].sum()
    
    # 选择需要的列
    woe_df = df[['Event', 'Non-Event', 'Event_Rate', 'Non_Event_Rate', 'WOE', 'IV']].copy()
    
    return woe_df, iv

def calculate_all_features_iv(X, y, top_n=10):
    """
    计算所有特征的IV值并排序
    
    Parameters:
    X: 特征数据
    y: 目标变量（需要转换为二分类）
    top_n: 返回前N个特征
    
    Returns:
    iv_df: 包含所有特征IV值的DataFrame
    """
    # 将目标变量转换为二分类（8为欺诈，其他为正常）
    y_binary = y.apply(lambda x: 1 if x == 8 else 0)
    
    # 存储所有特征的IV值
    iv_values = []
    
    print("计算所有特征的IV值...")
    
    # 对每个特征计算IV值
    for feature in X.columns:
        try:
            # 对于连续特征，需要先分箱
            if X[feature].dtype in ['int64', 'float64']:
                # 使用分位数分箱
                X_temp = X[feature].copy()
                # 处理缺失值
                if X_temp.isnull().sum() > 0:
                    X_temp = X_temp.fillna(X_temp.median())
                
                # 分箱
                if X_temp.nunique() > 10:
                    # 如果唯一值超过10个，进行分箱
                    X_binned = pd.qcut(X_temp, q=10, duplicates='drop')
                else:
                    # 如果唯一值较少，直接使用原始值
                    X_binned = X_temp
                
                woe_df, iv = calculate_woe_iv(pd.DataFrame({feature: X_binned, 'target': y_binary}), feature, 'target')
            else:
                # 对于分类特征，直接计算
                woe_df, iv = calculate_woe_iv(pd.DataFrame({feature: X[feature], 'target': y_binary}), feature, 'target')
            
            iv_values.append({'Feature': feature, 'IV': iv})
            print(f"特征 '{feature}' 的IV值: {iv:.4f}")
            
        except Exception as e:
            print(f"计算特征 '{feature}' 的IV值时出错: {e}")
            iv_values.append({'Feature': feature, 'IV': 0})
    
    # 创建IV值DataFrame并排序
    iv_df = pd.DataFrame(iv_values)
    iv_df = iv_df.sort_values('IV', ascending=False).reset_index(drop=True)
    
    print(f"\n前{top_n}个最具预测能力的特征:")
    print(iv_df.head(top_n))
    
    return iv_df

# 转换为二分类，需要修改模型
# "SUSPECTED_FRAUD" --> 8
# y_2 = y.apply(lambda x : 1 if x == 8 else 0).copy()

# 使用 stratify 保证训练集和测试集中类别比例一致
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=2021, stratify=y)

# 使用 SMOTE 对训练集进行过采样，平衡类别
print("原始训练集类别分布:")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=2021, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nSMOTE后训练集类别分布:")
print(pd.Series(y_train_resampled).value_counts())
print(f"\n原始训练集样本数: {len(y_train)}")
print(f"过采样后训练集样本数: {len(y_train_resampled)}")
print(f"增加的样本数: {len(y_train_resampled) - len(y_train)}")

"""
# RandomForestClassifier
print('\n========== RandomForestClassifier 模型评估 ==========')
from sklearn.ensemble import RandomForestClassifier
# 添加 class_weight='balanced' 处理不平衡数据
clf = RandomForestClassifier(
    max_depth=10,               # 增加树的深度
    n_estimators=200,           # 增加树的数量
    random_state=2021, 
    class_weight='balanced',    # 处理不平衡数据
    min_samples_split=5,        # 增加分割所需的最小样本数
    min_samples_leaf=2,         # 增加叶节点最小样本数
    max_features='sqrt'         # 限制特征数量
)
clf.fit(X_train_resampled, y_train_resampled)

y_pred = clf.predict(X_test)

# 打印特征重要性
print("\n特征重要性 (RandomForest):")
feature_importance_rf = clf.feature_importances_
feature_names = X_train.columns
feature_importance_df_rf = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_rf
}).sort_values(by='importance', ascending=False)
print(feature_importance_df_rf.head(20))

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred).value_counts() )
y_pred_2 = pd.Series(y_pred).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# 混淆矩阵
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
m = confusion_matrix(y_test_2, y_pred_2)
print('\n混淆矩阵：')
print(m)

# 准确率
print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2, y_pred_2):.4f}")

# 精确率、召回率、F1分数（针对欺诈类别）
print(f"精确率 (Precision): {precision_score(y_test_2, y_pred_2):.4f}")
print(f"召回率 (Recall): {recall_score(y_test_2, y_pred_2):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test_2, y_pred_2):.4f}")

# AUC-PR和AUC-ROC指标（对不平衡数据更敏感）
try:
    # 获取预测概率
    y_proba_fraud = None
    
    # 针对不同模型获取欺诈类别的概率
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)
        # 对于多分类问题，我们需要转换为二分类概率
        # 获取欺诈类别（标签8）的概率
        if hasattr(clf, "classes_"):
            fraud_class_index = list(clf.classes_).index(8) if 8 in clf.classes_ else -1
            if fraud_class_index >= 0:
                y_proba_fraud = y_proba[:, fraud_class_index]
    
    # 计算AUC指标
    if y_proba_fraud is not None:
        auc_roc = roc_auc_score(y_test_2, y_proba_fraud)
        auc_pr = average_precision_score(y_test_2, y_proba_fraud)
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
    else:
        print("模型不支持预测概率或未找到欺诈类别")
        
except Exception as e:
    print(f"计算AUC指标时出错: {e}")

# 分类报告
print('\n分类报告：')
print(classification_report(y_test_2, y_pred_2, target_names=['正常订单', '欺诈订单']))

#==================================================
# Code cell 32
#==================================================
"""


print('\n========== XGBClassifier 模型评估 ==========')

#XGBClassifier 
# 计算类别权重比例用于 XGBoost
# 针对多分类问题，使用平衡后的数据或保持原数据但不设置scale_pos_weight
xgr = xgb.XGBClassifier(
    learning_rate=0.05,         # 降低学习率
    n_estimators=2000,          # 增加树的数量
    max_depth=8,                # 适当增加深度
    min_child_weight=3,         # 增加子节点最小权重
    gamma=0.1,                  # 增加正则化
    subsample=0.7,              # 减少样本采样比例
    colsample_bytree=0.7,       # 减少特征采样比例
    objective='multi:softmax',
    eval_metric='mlogloss',     # 显式指定评估指标，避免警告
    random_state=27,
    tree_method='gpu_hist',     # 使用GPU加速
    predictor='gpu_predictor'   # 使用GPU进行预测
)

# 使用 SMOTE 平衡后的数据训练
xgr.fit(X_train_resampled, y_train_resampled)
y_pred = xgr.predict(X_test)

# 添加超参数调优代码
print("\n========== XGBoost超参数调优 ==========")
from sklearn.model_selection import GridSearchCV

# 第一组参数调优：learning_rate 和 n_estimators
param_grid_1 = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [1000, 1500, 2000, 2500]
}

# 创建基础模型用于调优
xgb_base = xgb.XGBClassifier(
    max_depth=8,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=27,
    tree_method='gpu_hist',
    predictor='gpu_predictor'
)

# 执行网格搜索
print("执行第一组参数调优 (learning_rate 和 n_estimators)...")
grid_search_1 = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid_1,
    scoring='f1_macro',  # 使用F1分数作为评估指标
    cv=3,  # 3折交叉验证
    n_jobs=-1,  # 使用所有CPU核心
    verbose=1
)

# 由于数据量较大，我们使用部分数据进行调优以节省时间
# 使用20%的训练数据进行快速调优
sample_size = int(0.2 * len(X_train_resampled))
X_train_sample = X_train_resampled[:sample_size]
y_train_sample = y_train_resampled[:sample_size]

grid_search_1.fit(X_train_sample, y_train_sample)

print("第一组参数调优完成!")
print(f"最佳参数: {grid_search_1.best_params_}")
print(f"最佳得分: {grid_search_1.best_score_:.4f}")

# 使用最佳参数重新训练模型
print("使用最佳参数重新训练模型...")
xgr_optimized = xgb.XGBClassifier(
    learning_rate=grid_search_1.best_params_['learning_rate'],
    n_estimators=grid_search_1.best_params_['n_estimators'],
    max_depth=8,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=27,
    tree_method='gpu_hist',
    predictor='gpu_predictor'
)

xgr_optimized.fit(X_train_resampled, y_train_resampled)
y_pred_optimized = xgr_optimized.predict(X_test)

# 计算所有特征的IV值
print("\n========== 特征IV值分析 ==========")
iv_df = calculate_all_features_iv(X_train, y_train, top_n=20)

# 打印特征重要性
print("\n特征重要性 (XGBoost):")
feature_importance_xgb = xgr_optimized.feature_importances_
feature_names = X_train.columns
feature_importance_df_xgb = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_xgb
}).sort_values(by='importance', ascending=False)
print(feature_importance_df_xgb.head(20))

# 打印前3个最重要特征的取值分布
print("\n========== 前3个最重要特征的取值分布 ==========")
top_3_features = feature_importance_df_xgb.head(3)['feature'].tolist()
for feature in top_3_features:
    print(f"\n{feature} 特征取值分布:")
    if feature in data.columns:
        print(data[feature].value_counts().sort_values(ascending=False))
    else:
        # 如果特征名在原始数据中不存在，尝试在编码后的数据中查找
        print("特征取值需要查看编码后的数据...")

# 深入分析前3个最重要特征对欺诈结果的影响
print("\n========== 前3个最重要特征的欺诈率分析 ==========")
# 重建原始数据和目标变量的对应关系（使用未过采样的数据）
X_train_original = X_train
y_train_original = y_train

# 创建包含原始特征和目标变量的数据框
train_data_with_target = X_train_original.copy()
train_data_with_target['Order_Status'] = y_train_original

for feature in top_3_features:
    print(f"\n{feature} 特征的欺诈率分析:")
    if feature in data.columns:
        # 获取原始数据中的特征值
        feature_values = data.loc[X_train_original.index, feature]
        
        # 计算每个特征值的欺诈率
        fraud_analysis = pd.DataFrame({
            'feature_value': feature_values,
            'order_status': y_train_original
        })
        
        # 计算每个特征值的总数量和欺诈数量
        fraud_stats = fraud_analysis.groupby('feature_value').agg({
            'order_status': ['count', lambda x: sum(x == 8)]
        }).round(4)
        
        # 重命名列
        fraud_stats.columns = ['total_count', 'fraud_count']
        fraud_stats['fraud_rate'] = fraud_stats['fraud_count'] / fraud_stats['total_count']
        
        # 按欺诈率排序
        fraud_stats = fraud_stats.sort_values('fraud_rate', ascending=False)
        print(fraud_stats)
    else:
        print("无法找到原始特征值进行分析...")

# 交叉分析：特征与目标变量的关系
print("\n========== 前3个最重要特征与Order Status的交叉分析 ==========")
for feature in top_3_features:
    print(f"\n{feature} 与 Order Status 的交叉分析:")
    if feature in data.columns:
        cross_tab = pd.crosstab(data.loc[X_train_original.index, feature], 
                               y_train_original, 
                               normalize='index').round(4)
        # 只显示欺诈类别(8)的比例，并按比例排序
        fraud_proportion = cross_tab[8].sort_values(ascending=False)
        print(fraud_proportion)
    else:
        print("无法进行交叉分析...")

### plot feature importance
fig,ax = plt.subplots(figsize=(15,15))
xgb.plot_importance(xgr_optimized,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred_optimized).value_counts() )
y_pred_2 = pd.Series(y_pred_optimized).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# 混淆矩阵
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
m = confusion_matrix(y_test_2, y_pred_2)
print('\n混淆矩阵：')
print(m)

# 准确率
print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2, y_pred_2):.4f}")

# 精确率、召回率、F1分数（针对欺诈类别）
print(f"精确率 (Precision): {precision_score(y_test_2, y_pred_2):.4f}")
print(f"召回率 (Recall): {recall_score(y_test_2, y_pred_2):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test_2, y_pred_2):.4f}")

# 分类报告
print('\n分类报告：')
print(classification_report(y_test_2, y_pred_2, target_names=['正常订单', '欺诈订单']))

# AUC-PR和AUC-ROC指标（对不平衡数据更敏感）
try:
    # 获取预测概率
    y_proba_fraud = None
    
    # 针对不同模型获取欺诈类别的概率
    if hasattr(xgr_optimized, "predict_proba"):
        y_proba = xgr_optimized.predict_proba(X_test)
        # 对于多分类问题，我们需要转换为二分类概率
        # 获取欺诈类别（标签8）的概率
        if hasattr(xgr_optimized, "classes_"):
            fraud_class_index = list(xgr_optimized.classes_).index(8) if 8 in xgr_optimized.classes_ else -1
            if fraud_class_index >= 0:
                y_proba_fraud = y_proba[:, fraud_class_index]
    
    # 计算AUC指标
    if y_proba_fraud is not None:
        y_test_binary_auc = y_test.apply(lambda x: 1 if x == 8 else 0)
        auc_roc = roc_auc_score(y_test_binary_auc, y_proba_fraud)
        auc_pr = average_precision_score(y_test_binary_auc, y_proba_fraud)
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
    else:
        print("模型不支持预测概率或未找到欺诈类别")
        
except Exception as e:
    print(f"计算AUC指标时出错: {e}")

#==================================================
# Code cell 33
#==================================================
"""
print('\n========== LightGBM 模型评估 ==========')

# LightGBM
# 使用 SMOTE 平衡后的数据训练
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=127,             # 增加叶子节点数
    max_depth=8,                # 增加最大深度
    learning_rate=0.05,         # 降低学习率
    n_estimators=500,           # 增加树的数量
    random_state=27,
    class_weight='balanced',
    min_child_samples=20,       # 增加子节点最小样本数
    min_child_weight=0.001,     # 增加子节点最小权重
    subsample=0.8,              # 样本采样比例
    colsample_bytree=0.8,       # 特征采样比例
    reg_alpha=0.1,              # L1正则化
    reg_lambda=0.1,             # L2正则化
    device='gpu'                # 启用GPU支持
)

# 添加异常处理来捕获训练过程中的错误
try:
    lgb_model.fit(X_train_resampled, y_train_resampled)
    y_pred_lgb = lgb_model.predict(X_test)
    
    # 打印特征重要性
    print("\n特征重要性 (LightGBM):")
    feature_importance = lgb_model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)
    print(feature_importance_df.head(20))
    
    # 转换为二分类标签
    y_test_2_lgb = y_test.apply(lambda x : 1 if x == 8 else 0).copy()
    y_pred_2_lgb = pd.Series(y_pred_lgb).apply(lambda x : 1 if x == 8 else 0).copy()
    
    # 混淆矩阵
    m_lgb = confusion_matrix(y_test_2_lgb, y_pred_2_lgb)
    print('\n混淆矩阵：')
    print(m_lgb)
    
    # 准确率
    print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
    
    # 精确率、召回率、F1分数（针对欺诈类别）
    print(f"精确率 (Precision): {precision_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
    
    # 分类报告
    print('\n分类报告：')
    print(classification_report(y_test_2_lgb, y_pred_2_lgb, target_names=['正常订单', '欺诈订单']))
    
    # AUC-PR和AUC-ROC指标（对不平衡数据更敏感）
    try:
        # 获取预测概率
        y_proba_fraud_lgb = None
        
        # 针对不同模型获取欺诈类别的概率
        if hasattr(lgb_model, "predict_proba"):
            y_proba_lgb = lgb_model.predict_proba(X_test)
            # 对于多分类问题，我们需要转换为二分类概率
            # 获取欺诈类别（标签8）的概率
            if hasattr(lgb_model, "classes_"):
                fraud_class_index_lgb = list(lgb_model.classes_).index(8) if 8 in lgb_model.classes_ else -1
                if fraud_class_index_lgb >= 0:
                    y_proba_fraud_lgb = y_proba_lgb[:, fraud_class_index_lgb]
        
        # 计算AUC指标
        if y_proba_fraud_lgb is not None:
            y_test_binary_auc_lgb = y_test.apply(lambda x: 1 if x == 8 else 0)
            auc_roc_lgb = roc_auc_score(y_test_binary_auc_lgb, y_proba_fraud_lgb)
            auc_pr_lgb = average_precision_score(y_test_binary_auc_lgb, y_proba_fraud_lgb)
            print(f"AUC-ROC: {auc_roc_lgb:.4f}")
            print(f"AUC-PR: {auc_pr_lgb:.4f}")
        else:
            print("模型不支持预测概率或未找到欺诈类别")
            
    except Exception as e:
        print(f"计算AUC指标时出错: {e}")

except Exception as e:
    print(f"LightGBM模型训练时出错: {e}")
    print("可能的原因：数据质量问题、类别不平衡、特征质量等")
    # 尝试使用CPU版本作为备选方案
    try:
        print("尝试使用CPU版本...")
        lgb_model_cpu = lgb.LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=127,             # 增加叶子节点数
            max_depth=8,                # 增加最大深度
            learning_rate=0.05,         # 降低学习率
            n_estimators=500,           # 增加树的数量
            random_state=27,
            class_weight='balanced',
            min_child_samples=20,       # 增加子节点最小样本数
            min_child_weight=0.001,     # 增加子节点最小权重
            subsample=0.8,              # 样本采样比例
            colsample_bytree=0.8,       # 特征采样比例
            reg_alpha=0.1,              # L1正则化
            reg_lambda=0.1              # L2正则化
            # 不指定device参数，使用默认CPU
        )
        lgb_model_cpu.fit(X_train_resampled, y_train_resampled)
        y_pred_lgb = lgb_model_cpu.predict(X_test)
        
        # 打印特征重要性
        print("\n特征重要性 (LightGBM CPU版本):")
        feature_importance = lgb_model_cpu.feature_importances_
        feature_names = X_train.columns
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values(by='importance', ascending=False)
        print(feature_importance_df.head(20))
        
        # 转换为二分类标签
        y_test_2_lgb = y_test.apply(lambda x : 1 if x == 8 else 0).copy()
        y_pred_2_lgb = pd.Series(y_pred_lgb).apply(lambda x : 1 if x == 8 else 0).copy()
        
        # 混淆矩阵
        m_lgb = confusion_matrix(y_test_2_lgb, y_pred_2_lgb)
        print('\n混淆矩阵（CPU版本）：')
        print(m_lgb)
        
        # 准确率
        print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
        
        # 精确率、召回率、F1分数（针对欺诈类别）
        print(f"精确率 (Precision): {precision_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
        print(f"召回率 (Recall): {recall_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
        print(f"F1分数 (F1-Score): {f1_score(y_test_2_lgb, y_pred_2_lgb):.4f}")
        
        print("CPU版本运行成功")
    except Exception as cpu_e:
        print(f"CPU版本也失败: {cpu_e}")
"""
"""
# 导入PyTorch逻辑回归模型
try:
    from logistic_regression_pytorch import train_binary_logistic_regression_pytorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("无法导入PyTorch逻辑回归模型，请确保logistic_regression_pytorch.py文件在同一目录下")

# PyTorch Binary Logistic Regression
if PYTORCH_AVAILABLE:
    print('\n========== PyTorch Binary Logistic Regression 模型评估 ==========')
    
    try:
        # 将数据转换为二分类问题
        y_train_binary = y_train_resampled.apply(lambda x: 1 if x == 8 else 0).values  # 使用SMOTE过采样后的训练数据
        y_test_binary = y_test.apply(lambda x: 1 if x == 8 else 0).values
        
        # 数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)  # 使用SMOTE过采样后的训练数据
        X_test_scaled = scaler.transform(X_test)
        
        # 调用PyTorch二分类LR模型，调整参数以改善训练效果
        model, predictions, probabilities = train_binary_logistic_regression_pytorch(
            X_train_scaled, y_train_binary, X_test_scaled, y_test_binary, 
            num_epochs=2000, learning_rate=0.001  # 增加训练轮数，降低学习率
        )
        
        # 打印特征重要性（PyTorch逻辑回归的权重）
        print("\n特征重要性 (PyTorch Logistic Regression):")
        feature_names = X_train.columns
        # 获取模型的权重
        if hasattr(model, 'linear'):
            weights = model.linear.weight.data.cpu().numpy().flatten()
            feature_importance_df_pytorch = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(weights)  # 使用权重的绝对值作为重要性
            }).sort_values(by='importance', ascending=False)
            print(feature_importance_df_pytorch.head(20))
        
        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
        accuracy = accuracy_score(y_test_binary, predictions)
        precision = precision_score(y_test_binary, predictions, zero_division=0)
        recall = recall_score(y_test_binary, predictions, zero_division=0)
        f1 = f1_score(y_test_binary, predictions, zero_division=0)
        
        print('\n混淆矩阵：')
        print(confusion_matrix(y_test_binary, predictions))
        print(f"\n准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1:.4f}")
        
        # 计算AUC-PR指标
        if probabilities is not None:
            auc_pr = average_precision_score(y_test_binary, probabilities)
            print(f"AUC-PR: {auc_pr:.4f}")
            
        # 分类报告
        print('\n分类报告：')
        print(classification_report(y_test_binary, predictions, target_names=['正常订单', '欺诈订单']))
        
    except Exception as e:
        print(f"PyTorch模型训练时出错: {e}")
else:
    print('\n========== PyTorch Binary Logistic Regression 模型评估 ==========')
    print("PyTorch不可用，跳过PyTorch逻辑回归模型训练")
"""

#==================================================
# Code cell 34
#==================================================

#  模型调优
# 交叉验证，
# 网格搜索


"""
# 查看当前类别分布
print("原始训练集类别分布:")
print(pd.Series(y_train).value_counts())
print("\nSMOTE后训练集类别分布:")
print(pd.Series(y_train_resampled).value_counts())


# 4. 尝试异常检测方法
print("\n========== 尝试异常检测方法 ==========")
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    
    # 准备二分类数据（正常订单 vs 欺诈订单）
    # 将标签8（欺诈）标记为-1，其他标记为1
    y_binary = y_train.apply(lambda x: -1 if x == 8 else 1)
    
    # Isolation Forest
    print("\n使用Isolation Forest进行异常检测...")
    iso_forest = IsolationForest(contamination=0.1, random_state=2021)
    iso_forest.fit(X_train)
    y_pred_iso = iso_forest.predict(X_test)
    
    # 转换预测结果以匹配评估格式
    y_test_binary = y_test.apply(lambda x: -1 if x == 8 else 1)
    y_pred_iso_binary = pd.Series(y_pred_iso).apply(lambda x: 1 if x == 1 else 0)
    y_test_binary_eval = y_test_binary.apply(lambda x: 1 if x == -1 else 0)
    
    print('\n========== Isolation Forest 模型评估 ==========')
    print('\n混淆矩阵：')
    print(confusion_matrix(y_test_binary_eval, y_pred_iso_binary))
    print(f"\n准确率 (Accuracy): {accuracy_score(y_test_binary_eval, y_pred_iso_binary):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test_binary_eval, y_pred_iso_binary):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_binary_eval, y_pred_iso_binary):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_binary_eval, y_pred_iso_binary):.4f}")
    
except ImportError:
    print("异常检测方法所需的库不可用")

"""
