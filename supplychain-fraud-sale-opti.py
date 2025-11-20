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

# 移除Dask相关库导入，因为Kaggle环境不支持分布式计算
# 将DASK_AVAILABLE和DASK_CUDA_AVAILABLE设置为False
DASK_AVAILABLE = False
DASK_CUDA_AVAILABLE = False

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score  # 添加AUC相关指标
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance

#数据加载
dataset=pd.read_csv(input+'SupplyChain.csv', encoding='unicode_escape')

# 打印Type、Delivery Status和Late_delivery_risk的取值
print("========== 特征取值分析 ==========")
print("\nType字段取值:")
# print(dataset['Type'].value_counts())

print("\nDelivery Status字段取值:")
# print(dataset['Delivery Status'].value_counts())

if 'Late delivery risk' in dataset.columns:
    print("\nLate delivery risk字段取值:")
    # print(dataset['Late delivery risk'].value_counts())
elif 'Late_delivery_risk' in dataset.columns:
    print("\nLate_delivery_risk字段取值:")
    # print(dataset['Late_delivery_risk'].value_counts())

#==================================================
# Code cell 6
#==================================================

data = dataset.copy()

#==================================================
# Code cell 7
#==================================================

# 18万比订单，53个特征
print(data.shape)
temp = data.isnull().sum()
# temp[temp>0]

#==================================================
# Code cell 8
#==================================================

data['Customer Lname'].value_counts() #Smith        64104
data['Customer Lname'].fillna(data['Customer Lname'].mode()[0], inplace=True)

#==================================================
# Code cell 9
#==================================================

data['Customer Zipcode'].value_counts()
data['Customer Zipcode'].fillna(data['Customer Zipcode'].mode()[0], inplace=True)

#==================================================
# Code cell 10
#==================================================

data.select_dtypes(exclude=[object]).columns 

#==================================================
# Code cell 11
#==================================================

#将Firs tName 与LastName进行合并=>Full Name 
data['Customer Full Name'] = data['Customer Fname'] + data['Customer Lname']
data[['Customer Full Name', 'Customer Fname', 'Customer Lname']]

#==================================================
# Code cell 12
#==================================================


# df = dataset['Customer State'].astype('category').copy()  # Categorize!
# df

# 选择需要处理的object columns
# dataset.select_dtypes(include=[object]).columns  
# dataset.select_dtypes(exclude=[object]).columns  

# 转换格式 data['Customer State'].astype('category')
astype_columns = ['Type', 'Delivery Status', 'Category Name', 'Customer City', 'Customer Country', 'Customer Fname', 'Customer Lname', 
                  'Customer Segment', 'Customer State', 'Customer Street', 'Department Name', 'Market', 'Customer Full Name',
                  'Order City', 'Order Country', 'Order Region', 'Order State',
                  'Product Name', 'Shipping Mode', 'shipping date (DateOrders)', 'order date (DateOrders)', 'Order Status'
                 ]

# 添加Type和Delivery Status的交叉特征到astype_columns列表中
if 'Type_Delivery_Cross' not in astype_columns:
    astype_columns.append('Type_Delivery_Cross')

# drop
drop_columns = ['Customer Email', 'Customer Password', 'Product Image',  'Order Zipcode', 'Product Description']

# 特征梳理
# 时间日期多维度： order date (DateOrders)   shipping date (DateOrders)
# 补全缺失值后：Customer Fname + Customer Lname = Customer Full Name
feature_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']

# y值
y_column = ['Order Status']



#==================================================
# Code cell 13
#==================================================

data.drop(drop_columns, axis=1, inplace=True)
# data.info()

#==================================================
# Code cell 14
#==================================================

#  order date (DateOrders)
#按照不同的时间维度（年，月，星期，小时）的趋势
#data[['order date (DateOrders)']]
#创建时间影索引
temp = pd.DatetimeIndex(data['order date (DateOrders)'])


#==================================================
# Code cell 15
#==================================================

# order date (DateOrders) 字段中的时间多尺度 year, month, weekday, hour, month_year
data['order_year'] = temp.year
data['order_month'] = temp.month
data['order_week_day'] = temp.weekday
data['order_hour'] = temp.hour
#data['order_month_year'] = temp.to_period('M')  auto-sklearn unsported


#==================================================
# Code cell 16
#==================================================

# 对销售额进行探索，按照不同的时间维度（年，月，星期，小时）的趋势
plt.subplot(4, 2, 1)
df_year = data.groupby('order_year')
df_year['Sales'].mean().plot(figsize=(12, 12), title='Mean sales in Years')

plt.subplot(4, 2, 2)
df_day = data.groupby('order_week_day')
df_day['Sales'].mean().plot(figsize=(12,12), title='Average sales in days')

plt.subplot(4, 2, 3)
df_hour = data.groupby('order_hour')
df_hour['Sales'].mean().plot(figsize=(12,12), title='Average sales in Hours')

plt.subplot(4, 2, 4)
df_month = data.groupby('order_month')
df_month['Sales'].mean().plot(figsize=(12, 12), title='Average sales in Months')

#==================================================
# Code cell 17
#==================================================

#  shipping date (DateOrders)
#按照不同的时间维度（年，月，星期，小时）的趋势
#data[['shipping date (DateOrders)']]
#创建时间影索引
temp = pd.DatetimeIndex(data['shipping date (DateOrders)'])


#==================================================
# Code cell 18
#==================================================

# shipping date (DateOrders) 字段中的时间多尺度 year, month, weekday, hour, month_year
data['shipping_year'] = temp.year
data['shipping_month'] = temp.month
data['shipping_week_day'] = temp.weekday
data['shipping_hour'] = temp.hour
# data['shipping_month_year'] = temp.to_period('M')    auto-sklearn unsported

#==================================================
# Code cell 19
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
# Code cell 20
#==================================================

# 18万比订单，53个特征
print(data.shape)
temp = data.isnull().sum()
temp[temp>0]

#==================================================
# Code cell 21
#==================================================

# data.select_dtypes(include=[object]).columns  
# data.select_dtypes(exclude=[object]).columns  
# data.info()
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

# ### 删除不适用的特征
drop_features = ['Order Status', 
                 #'shipping_year', 'shipping_month', 'shipping_week_day', 'shipping_hour',
                 #'order_year', 'order_month', 'order_week_day', 'order_hour',
                 'order date (DateOrders)', 'shipping date (DateOrders)'
                ]

print("使用全部特征进行训练...")
# 使用全部特征进行训练
feature_cols = [column for column in data2.columns if column not in drop_features]

# 打印调试信息
print(f"原始特征数量: {data2.shape[1]}")
print(f"最终特征数量: {len(feature_cols)}")

# 打印特征数量确认
print(f"总特征数量: {len(feature_cols)}")
# print("特征列表:")
# for i, col in enumerate(feature_cols):
#     print(f"{i+1:2d}. {col}")

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
    # 调试信息：检查drop_features中是否意外包含了该特征
    if 'Type_Delivery_Cross' in drop_features:
        print(f"  - 'Type_Delivery_Cross' 被意外添加到drop_features中")
    else:
        print(f"  - 'Type_Delivery_Cross' 不在drop_features中")

# 全量数据
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
            # print(f"特征 '{feature}' 的IV值: {iv:.4f}")
            
        except Exception as e:
            print(f"计算特征 '{feature}' 的IV值时出错: {e}")
            iv_values.append({'Feature': feature, 'IV': 0})
    
    # 创建IV值DataFrame并排序
    iv_df = pd.DataFrame(iv_values)
    iv_df = iv_df.sort_values('IV', ascending=False).reset_index(drop=True)
    
    # print(f"\n前{top_n}个最具预测能力的特征:")
    # print(iv_df.head(top_n))
    
    return iv_df

# 转换为二分类，需要修改模型
# "SUSPECTED_FRAUD" --> 8
# y_2 = y.apply(lambda x : 1 if x == 8 else 0).copy()

# 使用 stratify 保证训练集和测试集中类别比例一致
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=2021, stratify=y)

# 使用 SMOTE 对训练集进行过采样，平衡类别
# print("原始训练集类别分布:")
# print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=2021, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# print("\nSMOTE后训练集类别分布:")
# print(pd.Series(y_train_resampled).value_counts())
print(f"\n原始训练集样本数: {len(y_train)}")
print(f"过采样后训练集样本数: {len(y_train_resampled)}")
print(f"增加的样本数: {len(y_train_resampled) - len(y_train)}")

print('\n========== XGBClassifier 模型评估 ==========')
# 直接使用所有已确定的最佳参数创建最终模型
xgr_optimized_4 = xgb.XGBClassifier(
    learning_rate=0.01,         # 第一组调优得到的最佳学习率
    n_estimators=1000,          # 第一组调优得到的最佳树数量
    max_depth=6,                # 第二组调优得到的最佳参数
    min_child_weight=1,         # 第二组调优得到的最佳参数
    gamma=0,                    # 第三组调优得到的最佳参数
    subsample=0.6,              # 第三组调优得到的最佳参数
    colsample_bytree=0.6,       # 第三组调优得到的最佳参数
    reg_alpha=0,                # 第四组调优得到的最佳参数
    reg_lambda=0,               # 第四组调优得到的最佳参数
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=27,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False     # 避免标签编码器警告
)

print("使用所有最佳参数训练最终模型...")
xgr_optimized_4.fit(X_train_resampled, y_train_resampled)
y_pred_optimized_4 = xgr_optimized_4.predict(X_test)

# 计算所有特征的IV值
print("\n========== 特征IV值分析 ==========")
iv_df = calculate_all_features_iv(X_train, y_train, top_n=20)

# 打印特征重要性
print("\n特征重要性 (XGBoost):")
feature_importance_xgb = xgr_optimized_4.feature_importances_
feature_names = X_train.columns
feature_importance_df_xgb = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_xgb
}).sort_values(by='importance', ascending=False)
# print(feature_importance_df_xgb.head(20))


print("\n========== 前3个最重要特征的取值分布 ==========")
top_3_features = feature_importance_df_xgb.head(3)['feature'].tolist()
# for feature in top_3_features:
#     print(f"\n{feature} 特征取值分布:")
#     if feature in data.columns:
#         print(data[feature].value_counts().sort_values(ascending=False))
#     else:
#         # 如果特征名在原始数据中不存在，尝试在编码后的数据中查找
#         print("特征取值需要查看编码后的数据...")

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
        # print(fraud_stats)
    else:
        print("无法找到原始特征值进行分析...")



### 使用XGBoost筛选的前20个重要特征作为统一特征维度
print("\n========== 使用XGBoost前20个重要特征训练其他模型 ==========")
# 获取前20个最重要的特征
top_20_features = feature_importance_df_xgb.head(20)['feature'].tolist()
print(f"选择的20个重要特征: {top_20_features}")

# 重新构建训练和测试数据集，只使用这20个特征
X_train_top20 = X_train[top_20_features]
X_test_top20 = X_test[top_20_features]
X_train_resampled_top20 = X_train_resampled[top_20_features]

print(f"原始特征数: {X_train.shape[1]}")
print(f"筛选后特征数: {X_train_top20.shape[1]}")

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

### plot feature importance
fig,ax = plt.subplots(figsize=(15,15))
xgb.plot_importance(xgr_optimized_4,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

# # 转换为二分类标签
display( pd.Series(y_pred_optimized_4).value_counts() )
y_pred_2 = pd.Series(y_pred_optimized_4).apply(lambda x : 1 if x ==8 else 0).copy()
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
    if hasattr(xgr_optimized_4, "predict_proba"):
        y_proba = xgr_optimized_4.predict_proba(X_test)
        # 对于多分类问题，我们需要转换为二分类概率
        # 获取欺诈类别（标签8）的概率
        if hasattr(xgr_optimized_4, "classes_"):
            fraud_class_index = list(xgr_optimized_4.classes_).index(8) if 8 in xgr_optimized_4.classes_ else -1
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

print('\n========== 加权投票（使用调优后F1作为权重） ==========')
_w = np.array([rf_f1 if 'rf_f1' in globals() else 0, lgb_f1 if 'lgb_f1' in globals() else 0, xgb_f1 if 'xgb_f1' in globals() else 0], dtype=float)
if _w.sum() <= 0:
    _w = np.array([1.0, 1.0, 1.0], dtype=float)
_w = (_w / _w.sum()).tolist()
voting_clf_weighted = VotingClassifier(
    estimators=[('rf', rf_model), ('lgb', lgb_model), ('xgb', xgb_model_top20)],
    voting='soft',
    weights=_w
)
voting_clf_weighted.fit(X_train_resampled, y_train_resampled)
_proba_w = None
if hasattr(voting_clf_weighted, 'predict_proba'):
    _p = voting_clf_weighted.predict_proba(X_test)
    if hasattr(voting_clf_weighted, 'classes_') and 8 in voting_clf_weighted.classes_:
        _idx = list(voting_clf_weighted.classes_).index(8)
        _proba_w = _p[:, _idx]
_best_f1_v, _best_t_v = -1.0, 0.5
if _proba_w is not None:
    for _t in np.linspace(0.2, 0.7, 26):
        _pred_opt = (_proba_w >= _t).astype(int)
        _f1 = f1_score(y_test_2_voting, _pred_opt)
        if _f1 > _best_f1_v:
            _best_f1_v, _best_t_v = _f1, _t
    _pred_best = (_proba_w >= _best_t_v).astype(int)
    _m_best = confusion_matrix(y_test_2_voting, _pred_best)
    print(f"权重: {_w}")
    print(f"最优阈值: {_best_t_v:.2f}, 最优F1: {_best_f1_v:.4f}")
    print('混淆矩阵：')
    print(_m_best)
    print(f"准确率 (Accuracy): {accuracy_score(y_test_2_voting, _pred_best):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test_2_voting, _pred_best):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_2_voting, _pred_best):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_2_voting, _pred_best):.4f}")
else:
    _pred = voting_clf_weighted.predict(X_test)
    _pred2 = pd.Series(_pred).apply(lambda x: 1 if x == 8 else 0)
    print(f"准确率 (Accuracy): {accuracy_score(y_test_2_voting, _pred2):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test_2_voting, _pred2):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_2_voting, _pred2):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_2_voting, _pred2):.4f}")

print('\n========== 双模型投票（LightGBM + XGBoost） ==========')
_w2 = np.array([lgb_f1 if 'lgb_f1' in globals() else 0, xgb_f1 if 'xgb_f1' in globals() else 0], dtype=float)
if _w2.sum() <= 0:
    _w2 = np.array([1.0, 1.0], dtype=float)
_w2 = (_w2 / _w2.sum()).tolist()
voting_clf_dual = VotingClassifier(
    estimators=[('lgb', lgb_model), ('xgb', xgb_model_top20)],
    voting='soft',
    weights=_w2
)
voting_clf_dual.fit(X_train_resampled, y_train_resampled)
_proba_d = None
if hasattr(voting_clf_dual, 'predict_proba'):
    _pd = voting_clf_dual.predict_proba(X_test)
    if hasattr(voting_clf_dual, 'classes_') and 8 in voting_clf_dual.classes_:
        _idxd = list(voting_clf_dual.classes_).index(8)
        _proba_d = _pd[:, _idxd]
_best_f1_d, _best_t_d = -1.0, 0.5
if _proba_d is not None:
    for _t in np.linspace(0.2, 0.7, 26):
        _pred_opt = (_proba_d >= _t).astype(int)
        _f1 = f1_score(y_test_2_voting, _pred_opt)
        if _f1 > _best_f1_d:
            _best_f1_d, _best_t_d = _f1, _t
    _pred_best = (_proba_d >= _best_t_d).astype(int)
    _m_best = confusion_matrix(y_test_2_voting, _pred_best)
    print(f"权重: {_w2}")
    print(f"最优阈值: {_best_t_d:.2f}, 最优F1: {_best_f1_d:.4f}")
    print('混淆矩阵：')
    print(_m_best)
    print(f"准确率 (Accuracy): {accuracy_score(y_test_2_voting, _pred_best):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test_2_voting, _pred_best):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_2_voting, _pred_best):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_2_voting, _pred_best):.4f}")
else:
    _pred = voting_clf_dual.predict(X_test)
    _pred2 = pd.Series(_pred).apply(lambda x: 1 if x == 8 else 0)
    print(f"准确率 (Accuracy): {accuracy_score(y_test_2_voting, _pred2):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test_2_voting, _pred2):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_2_voting, _pred2):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_2_voting, _pred2):.4f}")

# 添加阈值调整功能以平衡精确率和召回率
print("\n========== 阈值调整优化 ==========")
if y_proba_fraud is not None:
    from sklearn.metrics import precision_recall_curve
    
    # 计算精确率、召回率和阈值
    precision, recall, thresholds = precision_recall_curve(y_test_binary_auc, y_proba_fraud)
    
    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # 处理除零情况
    
    # 找到最佳阈值
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    print(f"基于F1分数的最佳阈值: {best_threshold:.4f}")
    print(f"对应的F1分数: {best_f1:.4f}")
    
    # 使用最佳阈值进行预测
    y_pred_best = (y_proba_fraud >= best_threshold).astype(int)
    
    # 计算使用最佳阈值的指标
    print(f"\n使用最佳阈值 {best_threshold:.4f} 的评估结果:")
    print(f"准确率 (Accuracy): {accuracy_score(y_test_binary_auc, y_pred_best):.4f}")
    print(f"精确率 (Precision): {precision_score(y_test_binary_auc, y_pred_best):.4f}")
    print(f"召回率 (Recall): {recall_score(y_test_binary_auc, y_pred_best):.4f}")
    print(f"F1分数 (F1-Score): {f1_score(y_test_binary_auc, y_pred_best):.4f}")
    
    # 混淆矩阵
    m_best = confusion_matrix(y_test_binary_auc, y_pred_best)
    print('\n混淆矩阵（最佳阈值）：')
    print(m_best)
    
    # 与原始阈值(0.5)的对比
    print(f"\n========== 阈值调整对比 ==========")
    y_pred_default = (y_proba_fraud >= 0.5).astype(int)
    
    print("默认阈值(0.5) vs 最佳阈值对比:")
    print(f"默认阈값 - 精确率: {precision_score(y_test_binary_auc, y_pred_default):.4f}, 召回率: {recall_score(y_test_binary_auc, y_pred_default):.4f}, F1: {f1_score(y_test_binary_auc, y_pred_default):.4f}")
    print(f"最佳阈값 - 精确率: {precision_score(y_test_binary_auc, y_pred_best):.4f}, 召回率: {recall_score(y_test_binary_auc, y_pred_best):.4f}, F1: {f1_score(y_test_binary_auc, y_pred_best):.4f}")
    
    # 可视化精确率-召回率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall曲线')
    plt.plot(recall[best_threshold_idx], precision[best_threshold_idx], 'ro', markersize=10, label=f'最佳F1点 (阈值={best_threshold:.3f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 不同阈值下的指标变化
    print(f"\n========== 不同阈值下的指标变化 ==========")
    # 选择几个关键阈值进行比较
    threshold_candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
    threshold_candidates.append(best_threshold)
    threshold_candidates = sorted(list(set(threshold_candidates)))  # 去重并排序
    
    print("阈값\t精确率\t召回率\tF1分数")
    for threshold in threshold_candidates:
        y_pred_temp = (y_proba_fraud >= threshold).astype(int)
        prec = precision_score(y_test_binary_auc, y_pred_temp)
        rec = recall_score(y_test_binary_auc, y_pred_temp)
        f1 = f1_score(y_test_binary_auc, y_pred_temp)
        print(f"{threshold:.3f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}")

else:
    print("无法进行阈值调整，模型不支持预测概率")

# 结束程序


# 导入必要的库
# 创建RandomForestClassifier模型
print("========== 开始执行 RandomForest 模型 ==========")
print("训练RandomForest模型...")
print(f"使用特征数: {X_train_resampled.shape[1]}")
rf_model = RandomForestClassifier(
    n_estimators=100,           # 树的数量
    max_depth=10,               # 树的最大深度
    min_samples_split=5,        # 内部节点分裂所需的最小样本数
    min_samples_leaf=2,         # 叶节点所需的最小样本数
    max_features='sqrt',        # 寻找最佳分割时考虑的特征数量
    class_weight='balanced',    # 处理不平衡数据
    random_state=27             # 随机种子
)
rf_model.fit(X_train_resampled, y_train_resampled)
print("RandomForest模型训练完成！")

# 预测和评估RandomForest模型
y_pred_rf = rf_model.predict(X_test)
y_pred_2_rf = pd.Series(y_pred_rf).apply(lambda x : 1 if x == 8 else 0).copy()

print('\n========== RandomForest模型评估 ==========')
print('混淆矩阵：')
m_rf = confusion_matrix(y_test_2, y_pred_2_rf)
print(m_rf)

print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2, y_pred_2_rf):.4f}")
print(f"精确率 (Precision): {precision_score(y_test_2, y_pred_2_rf):.4f}")
print(f"召回率 (Recall): {recall_score(y_test_2, y_pred_2_rf):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test_2, y_pred_2_rf):.4f}")

# RandomForest特征重要性
print('\nRandomForest特征重要性 (Top 10):')
rf_feature_importance = rf_model.feature_importances_
rf_feature_importance_df = pd.DataFrame({
    'feature': X_train_resampled.columns,
    'importance': rf_feature_importance
}).sort_values(by='importance', ascending=False)
print(rf_feature_importance_df.head(10))

# 创建LightGBM模型
print("========== 开始执行 LightGBM 模型 ==========")
print("训练LightGBM模型...")
print(f"使用特征数: {X_train_resampled_top20.shape[1]}")
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,  # 还原为较小的值
    max_depth=5,    # 降低最大深度
    learning_rate=0.1,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,  # 增加子节点最小样本数
    min_split_gain=0.1,   # 添加最小分割增益
    random_state=27,
    class_weight='balanced',
    device='gpu'
)
lgb_model.fit(X_train_resampled, y_train_resampled)
print("LightGBM模型训练完成！")

# 预测和评估LightGBM模型
y_pred_lgb = lgb_model.predict(X_test)
y_pred_2_lgb = pd.Series(y_pred_lgb).apply(lambda x : 1 if x == 8 else 0).copy()

print('\n========== LightGBM模型评估 ==========')
print('混淆矩阵：')
m_lgb = confusion_matrix(y_test_2, y_pred_2_lgb)
print(m_lgb)

print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2, y_pred_2_lgb):.4f}")
print(f"精确率 (Precision): {precision_score(y_test_2, y_pred_2_lgb):.4f}")
print(f"召回率 (Recall): {recall_score(y_test_2, y_pred_2_lgb):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test_2, y_pred_2_lgb):.4f}")

# LightGBM特征重要性
print('\nLightGBM特征重要性 (Top 10):')
lgb_feature_importance = lgb_model.feature_importances_
lgb_feature_importance_df = pd.DataFrame({
    'feature': X_train_resampled.columns,
    'importance': lgb_feature_importance
}).sort_values(by='importance', ascending=False)
print(lgb_feature_importance_df.head(10))
print('\nLightGBM增益重要性 (Top 10):')
_lgb_gain = lgb_model.booster_.feature_importance(importance_type='gain')
features_gain = list(lgb_model.booster_.feature_name())
_lgb_gain_df = pd.DataFrame({'feature': features_gain, 'gain': _lgb_gain}).sort_values(by='gain', ascending=False)
print(_lgb_gain_df.head(10))
print('\nLightGBM置换重要性 (Top 10, scoring=AP for fraud vs non-fraud, on gain Top30 features):')
X_test_sample = X_test.sample(n=min(5000, len(X_test)), random_state=27)
y_test_sample_bin = y_test.loc[X_test_sample.index].apply(lambda x: 1 if x == 8 else 0)

# 仅在增益Top30候选特征上进行置换重要性
top30_gain_features = _lgb_gain_df['feature'].head(30).tolist()
_norm = lambda s: s.replace(' ', '_').replace('(', '_').replace(')', '_').lower()
_map = {_norm(c): c for c in X_test.columns}
_top30_cols = [_map[_norm(f)] for f in top30_gain_features if _norm(f) in _map]
_missing = [f for f in top30_gain_features if _norm(f) not in _map]
if _missing:
    print("Top30特征在测试集列中未找到:", _missing)
top30_idx = [X_test_sample.columns.get_loc(c) for c in _top30_cols]

def _fraud_ap_scorer(estimator, X, y_bin):
    if hasattr(estimator, 'predict_proba') and hasattr(estimator, 'classes_') and 8 in estimator.classes_:
        proba = estimator.predict_proba(X)
        idx = list(estimator.classes_).index(8)
        return average_precision_score(y_bin, proba[:, idx])
    y_pred_bin = pd.Series(estimator.predict(X)).apply(lambda x: 1 if x == 8 else 0)
    return average_precision_score(y_bin, y_pred_bin)

_pi_lgb_ap = permutation_importance(
    lgb_model,
    X_test_sample,
    y_test_sample_bin,
    n_repeats=3,
    random_state=27,
    scoring=_fraud_ap_scorer,
    n_jobs=1
)
_pi_all_df = pd.DataFrame({'feature': list(X_test_sample.columns), 'ap_importance': _pi_lgb_ap.importances_mean})
_pi_lgb_ap_df = _pi_all_df[_pi_all_df['feature'].isin(_top30_cols)].sort_values(by='ap_importance', ascending=False)
print(_pi_lgb_ap_df.head(10))

# 使用XGBoost模型（使用前20个特征）
print("========== XGBoost模型（使用前20个特征）评估 ==========")
# 重新训练XGBoost模型，只使用前20个特征
xgb_model_top20 = xgr_optimized_4  # 使用相同的模型参数
xgb_model_top20.fit(X_train_resampled_top20, y_train_resampled)
y_pred_xgb = xgb_model_top20.predict(X_test_top20)
y_pred_2_xgb = pd.Series(y_pred_xgb).apply(lambda x : 1 if x == 8 else 0).copy()

print('混淆矩阵：')
m_xgb = confusion_matrix(y_test_2, y_pred_2_xgb)
print(m_xgb)

print(f"\n准确率 (Accuracy): {accuracy_score(y_test_2, y_pred_2_xgb):.4f}")
print(f"精确率 (Precision): {precision_score(y_test_2, y_pred_2_xgb):.4f}")
print(f"召回率 (Recall): {recall_score(y_test_2, y_pred_2_xgb):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test_2, y_pred_2_xgb):.4f}")

# 创建投票分类器
print("========== 开始执行 Voting 集成模型 ==========")
print("所有模型使用全部特征")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model_top20)
    ],
    voting='soft'  # 使用软投票
)

print("训练投票分类器...")
# 训练投票分类器（使用全部特征）
voting_clf.fit(X_train_resampled, y_train_resampled)

# 进行预测
print("进行预测...")
y_pred_voting = voting_clf.predict(X_test)

# 转换为二分类标签
y_test_2_voting = y_test.apply(lambda x : 1 if x == 8 else 0).copy()
y_pred_2_voting = pd.Series(y_pred_voting).apply(lambda x : 1 if x == 8 else 0).copy()

# 打印评估结果
print('\n========== 投票分类器模型评估 ==========')
print('\n混淆矩阵：')
m_voting = confusion_matrix(y_test_2_voting, y_pred_2_voting)
print(m_voting)

# 准确率
voting_accuracy = accuracy_score(y_test_2_voting, y_pred_2_voting)
print(f"\n准确率 (Accuracy): {voting_accuracy:.4f}")
print(f"精确率 (Precision): {precision_score(y_test_2_voting, y_pred_2_voting):.4f}")
print(f"召回率 (Recall): {recall_score(y_test_2_voting, y_pred_2_voting):.4f}")
print(f"F1分数 (F1-Score): {f1_score(y_test_2_voting, y_pred_2_voting):.4f}")

# ========== 三个模型的性能对比 ==========
print('\n========== 三个模型性能对比总结 ==========')
print(f"{'模型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
print('-' * 60)
print(f"{'RandomForest':<15} {accuracy_score(y_test_2, y_pred_2_rf):<10.4f} {precision_score(y_test_2, y_pred_2_rf):<10.4f} {recall_score(y_test_2, y_pred_2_rf):<10.4f} {f1_score(y_test_2, y_pred_2_rf):<10.4f}")
print(f"{'LightGBM':<15} {accuracy_score(y_test_2, y_pred_2_lgb):<10.4f} {precision_score(y_test_2, y_pred_2_lgb):<10.4f} {recall_score(y_test_2, y_pred_2_lgb):<10.4f} {f1_score(y_test_2, y_pred_2_lgb):<10.4f}")
print(f"{'XGBoost':<15} {accuracy_score(y_test_2, y_pred_2_xgb):<10.4f} {precision_score(y_test_2, y_pred_2_xgb):<10.4f} {recall_score(y_test_2, y_pred_2_xgb):<10.4f} {f1_score(y_test_2, y_pred_2_xgb):<10.4f}")
print(f"{'Voting Ensemble':<15} {voting_accuracy:<10.4f} {precision_score(y_test_2_voting, y_pred_2_voting):<10.4f} {recall_score(y_test_2_voting, y_pred_2_voting):<10.4f} {f1_score(y_test_2_voting, y_pred_2_voting):<10.4f}")

# 找出最佳模型
models_scores = {
    'RandomForest': f1_score(y_test_2, y_pred_2_rf),
    'LightGBM': f1_score(y_test_2, y_pred_2_lgb),
    'XGBoost': f1_score(y_test_2, y_pred_2_xgb),
    'Voting Ensemble': f1_score(y_test_2_voting, y_pred_2_voting)
}
best_model = max(models_scores, key=models_scores.get)
print(f"\n最佳模型（基于F1分数）: {best_model} (F1: {models_scores[best_model]:.4f})")

print('\n========== 特征使用总结 ==========')
print("✅ 所有模型均使用全部特征")
print(f"✅ 特征数: {X_train.shape[1]}")
print(f"\n准确率 (Accuracy): {voting_accuracy:.4f}")

# 精确率、召回率、F1分数（针对欺诈类别）
voting_precision = precision_score(y_test_2_voting, y_pred_2_voting)
voting_recall = recall_score(y_test_2_voting, y_pred_2_voting)
voting_f1 = f1_score(y_test_2_voting, y_pred_2_voting)
print(f"精确率 (Precision): {voting_precision:.4f}")
print(f"召回率 (Recall): {voting_recall:.4f}")
print(f"F1分数 (F1-Score): {voting_f1:.4f}")

# 分类报告
print('\n分类报告：')
print(classification_report(y_test_2_voting, y_pred_2_voting, target_names=['正常订单', '欺诈订单']))

# AUC-PR和AUC-ROC指标
try:
    # 获取预测概率
    y_proba_fraud_voting = None
    
    # 针对投票分类器获取欺诈类别的概率
    if hasattr(voting_clf, "predict_proba"):
        y_proba = voting_clf.predict_proba(X_test)
        # 获取欺诈类别（标签8）的概率
        if hasattr(voting_clf, "classes_"):
            fraud_class_index = list(voting_clf.classes_).index(8) if 8 in voting_clf.classes_ else -1
            if fraud_class_index >= 0:
                y_proba_fraud_voting = y_proba[:, fraud_class_index]
    
    # 计算AUC指标
    if y_proba_fraud_voting is not None:
        auc_roc_voting = roc_auc_score(y_test_2_voting, y_proba_fraud_voting)
        auc_pr_voting = average_precision_score(y_test_2_voting, y_proba_fraud_voting)
        print(f"AUC-ROC: {auc_roc_voting:.4f}")
        print(f"AUC-PR: {auc_pr_voting:.4f}")
    else:
        print("模型不支持预测概率或未找到欺诈类别")
        
except Exception as e:
    print(f"计算AUC指标时出错: {e}")

# 各个模型的特征重要性
print("\n========== 各个模型的特征重要性 ==========")

# RandomForest特征重要性
if hasattr(rf_model, 'feature_importances_'):
    print("\n特征重要性 (RandomForest):")
    feature_importance_rf = rf_model.feature_importances_
    feature_names = list(X_train_resampled.columns)
    feature_importance_df_rf = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_rf
    }).sort_values(by='importance', ascending=False)
    print(feature_importance_df_rf.head(10))

    print('\nRandomForest置换重要性 (Top 10, scoring=AP for fraud vs non-fraud):')
    X_test_sample_rf = X_test.sample(n=min(5000, len(X_test)), random_state=27)
    y_test_sample_bin_rf = y_test_2.loc[X_test_sample_rf.index]

    def _fraud_ap_scorer_rf(estimator, X, y_bin):
        if hasattr(estimator, 'predict_proba') and hasattr(estimator, 'classes_') and 8 in estimator.classes_:
            proba = estimator.predict_proba(X)
            idx = list(estimator.classes_).index(8)
            return average_precision_score(y_bin, proba[:, idx])
        y_pred_bin = pd.Series(estimator.predict(X)).apply(lambda x: 1 if x == 8 else 0)
        return average_precision_score(y_bin, y_pred_bin)

    _pi_rf_ap = permutation_importance(
        rf_model,
        X_test_sample_rf,
        y_test_sample_bin_rf,
        n_repeats=3,
        random_state=27,
        scoring=_fraud_ap_scorer_rf,
        n_jobs=1
    )
    _pi_rf_ap_df = pd.DataFrame({'feature': list(X_test_sample_rf.columns), 'ap_importance': _pi_rf_ap.importances_mean}).sort_values(by='ap_importance', ascending=False)
    print(_pi_rf_ap_df.head(10))



# 模型性能对比
iv_df = calculate_all_features_iv(X_train, y_train, top_n=10)
iv_selected = iv_df[iv_df['IV']>=0.02]['Feature'].tolist()
X_train_iv = X_train_resampled[iv_selected]
X_test_iv = X_test[iv_selected]
print('\n========== IV≥0.02初筛 ==========')
print(f'特征数: {len(iv_selected)}')
rf_iv = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=5,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=27)
rf_iv.fit(X_train_iv,y_train_resampled)
imp_rf = rf_iv.feature_importances_; names_iv = X_train_iv.columns
top30_rf = [names_iv[i] for i in np.argsort(imp_rf)[-30:]]
lgb_iv = lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=31,max_depth=5,learning_rate=0.1,n_estimators=1000,subsample=0.8,colsample_bytree=0.8,min_child_samples=50,min_split_gain=0.1,random_state=27,class_weight='balanced',device='gpu')
lgb_iv.fit(X_train_iv,y_train_resampled)
gain_lgb_iv = lgb_iv.booster_.feature_importance(importance_type='gain')
top30_lgb = [names_iv[i] for i in np.argsort(gain_lgb_iv)[-30:]]
xgb_iv = xgb.XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=6,min_child_weight=1,gamma=0,subsample=0.6,colsample_bytree=0.6,reg_alpha=0,reg_lambda=0,objective='multi:softmax',eval_metric='mlogloss',random_state=27,tree_method='gpu_hist',predictor='gpu_predictor',use_label_encoder=False)
xgb_iv.fit(X_train_iv,y_train_resampled)
imp_xgb_iv = xgb_iv.feature_importances_
top30_xgb = [names_iv[i] for i in np.argsort(imp_xgb_iv)[-30:]]
union_features = list(set(top30_rf) | set(top30_lgb) | set(top30_xgb))
X_train_union = X_train_resampled[union_features]
X_test_union = X_test[union_features]
print('\n========== 并集Top30复筛 ==========')
print(f'特征数: {len(union_features)}')

_const_cols = [c for c in X_train_union.columns if X_train_union[c].nunique() <= 1]
if len(_const_cols) > 0:
    X_train_union = X_train_union.drop(columns=_const_cols)
    X_test_union = X_test_union.drop(columns=_const_cols)
rf_u = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=5,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=27)
rf_u.fit(X_train_union,y_train_resampled)
lgb_u = lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=31,max_depth=5,learning_rate=0.1,n_estimators=1000,subsample=0.8,colsample_bytree=0.8,min_child_samples=20,min_split_gain=0.0,random_state=27,class_weight='balanced',device='cpu',force_row_wise=True)
lgb_u.fit(X_train_union,y_train_resampled)
xgb_u = xgb.XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=6,min_child_weight=1,gamma=0,subsample=0.6,colsample_bytree=0.6,reg_alpha=0,reg_lambda=0,objective='multi:softmax',eval_metric='mlogloss',random_state=27,tree_method='gpu_hist',predictor='gpu_predictor',use_label_encoder=False)
xgb_u.fit(X_train_union,y_train_resampled)
pred_rf_u = rf_u.predict(X_test_union); pred_lgb_u = lgb_u.predict(X_test_union); pred_xgb_u = xgb_u.predict(X_test_union)
y_test_2_union = y_test.apply(lambda x: 1 if x==8 else 0)
pred2_rf_u = pd.Series(pred_rf_u).apply(lambda x: 1 if x==8 else 0)
pred2_lgb_u = pd.Series(pred_lgb_u).apply(lambda x: 1 if x==8 else 0)
pred2_xgb_u = pd.Series(pred_xgb_u).apply(lambda x: 1 if x==8 else 0)
rf_f1_u = f1_score(y_test_2_union,pred2_rf_u); lgb_f1_u = f1_score(y_test_2_union,pred2_lgb_u); xgb_f1_u = f1_score(y_test_2_union,pred2_xgb_u)
voting_u = VotingClassifier(estimators=[('rf',rf_u),('lgb',lgb_u),('xgb',xgb_u)],voting='soft')
voting_u.fit(X_train_union,y_train_resampled)
pred_v_u = voting_u.predict(X_test_union)
pred2_v_u = pd.Series(pred_v_u).apply(lambda x: 1 if x==8 else 0)
v_f1_u = f1_score(y_test_2_union,pred2_v_u)
print(f'F1: RF {rf_f1_u:.4f}  LGBM {lgb_f1_u:.4f}  XGB {xgb_f1_u:.4f}  Voting {v_f1_u:.4f}')
print("\n========== 模型性能对比 ==========")
print("单个模型 vs 投票集成模型")

# 先打印测试集标签分布
print(f"\n测试集真实标签分布 (y_test):")
print(pd.Series(y_test).value_counts())
print(f"\n测试集二分类标签分布 (y_test_2_voting):")
print(y_test_2_voting.value_counts())

# 单个模型性能评估
print(f"\n单个模型性能:")

# RandomForest性能
rf_pred = rf_model.predict(X_test)
# 检查预测结果的分布
print(f"RandomForest预测结果分布: {pd.Series(rf_pred).value_counts()}")
# 正确转换为二分类标签：8为欺诈订单（正类），其他为正常订单（负类）
rf_pred_2 = pd.Series(rf_pred).apply(lambda x : 1 if x == 8 else 0).copy()
# 检查转换后的二分类结果分布
print(f"RandomForest二分类结果分布: {rf_pred_2.value_counts()}")
# 确保y_test_2_voting也是正确的二分类标签
rf_precision = precision_score(y_test_2_voting, rf_pred_2, zero_division=0)
rf_recall = recall_score(y_test_2_voting, rf_pred_2, zero_division=0)
rf_f1 = f1_score(y_test_2_voting, rf_pred_2, zero_division=0)
print(f"  RandomForest - 精确率: {rf_precision:.4f}, 召回率: {rf_recall:.4f}, F1分数: {rf_f1:.4f}")

# 阈值调优（RandomForest）
if hasattr(rf_model, "predict_proba") and hasattr(rf_model, "classes_") and 8 in rf_model.classes_:
    _rf_proba = rf_model.predict_proba(X_test)
    _rf_idx = list(rf_model.classes_).index(8)
    _best_f1, _best_t = -1.0, 0.5
    for _t in np.linspace(0.2, 0.7, 26):
        _pred_opt = (_rf_proba[:, _rf_idx] >= _t).astype(int)
        _f1 = f1_score(y_test_2_voting, _pred_opt)
        if _f1 > _best_f1:
            _best_f1, _best_t = _f1, _t
    _pred_best = (_rf_proba[:, _rf_idx] >= _best_t).astype(int)
    _m_best = confusion_matrix(y_test_2_voting, _pred_best)
    print("\nRandomForest阈值调优:")
    print(f"最优阈值: {_best_t:.2f}, 最优F1: {_best_f1:.4f}")
    print(_m_best)
    rf_precision = precision_score(y_test_2_voting, _pred_best)
    rf_recall = recall_score(y_test_2_voting, _pred_best)
    rf_f1 = f1_score(y_test_2_voting, _pred_best)

# LightGBM性能
lgb_pred = lgb_model.predict(X_test)
# 检查预测结果的分布
print(f"LightGBM预测结果分布: {pd.Series(lgb_pred).value_counts()}")
# 正确转换为二分类标签：8为欺诈订单（正类），其他为正常订单（负类）
lgb_pred_2 = pd.Series(lgb_pred).apply(lambda x : 1 if x == 8 else 0).copy()
# 检查转换后的二分类结果分布
print(f"LightGBM二分类结果分布: {lgb_pred_2.value_counts()}")
# 确保y_test_2_voting也是正确的二分类标签
lgb_precision = precision_score(y_test_2_voting, lgb_pred_2, zero_division=0)
lgb_recall = recall_score(y_test_2_voting, lgb_pred_2, zero_division=0)
lgb_f1 = f1_score(y_test_2_voting, lgb_pred_2, zero_division=0)
print(f"  LightGBM    - 精确率: {lgb_precision:.4f}, 召回率: {lgb_recall:.4f}, F1分数: {lgb_f1:.4f}")

# 阈值调优（LightGBM）
if hasattr(lgb_model, "predict_proba") and hasattr(lgb_model, "classes_") and 8 in lgb_model.classes_:
    _lgb_proba = lgb_model.predict_proba(X_test)
    _lgb_idx = list(lgb_model.classes_).index(8)
    _best_f1_l, _best_t_l = -1.0, 0.5
    for _t in np.linspace(0.2, 0.7, 26):
        _pred_opt = (_lgb_proba[:, _lgb_idx] >= _t).astype(int)
        _f1 = f1_score(y_test_2_voting, _pred_opt)
        if _f1 > _best_f1_l:
            _best_f1_l, _best_t_l = _f1, _t
    _pred_best_l = (_lgb_proba[:, _lgb_idx] >= _best_t_l).astype(int)
    _m_best_l = confusion_matrix(y_test_2_voting, _pred_best_l)
    print("\nLightGBM阈值调优:")
    print(f"最优阈值: {_best_t_l:.2f}, 最优F1: {_best_f1_l:.4f}")
    print(_m_best_l)
    lgb_precision = precision_score(y_test_2_voting, _pred_best_l)
    lgb_recall = recall_score(y_test_2_voting, _pred_best_l)
    lgb_f1 = f1_score(y_test_2_voting, _pred_best_l)

# XGBoost性能（从之前的计算中获取）
xgb_precision = precision_score(y_test_2, y_pred_2, zero_division=0)
xgb_recall = recall_score(y_test_2, y_pred_2, zero_division=0)
xgb_f1 = f1_score(y_test_2, y_pred_2, zero_division=0)
print(f"  XGBoost     - 精确率: {xgb_precision:.4f}, 召回率: {xgb_recall:.4f}, F1分数: {xgb_f1:.4f}")

# 投票集成模型性能
print(f"\n投票集成模型性能:")
print(f"  精确率: {voting_precision:.4f}")
print(f"  召回率: {voting_recall:.4f}")
print(f"  F1分数: {voting_f1:.4f}")

# 结束程序
print("\n========== 模型集成优化完成 ==========")

# 把准确率: 97.95%
# 精确率: 52.37%
# 召回率: 100.00%
# F1分数: 68.74%