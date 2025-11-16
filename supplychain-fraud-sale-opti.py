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

# 尝试导入Dask相关库用于分布式训练
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    # 尝试导入LocalCUDACluster，如果失败则使用替代方案
    try:
        from dask_cuda import LocalCUDACluster
        DASK_CUDA_AVAILABLE = True
    except ImportError:
        LocalCUDACluster = None
        DASK_CUDA_AVAILABLE = False
        print("警告: 未找到dask-cuda库，将使用标准Dask集群")
    from xgboost import dask as dxgb
    DASK_AVAILABLE = True
    print("成功导入Dask和XGBoost分布式训练相关库")
except ImportError as e:
    DASK_AVAILABLE = False
    DASK_CUDA_AVAILABLE = False
    print(f"警告: 无法导入Dask相关库: {e}")
    print("将使用标准XGBoost训练")

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score  # 添加AUC相关指标
from sklearn.pipeline import Pipeline

#数据加载
dataset=pd.read_csv(input+'SupplyChain.csv', encoding='unicode_escape')

# 打印Type、Delivery Status和Late_delivery_risk的取值
print("========== 特征取值分析 ==========")
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
data.info()

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
drop_features = ['Order Status', 
                 #'shipping_year', 'shipping_month', 'shipping_week_day', 'shipping_hour',
                 #'order_year', 'order_month', 'order_week_day', 'order_hour',
                 'order date (DateOrders)', 'shipping date (DateOrders)'
                ]

astype_features = [
                   'shipping_year', 'shipping_month', 'shipping_week_day', 'shipping_hour',
                   'order_year', 'order_month', 'order_week_day', 'order_hour'
                  ]

for column in astype_features:
    data[column] = data[column].astype('category')

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

# 打印调试信息
print(f"原始特征数量: {data2.shape[1]}")
print(f"最终特征数量: {len(feature_cols)}")

# 打印特征数量确认
print(f"总特征数量: {len(feature_cols)}")
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

print('\n========== XGBClassifier 模型评估 ==========')

#XGBClassifier 
# 使用第一组参数调优得到的最佳参数
xgr = xgb.XGBClassifier(
    learning_rate=0.01,         # 最佳学习率
    n_estimators=1000,          # 最佳树数量
    max_depth=8,                # 适当增加深度
    min_child_weight=3,         # 增加子节点最小权重
    gamma=0.1,                  # 增加正则化
    subsample=0.7,              # 减少样本采样比例
    colsample_bytree=0.7,       # 减少特征采样比例
    objective='multi:softmax',
    eval_metric='mlogloss',     # 显式指定评估指标，避免警告
    random_state=27,
    tree_method='gpu_hist',     # 使用GPU加速
    predictor='gpu_predictor',   # 使用GPU进行预测
    use_label_encoder=False     # 避免标签编码器警告
)

# 首先使用标准XGBoost训练模型，避免因Dask集群创建问题导致卡住
print("开始训练XGBoost模型...")
xgr.fit(X_train_resampled, y_train_resampled)
print("XGBoost模型训练完成")

# 进行预测
y_pred = xgr.predict(X_test)
print("模型预测完成")

# 只有在需要使用Dask进行参数调优时才创建集群
# 注释掉直接创建Dask集群的代码，避免程序卡住
'''
# 创建Dask集群
client, cluster = setup_dask_cluster()

# 使用Dask XGBoost训练模型
xgr, y_pred = train_with_dask_xgboost(client, X_train_resampled, y_train_resampled, X_test, y_test)

# 关闭Dask集群
if client is not None:
    client.close()
if cluster is not None:
    cluster.close()
'''

# 计算所有特征的IV值
print("\n========== 特征IV值分析 ==========")
iv_df = calculate_all_features_iv(X_train, y_train, top_n=20)

# 打印特征重要性
print("\n特征重要性 (XGBoost):")
feature_importance_xgb = xgr.feature_importances_
feature_names = X_train.columns
feature_importance_df_xgb = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance_xgb
}).sort_values(by='importance', ascending=False)
print(feature_importance_df_xgb.head(20))

# 添加第三组参数调优代码
print("\n========== XGBoost第三组参数调优 ==========")

# 第三组参数调优：gamma, subsample, colsample_bytree
param_grid_3 = {
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# 创建基础模型用于调优（使用前两组调优得到的最佳参数）
xgb_base_3 = xgb.XGBClassifier(
    learning_rate=0.01,         # 第一组调优得到的最佳学习率
    n_estimators=1000,          # 第一组调优得到的最佳树数量
    max_depth=6,                # 直接使用最佳参数值
    min_child_weight=1,         # 直接使用最佳参数值
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=27,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False     # 避免标签编码器警告
)

# 尝试使用Dask进行参数调优，如果失败则回退到标准方法
grid_search_3 = None

# 添加Dask集群设置函数
def setup_dask_cluster():
    """设置Dask集群用于分布式训练"""
    if not DASK_AVAILABLE:
        print("Dask不可用，使用标准XGBoost训练")
        return None, None
    
    try:
        print("开始创建Dask集群...")
        if DASK_CUDA_AVAILABLE and LocalCUDACluster is not None:
            # 创建本地CUDA集群，使用所有可用的GPU
            print("尝试创建CUDA集群...")
            cluster = LocalCUDACluster(n_workers=2, threads_per_worker=1)  # 恢复到2个workers
            print("使用CUDA集群进行多GPU训练")
        else:
            # 使用标准本地集群作为回退方案
            print("尝试创建标准集群...")
            cluster = LocalCluster(n_workers=2, threads_per_worker=2)  # 恢复到原来的配置
            print("使用标准集群进行训练")
        
        client = Client(cluster, timeout='60s')  # 保持超时机制
        print(f"成功创建Dask集群: {cluster}")
        print(f"客户端连接: {client}")
        return client, cluster
    except Exception as e:
        print(f"创建Dask集群时出错: {e}")
        return None, None

# 只有在确实需要使用Dask时才创建集群
if DASK_AVAILABLE:
    try:
        # 导入Dask相关函数（如果尚未定义）
        # 注意：这些函数应该已经在文件中定义了
        print("尝试使用Dask进行参数调优...")
        
        # 使用Dask进行参数调优的函数
        def tune_with_dask_xgboost_local(X_train, y_train):
            """使用Dask XGBoost进行参数调优"""
            if not DASK_AVAILABLE:
                print("Dask不可用，使用标准参数调优")
                return None
            
            client = None
            cluster = None
            
            try:
                # 创建Dask集群
                print("正在创建Dask集群...")
                client, cluster = setup_dask_cluster()
                if client is None or cluster is None:
                    print("无法创建Dask集群，使用标准参数调优")
                    return None
                
                print("开始使用Dask XGBoost进行参数调优...")
                
                # 将pandas数据转换为Dask数据格式
                print("转换数据格式...")
                X_train_dask = dd.from_pandas(X_train, npartitions=4)  # 恢复到4个分区
                y_train_dask = dd.from_pandas(y_train, npartitions=4)  # 恢复到4个分区
                
                # 创建DaskDMatrix
                print("创建DaskDMatrix...")
                dtrain = dxgb.DaskDMatrix(client, X_train_dask, y_train_dask)
                
                # 设置XGBoost参数，启用GPU加速
                params = {
                    'objective': 'multi:softmax',
                    'num_class': len(y_train.unique()),
                    'tree_method': 'hist',  # 使用histogram算法
                    'device': 'cuda',  # 启用GPU
                    'eval_metric': 'mlogloss',
                    'learning_rate': 0.01,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'random_state': 27
                }
                
                # 由于参数组合较多，我们使用更小的样本进行调优以节省时间
                # 使用10%的训练数据进行快速调优
                print("准备训练数据...")
                sample_size = int(0.1 * len(X_train))
                X_train_sample = X_train[:sample_size]
                y_train_sample = y_train[:sample_size]
                
                # 创建基础模型用于调优，使用GPU加速
                xgb_base_3_dask = xgb.XGBClassifier(
                    learning_rate=0.01,
                    n_estimators=1000,  # 恢复到1000棵树
                    max_depth=6,
                    min_child_weight=1,
                    objective='multi:softmax',
                    eval_metric='mlogloss',
                    random_state=27,
                    tree_method='gpu_hist',  # 使用GPU加速
                    predictor='gpu_predictor',  # 使用GPU进行预测
                    use_label_encoder=False
                )
                
                # 执行网格搜索
                print("执行第三组参数调优 (gamma, subsample, colsample_bytree)...")
                grid_search_3_dask = GridSearchCV(
                    estimator=xgb_base_3_dask,
                    param_grid=param_grid_3,
                    scoring='f1_macro',
                    cv=3,  # 恢复到3折交叉验证
                    n_jobs=-1,  # 使用所有可用的CPU核心
                    verbose=1
                )
                
                print("开始训练...")
                grid_search_3_dask.fit(X_train_sample, y_train_sample)
                print("训练完成")
                
                # 关闭Dask集群
                if client is not None:
                    client.close()
                if cluster is not None:
                    cluster.close()
                
                return grid_search_3_dask
                
            except Exception as e:
                print(f"使用Dask XGBoost进行参数调优时出错: {e}")
                # 确保关闭集群
                try:
                    if client is not None:
                        client.close()
                    if cluster is not None:
                        cluster.close()
                except:
                    pass
                return None
            finally:
                # 确保无论如何都关闭集群
                try:
                    if client is not None:
                        client.close()
                    if cluster is not None:
                        cluster.close()
                except:
                    pass
        
        # 尝试使用Dask进行参数调优
        print("开始Dask参数调优...")
        grid_search_3 = tune_with_dask_xgboost_local(X_train_resampled, y_train_resampled)
        print("Dask参数调优结束")
        
    except Exception as e:
        print(f"尝试使用Dask进行参数调优时出错: {e}")
        grid_search_3 = None

# 如果Dask不可用或Dask调优失败，回退到标准参数调优方法
if grid_search_3 is None:
    print("使用标准参数调优方法")
    # 执行网格搜索
    print("执行第三组参数调优 (gamma, subsample, colsample_bytree)...")
    grid_search_3 = GridSearchCV(
        estimator=xgb_base_3,
        param_grid=param_grid_3,
        scoring='f1_macro',  # 使用F1分数作为评估指标
        cv=3,  # 3折交叉验证
        n_jobs=-1,  # 使用所有CPU核心
        verbose=1
    )
    
    # 由于参数组合较多，我们使用更小的样本进行调优以节省时间
    # 使用10%的训练数据进行快速调优
    sample_size = int(0.1 * len(X_train_resampled))
    X_train_sample = X_train_resampled[:sample_size]
    y_train_sample = y_train_resampled[:sample_size]

    grid_search_3.fit(X_train_sample, y_train_sample)

print("第三组参数调优完成!")
print(f"最佳参数: {grid_search_3.best_params_}")
print(f"最佳得分: {grid_search_3.best_score_:.4f}")

# 使用最佳参数重新训练模型
print("使用第三组最佳参数重新训练模型...")
xgr_optimized_3 = xgb.XGBClassifier(
    learning_rate=0.01,         # 第一组调优得到的最佳学习率
    n_estimators=1000,          # 第一组调优得到的最佳树数量
    max_depth=6,                # 直接使用最佳参数值
    min_child_weight=1,         # 直接使用最佳参数值
    gamma=grid_search_3.best_params_['gamma'],
    subsample=grid_search_3.best_params_['subsample'],
    colsample_bytree=grid_search_3.best_params_['colsample_bytree'],
    objective='multi:softmax',
    eval_metric='mlogloss',
    random_state=27,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    use_label_encoder=False     # 避免标签编码器警告
)

xgr_optimized_3.fit(X_train_resampled, y_train_resampled)
y_pred_optimized_3 = xgr_optimized_3.predict(X_test)

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
xgb.plot_importance(xgr_optimized_3,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred_optimized_3).value_counts() )
y_pred_2 = pd.Series(y_pred_optimized_3).apply(lambda x : 1 if x ==8 else 0).copy()
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
    if hasattr(xgr_optimized_3, "predict_proba"):
        y_proba = xgr_optimized_3.predict_proba(X_test)
        # 对于多分类问题，我们需要转换为二分类概率
        # 获取欺诈类别（标签8）的概率
        if hasattr(xgr_optimized_3, "classes_"):
            fraud_class_index = list(xgr_optimized_3.classes_).index(8) if 8 in xgr_optimized_3.classes_ else -1
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