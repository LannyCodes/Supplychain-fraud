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
from sklearn.tree import DecisionTreeClassifier

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

## 处理数据不平衡
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

#==================================================
# Code cell 5
#==================================================

#数据加载
dataset=pd.read_csv(input+'SupplyChain.csv', encoding='unicode_escape')


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
feature_cols = [column for column in data2.columns if column not in drop_features]
# display(feature_cols)

astype_features = [
                   'shipping_year', 'shipping_month', 'shipping_week_day', 'shipping_hour',
                   'order_year', 'order_month', 'order_week_day', 'order_hour'
                  ]

for column in astype_features:
    data[column] = data[column].astype('category')

# # ###随机采样，测试程序用，设置一个小样本的数据集，快速预览模型
# np.random.seed(10)
# 
# #按照百分比抽样，不放回
# data2_sample = data2.sample(frac=0.01) #抽取20%的数据
# display(data2_sample.shape)
# 
# # 样本数据
# X = data2_sample[feature_cols]
# y = data2_sample['Order Status']
# 
# 全量特征
# AC 0.315976069133614
# CPU times: user 3min 24s, sys: 1min 20s, total: 4min 44s
# Wall time: 1min 12s



# 全量数据
X = data2[feature_cols]
y = data2['Order Status']

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

#==================================================
# Code cell 27
#==================================================

"""

# GaussianNB
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
# 使用平衡后的数据进行训练
y_pred = gnb.fit(X_train_resampled, y_train_resampled).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred).value_counts() )
y_pred_2 = pd.Series(y_pred).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# ===== 详细评估指标 =====
print('\n========== LinearSVC 模型评估 ==========')

# 混淆矩阵
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

#==================================================
# Code cell 28
#==================================================



# LinearSVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# 添加 class_weight='balanced' 处理不平衡数据
clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5, class_weight='balanced'))
clf.fit(X_train_resampled, y_train_resampled)

# print(clf.named_steps['linearsvc'].coef_)
# print(clf.named_steps['linearsvc'].intercept_)

y_pred = clf.predict(X_test)

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred).value_counts() )
y_pred_2 = pd.Series(y_pred).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# ===== 详细评估指标 =====
print('\n========== KNeighborsClassifier 模型评估 ==========')

# 混淆矩阵
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

#==================================================
# Code cell 29
#==================================================



# KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
# KNN 使用平衡后的数据
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train_resampled, y_train_resampled)
      
y_pred = neigh.predict(X_test)

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred).value_counts() )
y_pred_2 = pd.Series(y_pred).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# ===== 详细评估指标 =====
print('\n========== LinearDiscriminantAnalysis 模型评估 ==========')

# 混淆矩阵
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

#==================================================
# Code cell 30
#==================================================



# LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis()
# 使用平衡后的数据
clf.fit(X_train_resampled, y_train_resampled)

y_pred = clf.predict(X_test)

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred).value_counts() )
y_pred_2 = pd.Series(y_pred).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# ===== 详细评估指标 =====
print('\n========== DecisionTreeClassifier 模型评估 ==========')

# 混淆矩阵
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

#==================================================
# Code cell 31
#==================================================



# DecisionTreeClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
# 添加 class_weight='balanced' 处理不平衡数据
clf = DecisionTreeClassifier(random_state=2021, class_weight='balanced')

clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test)

# 转换为二分类标签
display( pd.Series(y_test).value_counts() )
y_test_2 = y_test.apply(lambda x : 1 if x ==8 else 0).copy()
y_test_2.value_counts()

# # 转换为二分类标签
display( pd.Series(y_pred).value_counts() )
y_pred_2 = pd.Series(y_pred).apply(lambda x : 1 if x ==8 else 0).copy()
pd.Series(y_pred_2).value_counts()

# ===== 详细评估指标 =====
print('\n========== RandomForestClassifier 模型评估 ==========')

# 混淆矩阵
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

#==================================================
# Code cell 32
#==================================================



# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
# 添加 class_weight='balanced' 处理不平衡数据
clf = RandomForestClassifier(max_depth=7, random_state=2021, class_weight='balanced')
clf.fit(X_train_resampled, y_train_resampled)

y_pred = clf.predict(X_test)

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

# 分类报告
print('\n分类报告：')
print(classification_report(y_test_2, y_pred_2, target_names=['正常订单', '欺诈订单']))

#==================================================
# Code cell 33
#==================================================


print('\n========== XGBClassifier 模型评估 ==========')
"""

#XGBClassifier 
# 计算类别权重比例用于 XGBoost
# 针对多分类问题，使用平衡后的数据或保持原数据但不设置scale_pos_weight
xgr = xgb.XGBClassifier(learning_rate=0.1,
                        n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='multi:softmax', # 指定损失函数
                        random_state=27            # 随机数
                        )

# 使用 SMOTE 平衡后的数据训练
xgr.fit(X_train_resampled, y_train_resampled)
y_pred = xgr.predict(X_test)

### plot feature importance
fig,ax = plt.subplots(figsize=(15,15))
xgb.plot_importance(xgr,
                height=0.5,
                ax=ax,
                max_num_features=64)
plt.show()

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

# 分类报告
print('\n分类报告：')
print(classification_report(y_test_2, y_pred_2, target_names=['正常订单', '欺诈订单']))

#==================================================
# Code cell 34
#==================================================
print('\n========== LogisticRegression 模型评估 ==========')
# # 训练
# LR = sklearn.linear_model.LinearRegression()  #报错
# LR = sklearn.linear_model.LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000)

# 多分类，添加 class_weight='balanced' 处理不平衡数据
LR = sklearn.linear_model.LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000, class_weight='balanced') 

reg = LR.fit(X_train_resampled, y_train_resampled)
reg.score(X_train, y_train)
reg.coef_
reg.intercept_
y_pred = reg.predict(X_test)

print("LR")
print(classification_report(y_test, y_pred))
print("AC",accuracy_score(y_test, y_pred))


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

# 分类报告
print('\n分类报告：')
print(classification_report(y_test_2, y_pred_2, target_names=['正常订单', '欺诈订单']))

#==================================================
# Code cell 35
#==================================================

#  模型调优
# 交叉验证，
# 网格搜索

# 查看当前类别分布
print("原始训练集类别分布:")
print(pd.Series(y_train).value_counts())
print("\nSMOTE后训练集类别分布:")
print(pd.Series(y_train_resampled).value_counts())

# 尝试其他采样方法
print("\n========== 尝试改进的采样方法 ==========")

# 1. 尝试Borderline-SMOTE
try:
    from imblearn.over_sampling import BorderlineSMOTE
    print("\n使用Borderline-SMOTE进行过采样...")
    borderline_smote = BorderlineSMOTE(random_state=2021, k_neighbors=5)
    X_train_borderline, y_train_borderline = borderline_smote.fit_resample(X_train, y_train)
    print("Borderline-SMOTE后训练集类别分布:")
    print(pd.Series(y_train_borderline).value_counts())
except ImportError:
    print("BorderlineSMOTE不可用")

# 2. 尝试ADASYN
try:
    from imblearn.over_sampling import ADASYN
    print("\n使用ADASYN进行过采样...")
    adasyn = ADASYN(random_state=2021, n_neighbors=5)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
    print("ADASYN后训练集类别分布:")
    print(pd.Series(y_train_adasyn).value_counts())
except ImportError:
    print("ADASYN不可用")

# 3. 尝试SMOTEENN混合方法
try:
    from imblearn.combine import SMOTEENN
    print("\n使用SMOTEENN进行混合采样...")
    smoteenn = SMOTEENN(random_state=2021, smote=SMOTE(k_neighbors=5))
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
    print("SMOTEENN后训练集类别分布:")
    print(pd.Series(y_train_smoteenn).value_counts())
except ImportError:
    print("SMOTEENN不可用")

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

# 5. 使用不同的评估指标
print("\n========== 使用AUC-PR评估指标 ==========")
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    # 使用XGBoost模型替代随机森林模型
    # 计算预测概率
    y_proba = xgr.predict_proba(X_test)
    
    # 对于多分类问题，我们需要转换为二分类概率
    # 获取欺诈类别（标签8）的概率
    fraud_class_index = list(xgr.classes_).index(8) if 8 in xgr.classes_ else -1
    
    if fraud_class_index >= 0:
        y_proba_fraud = y_proba[:, fraud_class_index]
        y_test_binary_auc = y_test.apply(lambda x: 1 if x == 8 else 0)
        
        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(y_test_binary_auc, y_proba_fraud)
        auc_pr = average_precision_score(y_test_binary_auc, y_proba_fraud)
        
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
    else:
        print("未找到欺诈类别")
        
except Exception as e:
    print(f"计算AUC指标时出错: {e}")

print("\n========== 采样方法对比完成 ==========")

#==================================================
# Code cell 37
#==================================================
