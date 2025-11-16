#!/usr/bin/env python
# coding: utf-8

"""
使用Dask和XGBoost进行分布式GPU训练的示例代码
此代码可以在Kaggle环境中使用多个GPU加速XGBoost训练
"""

import numpy as np
import pandas as pd
import warnings
import time

# 忽略警告信息
warnings.filterwarnings('ignore')

# 数据加载路径
input_path = '/kaggle/input/source/'  # 修改为正确的数据集路径

#==================================================
# 导入必要的库
#==================================================
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
    
    import xgboost as xgb
    from xgboost import dask as dxgb
    print("成功导入Dask和XGBoost相关库")
except ImportError as e:
    print(f"导入库时出错: {e}")
    print("请确保在Kaggle环境中安装了必要的库")

#==================================================
# 数据加载和预处理
#==================================================
def load_and_preprocess_data():
    """加载并预处理数据"""
    print("开始加载数据...")
    # 加载数据
    dataset = pd.read_csv(input_path + 'SupplyChain.csv', encoding='unicode_escape')
    print(f"数据加载完成，形状: {dataset.shape}")
    
    # 选择需要的特征列（这里使用您优化后的特征）
    # 根据您的优化版本，只保留前4个高价值特征
    high_value_features = ['Delivery Status', 'Type_Delivery_Cross', 'Type', 'Late_delivery_risk']
    
    # 选择特征和目标变量
    feature_cols = [col for col in dataset.columns if col in high_value_features]
    target_col = 'Order Status'
    
    # 提取特征和目标变量
    X = dataset[feature_cols]
    y = dataset[target_col]
    
    print(f"特征列: {feature_cols}")
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    return X, y

#==================================================
# 创建Dask集群并进行分布式训练
#==================================================
def setup_dask_cluster():
    """设置Dask集群用于分布式训练"""
    try:
        if DASK_CUDA_AVAILABLE and LocalCUDACluster is not None:
            # 创建本地CUDA集群，使用所有可用的GPU
            # n_workers参数设置为GPU数量，threads_per_worker根据需要调整
            cluster = LocalCUDACluster(n_workers=2, threads_per_worker=1)
            print("使用CUDA集群进行多GPU训练")
        else:
            # 使用标准本地集群作为回退方案
            cluster = LocalCluster(n_workers=2, threads_per_worker=2)
            print("使用标准集群进行训练")
        
        client = Client(cluster)
        print(f"成功创建Dask集群: {cluster}")
        print(f"客户端连接: {client}")
        return client, cluster
    except Exception as e:
        print(f"创建Dask集群时出错: {e}")
        return None, None

#==================================================
# 使用Dask XGBoost进行分布式训练
#==================================================
def train_with_dask_xgboost(client, X, y):
    """使用Dask XGBoost进行分布式训练"""
    try:
        print("开始使用Dask XGBoost进行分布式训练...")
        
        # 将pandas数据转换为Dask数据格式
        X_dask = dd.from_pandas(X, npartitions=4)
        y_dask = dd.from_pandas(y, npartitions=4)
        
        # 创建DaskDMatrix
        dtrain = dxgb.DaskDMatrix(client, X_dask, y_dask)
        
        # 设置XGBoost参数，启用GPU加速
        params = {
            'objective': 'multi:softmax',
            'num_class': len(y.unique()),
            'tree_method': 'hist',      # 使用histogram算法
            'device': 'cuda',           # 启用GPU
            'eval_metric': 'mlogloss',
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0.1,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'random_state': 27
        }
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 执行分布式训练
        output = dxgb.train(
            client,
            params,
            dtrain,
            num_boost_round=1000,  # 树的数量
            evals=[(dtrain, 'train')]
        )
        
        # 获取训练好的模型
        booster = output['booster']
        
        # 记录训练结束时间
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Dask XGBoost训练完成，耗时: {training_time:.2f}秒")
        
        return booster
        
    except Exception as e:
        print(f"使用Dask XGBoost训练时出错: {e}")
        return None

#==================================================
# 主函数
#==================================================
def main():
    """主函数"""
    print("========== 开始XGBoost分布式GPU训练 ==========")
    
    # 1. 加载和预处理数据
    X, y = load_and_preprocess_data()
    
    # 2. 设置Dask集群
    client, cluster = setup_dask_cluster()
    if client is None:
        print("无法创建Dask集群，退出程序")
        return
    
    try:
        # 3. 使用Dask XGBoost进行分布式训练
        booster = train_with_dask_xgboost(client, X, y)
        
        if booster is not None:
            print("模型训练成功!")
            # 可以在这里添加模型保存代码
            # booster.save_model('xgboost_dask_gpu_model.json')
        else:
            print("模型训练失败")
            
    finally:
        # 关闭集群连接
        if client:
            client.close()
        if cluster:
            cluster.close()
    
    print("========== XGBoost分布式GPU训练结束 ==========")

#==================================================
# 程序入口
#==================================================
if __name__ == "__main__":
    main()