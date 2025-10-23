# -*- coding: utf-8 -*-
"""
超参数调优主程序
"""

import time
import pandas as pd

from src.data_preprocessing import get_datasets
from src.optimization import HyperparameterOptimizer
from config import DEFAULT_N_TRIALS


def run_hyperparameter_tuning(n_trials=DEFAULT_N_TRIALS, 
                              methods=None, 
                              models=None,
                              model_type='tree',
                              categorical_encoding='category'):
    """
    运行超参数调优
    
    Parameters:
    -----------
    n_trials : int
        优化试验次数
    methods : list, optional
        指定采样方法列表，如 ['model_balanced', 'oversample']
    models : list, optional
        指定模型列表，如 ['XGBoost', 'LightGBM']
    model_type : str
        模型类型：'tree' 或 'linear'
    categorical_encoding : str
        类别变量编码方式
    """
    # 获取数据集
    datasets = get_datasets()
    
    # 遍历数据集
    for dataset_name, dataset_info in datasets.items():
        print(f'\n{"="*60}')
        print(f'超参数调优 - 数据集: {dataset_name}')
        print(f'{"="*60}')
        
        start_time = time.time()
        
        # 准备数据
        df = dataset_info['data']
        feature_cols = dataset_info['features']
        
        X = df[feature_cols]
        y = df['positive']
        
        # 创建优化器
        optimizer = HyperparameterOptimizer(
            X=X,
            y=y,
            feature_cols=feature_cols,
            model_type=model_type,
            categorical_encoding=categorical_encoding
        )
        
        # 运行优化
        results_df = optimizer.run_optimization(
            n_trials=n_trials,
            methods=methods,
            models=models
        )
        
        elapsed_time = time.time() - start_time
        
        # 输出结果
        if not results_df.empty:
            print(f'\n{"="*60}')
            print(f'{dataset_name} - 超参数调优结果')
            print(f'{"="*60}')
            print(f'总耗时: {elapsed_time:.1f}秒')
            print()
            print(results_df[
                ['method', 'model', 'score', 'f1_score', 'precision', 'sensitivity', 'auc']
            ].head(10).to_string(index=False))
            
            # 保存结果
            filename = f'results_{dataset_name}_hyperparameter_tuning.csv'
            results_df.to_csv(filename, index=False)
            print(f'\n结果已保存到: {filename}')
        else:
            print("没有成功的优化结果")


def run_quick_test():
    """快速测试（少量试验）"""
    print("运行快速测试（每个模型3次试验）")
    run_hyperparameter_tuning(
        n_trials=3,
        methods=['model_balanced', 'oversample'],
        models=['XGBoost', 'LightGBM'],
        model_type='tree',
        categorical_encoding='category'
    )


def run_full_optimization():
    """完整优化（所有模型和方法）"""
    print("运行完整超参数优化")
    run_hyperparameter_tuning(
        n_trials=100,
        model_type='tree',
        categorical_encoding='category'
    )


def run_tree_models_optimization():
    """仅优化树模型"""
    print("运行树模型超参数优化")
    run_hyperparameter_tuning(
        n_trials=50,
        methods=['model_balanced', 'undersample', 'oversample', 'smotenc'],
        models=['XGBoost', 'LightGBM', 'CatBoost'],
        model_type='tree',
        categorical_encoding='category'
    )


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'quick':
            run_quick_test()
        elif mode == 'full':
            run_full_optimization()
        elif mode == 'tree':
            run_tree_models_optimization()
        else:
            print("使用方式: python main_hyperparameter_tuning.py [quick|full|tree]")
    else:
        # 默认运行快速测试
        run_quick_test()