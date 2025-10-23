# -*- coding: utf-8 -*-
"""
超参数优化模块
"""

import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler

from .feature_engineering import FeatureProcessor
from .sampling_methods import SamplingStrategy, calculate_pos_weight
from .model_factory import ModelFactory, get_param_space
from .evaluation import evaluate_model, calculate_composite_score
from config import RANDOM_STATE, CV_FOLDS


class HyperparameterOptimizer:
    """
    超参数优化器
    """
    
    def __init__(self, X, y, feature_cols, model_type='tree', categorical_encoding='category'):
        """
        Parameters:
        -----------
        X : DataFrame or array-like
            特征数据
        y : array-like
            标签数据
        feature_cols : list
            特征列名
        model_type : str
            模型类型：'linear' 或 'tree'
        categorical_encoding : str
            类别变量编码方式：'label', 'dummy', 'category'
        """
        self.X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_cols)
        self.y = np.asarray(y)
        self.feature_cols = feature_cols
        self.model_type = model_type
        self.categorical_encoding = categorical_encoding
        
        # 计算正负样本权重比例
        self.pos_weight = calculate_pos_weight(self.y)
        
        # 结果存储
        self.results = []
        
    def get_model_configs(self):
        """获取模型配置"""
        if self.model_type == 'tree':
            return ModelFactory.get_tree_model_configs()
        else:
            return ModelFactory.get_model_configs()
    
    def objective(self, trial, method_name, model_class, model_name):
        """
        Optuna优化目标函数
        """
        # 获取参数空间并建议超参数
        param_space = get_param_space(model_name)
        params = self._suggest_params(trial, param_space)
        params = self._fix_param_constraints(params, model_name)
        
        try:
            # 交叉验证评估
            metrics = self._cross_validate(model_class, model_name, method_name, params)
            return calculate_composite_score(metrics)
        except Exception as e:
            print(f'Trial failed: {e}')
            return 0.0
    
    def _suggest_params(self, trial, param_space):
        """建议超参数"""
        params = {}
        for param, values in param_space.items():
            if isinstance(values, tuple) and len(values) == 2:
                min_val, max_val = values
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param] = trial.suggest_int(param, min_val, max_val)
                else:
                    if param in ['learning_rate', 'reg_alpha', 'reg_lambda', 'min_child_weight', 'l2_leaf_reg']:
                        params[param] = trial.suggest_float(param, min_val, max_val, log=True)
                    else:
                        params[param] = trial.suggest_float(param, min_val, max_val)
            elif isinstance(values, list):
                params[param] = trial.suggest_categorical(param, values)
        return params
    
    def _fix_param_constraints(self, params, model_name):
        """修复参数约束"""
        if 'LightGBM' in model_name:
            max_depth = params.get('max_depth', -1)
            if max_depth not in (-1, None) and 'num_leaves' in params:
                max_leaves = 2 ** max_depth
                if params['num_leaves'] >= max_leaves:
                    params['num_leaves'] = max(max_leaves - 1, 15)
        return params
    
    def _cross_validate(self, model_class, model_name, method_name, params):
        """交叉验证评估"""
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        metrics_list = []
        
        for train_idx, val_idx in cv.split(self.X, self.y):
            # 划分数据
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # 特征处理
            processor = FeatureProcessor(
                self.feature_cols, 
                model_type=self.model_type,
                categorical_encoding=self.categorical_encoding
            )
            X_train_processed = processor.fit_transform(X_train)
            X_val_processed = processor.transform(X_val)
            
            # 采样
            if method_name != 'model_balanced':
                sampler = SamplingStrategy(
                    method=method_name,
                    categorical_indices=processor.get_categorical_indices(),
                    pos_weight_ratio=self.pos_weight
                )
                X_train_processed, y_train = sampler.apply(X_train_processed.values, y_train)
                X_train_processed = pd.DataFrame(X_train_processed, columns=processor.feature_cols)
            
            # 创建和训练模型
            model = ModelFactory.create_model(
                method_name, model_name, 
                pos_weight=self.pos_weight,
                categorical_cols=processor.get_categorical_columns(),
                **params
            )
            
            # 训练
            if 'LightGBM' in model_name and processor.get_categorical_columns():
                model.fit(X_train_processed, y_train, categorical_feature=processor.get_categorical_columns())
            else:
                model.fit(X_train_processed, y_train)
            
            # 预测和评估
            y_prob = model.predict_proba(X_val_processed)[:, 1]
            metrics = evaluate_model(y_val, y_prob, model_name=model_name)
            metrics_list.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key != 'model':
                values = [m[key] for m in metrics_list if not np.isnan(m[key])]
                avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def optimize_model(self, method_name, model_name, model_class, n_trials=100):
        """优化单个模型"""
        print(f'优化 {method_name} - {model_name}')
        
        # 创建study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=RANDOM_STATE),
            study_name=f'{method_name}_{model_name}'
        )
        
        # 优化
        study.optimize(
            lambda trial: self.objective(trial, method_name, model_class, model_name),
            n_trials=n_trials,
            show_progress_bar=False
        )
        
        # 最终评估
        best_metrics = self._cross_validate(model_class, model_name, method_name, study.best_params)
        best_score = calculate_composite_score(best_metrics)
        
        # 记录结果
        result = {
            'method': method_name,
            'model': model_name,
            'best_params': study.best_params,
            'score': best_score,
            **best_metrics
        }
        
        self.results.append(result)
        
        print(f'最佳评分: {best_score:.3f}')
        print(f'F1: {best_metrics.get("f1_score", 0):.3f}, '
              f'Precision: {best_metrics.get("precision", 0):.3f}, '
              f'Recall: {best_metrics.get("sensitivity", 0):.3f}')
        print('-' * 50)
        
        return result
    
    def run_optimization(self, n_trials=100, methods=None, models=None):
        """运行完整优化流程"""
        print(f'开始超参数优化')
        print(f'数据集大小: {self.X.shape}')
        print(f'正样本比例: {np.mean(self.y):.3f}')
        print(f'正负样本权重比例: {self.pos_weight:.3f}')
        print('=' * 80)
        
        model_configs = self.get_model_configs()
        
        # 过滤方法和模型
        if methods:
            model_configs = {k: v for k, v in model_configs.items() if k in methods}
        if models:
            for method in model_configs:
                model_configs[method] = {k: v for k, v in model_configs[method].items() if k in models}
        
        # 遍历优化
        for method_name, models_dict in model_configs.items():
            print(f'\n处理方法: {method_name}')
            print('=' * 40)
            
            for model_name, model_class in models_dict.items():
                try:
                    self.optimize_model(method_name, model_name, model_class, n_trials)
                except Exception as e:
                    print(f'模型 {method_name}-{model_name} 优化失败: {e}')
                    continue
        
        return self.get_results_dataframe()
    
    def get_results_dataframe(self):
        """获取结果DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        return df.sort_values('score', ascending=False)