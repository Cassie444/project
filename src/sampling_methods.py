# -*- coding: utf-8 -*-
"""
采样方法模块
"""

from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from config import RANDOM_STATE


class SamplingStrategy:
    """
    采样策略类
    """
    
    def __init__(self, method='original', categorical_indices=None, pos_weight_ratio=1.0):
        """
        Parameters:
        -----------
        method : str
            采样方法：'original', 'undersample', 'oversample', 'smotenc'
        categorical_indices : list
            类别特征的索引列表（用于SMOTENC）
        pos_weight_ratio : float
            正负样本权重比例
        """
        self.method = method
        self.categorical_indices = categorical_indices or []
        self.pos_weight_ratio = pos_weight_ratio
        
    def apply(self, X_train, y_train):
        """
        应用采样策略
        """
        if self.method == 'original':
            return X_train, y_train
        
        # 根据不平衡程度确定采样比例
        sampling_ratio = 0.2 if self.pos_weight_ratio > 10 else 1.0
        
        if self.method == 'undersample':
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_ratio, 
                random_state=RANDOM_STATE
            )
        elif self.method == 'oversample':
            sampler = RandomOverSampler(
                sampling_strategy=sampling_ratio, 
                random_state=RANDOM_STATE
            )
        elif self.method == 'smotenc':
            # 确保k_neighbors不超过正样本数
            k_neighbors = max(1, min(5, int(y_train.sum()) - 1))
            sampler = SMOTENC(
                sampling_strategy=sampling_ratio,
                categorical_features=self.categorical_indices,
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors
            )
        else:
            raise ValueError(f"未知的采样方法: {self.method}")
        
        return sampler.fit_resample(X_train, y_train)


def calculate_pos_weight(y_train):
    """
    计算正负样本权重比例
    """
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    return negative_count / max(1, positive_count)