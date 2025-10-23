# -*- coding: utf-8 -*-
"""
特征工程模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import CONTINUOUS_VARS


class FeatureProcessor:
    """
    特征处理器：支持不同模型类型的特征预处理
    """

    def __init__(
        self,
        feature_cols,
        model_type="tree",
        categorical_encoding="label",
        standardize_continuous=True,
    ):
        """
        Parameters:
        -----------
        feature_cols : list
            特征列名列表
        model_type : str, default='tree'
            模型类型：'linear' 或 'tree'
        categorical_encoding : str, default='label'
            类别变量编码方式：'label', 'dummy', 'category'
        standardize_continuous : bool, default=True
            是否标准化连续变量
        """
        self.feature_cols = feature_cols
        self.model_type = model_type
        self.categorical_encoding = categorical_encoding
        self.standardize_continuous = standardize_continuous

        # 识别连续变量和类别变量
        self.continuous_cols = [col for col in feature_cols if col in CONTINUOUS_VARS]
        self.categorical_cols = [
            col for col in feature_cols if col not in CONTINUOUS_VARS
        ]

        # 初始化处理器
        self.scaler = StandardScaler() if standardize_continuous else None
        self.fitted = False

    def get_categorical_indices(self):
        """获取类别变量的索引"""
        if self.categorical_encoding == "category":
            return [
                i
                for i, col in enumerate(self.feature_cols)
                if col in self.categorical_cols
            ]
        return []

    def get_categorical_columns(self):
        """获取类别变量的列名"""
        if self.categorical_encoding == "category":
            return self.categorical_cols
        return []

    def fit(self, X_train):
        """
        在训练集上拟合处理器
        """
        X_processed = X_train.copy()

        # 连续变量标准化
        if self.standardize_continuous and self.continuous_cols and self.scaler:
            self.scaler.fit(X_processed[self.continuous_cols])

        self.fitted = True
        return self

    def transform(self, X, is_training=False):
        """
        转换特征

        Parameters:
        -----------
        X : DataFrame
            输入特征
        is_training : bool
            是否为训练集（用于fit_transform）
        """
        if not self.fitted and not is_training:
            raise ValueError("处理器尚未拟合，请先调用fit()方法")

        X_processed = X.copy()

        # 连续变量标准化
        if self.standardize_continuous and self.continuous_cols and self.scaler:
            if is_training:
                X_processed[self.continuous_cols] = self.scaler.fit_transform(
                    X_processed[self.continuous_cols]
                )
            else:
                X_processed[self.continuous_cols] = self.scaler.transform(
                    X_processed[self.continuous_cols]
                )

        # 类别变量处理
        if self.categorical_encoding == "dummy":
            # 线性模型使用dummy编码
            X_processed = pd.get_dummies(
                X_processed, columns=self.categorical_cols, drop_first=True
            )
        elif self.categorical_encoding == "category":
            # 现代树模型使用category类型
            for col in self.categorical_cols:
                X_processed[col] = X_processed[col].astype("category")
        else:
            for col in self.categorical_cols:
                X_processed[col] = X_processed[col].astype("int")

        return X_processed

    def fit_transform(self, X_train):
        """
        拟合并转换训练集
        """
        self.fit(X_train)
        return self.transform(X_train, is_training=True)
