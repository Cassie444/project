# -*- coding: utf-8 -*-
"""
模型工厂模块
"""

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from config import RANDOM_STATE


class ModelFactory:
    """
    模型工厂类
    """

    @staticmethod
    def get_model_configs():
        """
        获取所有模型配置
        """
        return {
            "original": {
                "SGD_Classifier": SGDClassifier,
                "Logistic_Regression": LogisticRegression,
                "AdaBoost": AdaBoostClassifier,
                "Gradient_Boosting": GradientBoostingClassifier,
                "Random_Forest": RandomForestClassifier,
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
            "model_balanced": {
                "SGD_Classifier_Balanced": SGDClassifier,
                "Logistic_Balanced": LogisticRegression,
                "EasyEnsemble": EasyEnsembleClassifier,
                "Gradient_Boosting_Balanced": GradientBoostingClassifier,
                "BalancedRandomForest": BalancedRandomForestClassifier,
                "XGBoost_Balanced": xgb.XGBClassifier,
                "LightGBM_Balanced": lgb.LGBMClassifier,
                "CatBoost_Balanced": CatBoostClassifier,
            },
            "undersample": {
                "SGD_Classifier": SGDClassifier,
                "Logistic_Regression": LogisticRegression,
                "AdaBoost": AdaBoostClassifier,
                "Gradient_Boosting": GradientBoostingClassifier,
                "Random_Forest": RandomForestClassifier,
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
            "oversample": {
                "SGD_Classifier": SGDClassifier,
                "Logistic_Regression": LogisticRegression,
                "AdaBoost": AdaBoostClassifier,
                "Gradient_Boosting": GradientBoostingClassifier,
                "Random_Forest": RandomForestClassifier,
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
            "smotenc": {
                "SGD_Classifier": SGDClassifier,
                "Logistic_Regression": LogisticRegression,
                "AdaBoost": AdaBoostClassifier,
                "Gradient_Boosting": GradientBoostingClassifier,
                "Random_Forest": RandomForestClassifier,
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
        }

    @staticmethod
    def get_tree_model_configs():
        """
        获取仅树模型配置（用于现代树模型的类别特征处理）
        """
        return {
            "original": {
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
            "model_balanced": {
                "XGBoost_Balanced": xgb.XGBClassifier,
                "LightGBM_Balanced": lgb.LGBMClassifier,
                "CatBoost_Balanced": CatBoostClassifier,
            },
            "undersample": {
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
            "oversample": {
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
            "smotenc": {
                "XGBoost": xgb.XGBClassifier,
                "LightGBM": lgb.LGBMClassifier,
                "CatBoost": CatBoostClassifier,
            },
        }

    @staticmethod
    def create_model(
        method_name, model_name, pos_weight=1.0, categorical_cols=None, **kwargs
    ):
        """
        创建模型实例

        Parameters:
        -----------
        method_name : str
            采样方法名
        model_name : str
            模型名称
        pos_weight : float
            正样本权重
        categorical_cols : list
            类别特征列名（用于CatBoost）
        **kwargs : dict
            额外的模型参数
        """
        # 默认参数
        base_params = {"random_state": RANDOM_STATE}
        base_params.update(kwargs)

        # 平衡模型的特殊处理
        if method_name == "model_balanced":
            return ModelFactory._create_balanced_model(
                model_name, pos_weight, categorical_cols, base_params
            )
        else:
            return ModelFactory._create_standard_model(
                model_name, categorical_cols, base_params
            )

    @staticmethod
    def _create_balanced_model(model_name, pos_weight, categorical_cols, params):
        """创建平衡模型"""
        if model_name == "EasyEnsemble":
            return EasyEnsembleClassifier(n_estimators=10, **params)
        elif model_name == "Gradient_Boosting_Balanced":
            return GradientBoostingClassifier(
                learning_rate=0.01, n_estimators=1500, **params
            )
        elif model_name == "SGD_Classifier_Balanced":
            return SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1000, **params)
        elif model_name == "Logistic_Balanced":
            return LogisticRegression(
                class_weight="balanced",
                C=1.0,
                solver="liblinear",
                max_iter=1000,
                **params,
            )
        elif model_name == "BalancedRandomForest":
            return BalancedRandomForestClassifier(n_estimators=1000, **params)
        elif model_name == "XGBoost_Balanced":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=pos_weight,
                max_depth=6,
                learning_rate=0.05,
                n_estimators=1000,
                eval_metric="logloss",
                enable_categorical=True,
                tree_method="hist",
                **params,
            )
        elif model_name == "LightGBM_Balanced":
            return lgb.LGBMClassifier(
                objective="binary",
                scale_pos_weight=pos_weight,
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=1000,
                verbose=-1,
                **params,
            )
        elif model_name == "CatBoost_Balanced":
            return CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                auto_class_weights="Balanced",
                verbose=False,
                cat_features=categorical_cols,
                **params,
            )
        else:
            raise ValueError(f"未知的平衡模型: {model_name}")

    @staticmethod
    def _create_standard_model(model_name, categorical_cols, params):
        """创建标准模型"""
        if model_name == "AdaBoost":
            return AdaBoostClassifier(n_estimators=1000, learning_rate=0.05, **params)
        elif model_name == "Gradient_Boosting":
            return GradientBoostingClassifier(
                learning_rate=0.01, n_estimators=1500, **params
            )
        elif model_name == "SGD_Classifier":
            return SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1000, **params)
        elif model_name == "Logistic_Regression":
            return LogisticRegression(
                penalty="l2", C=0.01, solver="liblinear", max_iter=1000, **params
            )
        elif model_name == "Random_Forest":
            return RandomForestClassifier(criterion="gini", n_estimators=2000, **params)
        elif model_name == "XGBoost":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                booster="gbtree",
                gamma=0,
                max_depth=4,
                reg_lambda=7,
                eval_metric="logloss",
                enable_categorical=True,
                tree_method="hist",
                **params,
            )
        elif model_name == "LightGBM":
            return lgb.LGBMClassifier(
                objective="binary",
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                **params,
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                verbose=False,
                cat_features=categorical_cols,
                **params,
            )
        else:
            raise ValueError(f"未知的标准模型: {model_name}")


def get_param_space(model_name):
    """
    获取模型的超参数搜索空间
    """
    spaces = {
        "AdaBoost": {"n_estimators": (50, 300), "learning_rate": (0.1, 1.0)},
        "EasyEnsemble": {
            "n_estimators": (10, 100),
            "sampling_strategy": ["auto"],
            "replacement": [False],
        },
        "BalancedRandomForest": {
            "n_estimators": (100, 500),
            "max_depth": (5, 15),
            "min_samples_split": (5, 30),
            "min_samples_leaf": (2, 15),
            "max_features": ["sqrt", "log2", 0.5, 0.6, 0.7, 0.8],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
        },
        "Gradient_Boosting": {
            "n_estimators": (100, 500),
            "learning_rate": (0.01, 0.2),
            "max_depth": (3, 7),
            "subsample": (0.6, 0.95),
            "min_samples_split": (5, 30),
            "min_samples_leaf": (2, 15),
            "max_features": ["sqrt", "log2", 0.5, 0.6, 0.7, 0.8],
        },
        "SGD_Classifier": {
            "loss": ["log_loss", "modified_huber", "hinge"],
            "alpha": (0.0001, 0.01),
            "l1_ratio": (0, 1),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "eta0": (0.001, 0.1),
            "penalty": ["l1", "l2"],
            "max_iter": (500, 2000),
        },
        "Logistic_Regression": {
            "C": (0.01, 10.0),
            "penalty": ["l1", "l2"],
            "l1_ratio": (0, 1),
            "solver": ["liblinear", "saga"],
            "max_iter": (500, 2000),
        },
        "Random_Forest": {
            "n_estimators": (100, 500),
            "max_depth": (5, 15),
            "min_samples_split": (5, 30),
            "min_samples_leaf": (2, 15),
            "max_features": ["sqrt", "log2", 0.5, 0.6, 0.7, 0.8],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
        },
        "XGBoost": {
            "n_estimators": (800, 3000),
            "learning_rate": (0.02, 0.2),
            "max_depth": (3, 7),
            "min_child_weight": (2, 20),
            "subsample": (0.6, 0.95),
            "colsample_bytree": (0.6, 0.95),
            "reg_alpha": (0, 5),
            "reg_lambda": (0.5, 10),
            "gamma": (0, 5),
            "max_delta_step": (0, 5),
        },
        "LightGBM": {
            "n_estimators": (800, 3000),
            "learning_rate": (0.02, 0.2),
            "max_depth": [-1, 3, 4, 5, 6, 7, 8],
            "num_leaves": (15, 255),
            "min_child_samples": (10, 120),
            "min_child_weight": (0.001, 0.1),
            "subsample": (0.6, 0.95),
            "colsample_bytree": (0.6, 0.95),
            "reg_alpha": (0, 5),
            "reg_lambda": (0.5, 10),
            "min_split_gain": (0, 1),
            "max_bin": (63, 255),
        },
        "CatBoost": {
            "iterations": (800, 3000),
            "learning_rate": (0.02, 0.2),
            "depth": (4, 8),
            "l2_leaf_reg": (1, 10),
            "border_count": (32, 255),
            "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
            "bootstrap_type": ["Bayesian", "Bernoulli"],
        },
    }

    # 处理平衡模型的名称映射
    name_mapping = {
        "EasyEnsemble": "EasyEnsemble",
        "BalancedRandomForest": "BalancedRandomForest",
        "Gradient_Boosting_Balanced": "Gradient_Boosting",
        "SGD_Classifier_Balanced": "SGD_Classifier",
        "Logistic_Balanced": "Logistic_Regression",
        "XGBoost_Balanced": "XGBoost",
        "LightGBM_Balanced": "LightGBM",
        "CatBoost_Balanced": "CatBoost",
    }

    base_name = name_mapping.get(model_name, model_name.replace("_Balanced", ""))
    return spaces.get(base_name, {})
