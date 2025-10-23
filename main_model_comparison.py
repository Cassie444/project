# -*- coding: utf-8 -*-
"""
模型对比主程序（重构版 - 分离训练和评估）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import load_and_prepare_data, get_categorical_feature_indices
from src.feature_engineering import FeatureProcessor
from src.sampling_methods import SamplingStrategy, calculate_pos_weight
from src.model_factory import ModelFactory
from src.model_training import ModelTrainer, ModelEvaluator
from config import *


def run_model_comparison(
    model_configs,
    dataset_info,
    dataset_name,
    encoding_type="label",
    comparison_name="standard",
):
    """
    通用的模型对比函数

    Parameters:
    -----------
    model_configs : dict
        模型配置字典
    dataset_info : dict
        数据集信息
    dataset_name : str
        数据集名称
    encoding_type : str
        编码类型：'auto', 'label', 'category'
    comparison_name : str
        对比实验名称

    Returns:
    --------
    pd.DataFrame : 评估结果
    """
    print(f"\n使用数据集: {dataset_name} - {comparison_name}对比")

    # 准备数据
    df = dataset_info["data"]
    feature_cols = dataset_info["features"]

    X = df[feature_cols]
    y = df[TARGET_VAR]

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pos_weight = calculate_pos_weight(y_train)
    print(f"正负样本权重比例: {pos_weight:.3f}")

    # 初始化训练器和评估器
    trainer = ModelTrainer(save_dir=f"saved_models/{dataset_name}_{comparison_name}")
    evaluator = ModelEvaluator()

    # 存储所有结果
    all_results = []

    # 遍历方法和模型
    for method_name, models_dict in model_configs.items():
        print(f"\n处理方法: {method_name}")

        for model_name, model_class in models_dict.items():
            # 确定编码方式
            if encoding_type == "auto":
                encoding = ModelFactory.get_required_encoding(model_name)
            else:
                encoding = encoding_type

            model_type = ModelFactory.get_model_type(model_name)
            print(f"  {model_name} (类型: {model_type}, 编码: {encoding})", end="")

            # 特征处理
            processor = FeatureProcessor(
                continuous_vars=CONTINUOUS_VARS,
                categorical_encoding=encoding,
                model_type=model_type,
            )

            X_train_processed = processor.fit_transform(X_train)
            X_test_processed = processor.transform(X_test)

            # 采样
            sampler = SamplingStrategy(
                method=method_name,
                categorical_indices=get_categorical_feature_indices(feature_cols),
                pos_weight_ratio=pos_weight,
            )

            X_train_sampled, y_train_sampled = sampler.apply(
                X_train_processed.values, y_train.values
            )

            if isinstance(X_train_processed, pd.DataFrame):
                X_train_sampled = pd.DataFrame(
                    X_train_sampled, columns=X_train_processed.columns
                )

            # 恢复category类型（如果需要）
            if encoding == "category":
                for col in processor.get_categorical_columns():
                    X_train_sampled[col] = X_train_sampled[col].astype("category")

            # 训练模型
            trained_model = trainer.train_model(
                model_name,
                method_name,
                X_train_sampled,
                y_train_sampled,
                pos_weight,
                processor,
                encoding,
            )

            if "error" not in trained_model:
                # 保存模型
                model_filepath = trainer.save_model(trained_model, dataset_name)

                # 评估模型
                metrics = evaluator.evaluate_trained_model(
                    trained_model, X_test_processed, y_test
                )

                if "error" not in metrics:
                    metrics["model_filepath"] = model_filepath
                    all_results.append(metrics)

                    print(
                        f" -> F1: {metrics['f1_score']:.3f}, "
                        f"Precision: {metrics['precision']:.3f}, "
                        f"AUC: {metrics['auc']:.3f}"
                    )
                    print(
                        f"      校准: {metrics['calibration_method']}, "
                        f"需要校准: {metrics['needs_calibration']}, "
                        f"ECE: {metrics['ece_before']:.3f}->{metrics['ece_after']:.3f}"
                    )
                    print(f"      模型已保存: {model_filepath}")
                else:
                    print(f" -> 评估失败: {metrics['error']}")
            else:
                print(f" -> 训练失败: {trained_model['error']}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values("f1_score", ascending=False)

        filename = f"results_{dataset_name}_{comparison_name}_comparison.csv"
        results_df.to_csv(filename, index=False)

        print(f"\n{dataset_name} - {comparison_name}最佳结果:")
        display_cols = [
            "method",
            "base_model",
            "model_type",
            "f1_score",
            "precision",
            "sensitivity",
            "auc",
            "calibration_method",
            "needs_calibration",
            "ece_after",
        ]
        print(results_df.head(10)[display_cols].to_string(index=False))
        print(f"\n结果已保存到: {filename}")

    return results_df


def run_all_model_comparisons():
    """
    运行所有模型对比实验
    """
    print("=" * 80)
    print("运行模型对比实验（训练评估分离 + 智能校准 + 模型保存）")
    print("=" * 80)

    # 加载数据
    datasets = load_and_prepare_data()

    # 1. 标准模型对比（自动编码选择）
    print("\n" + "=" * 60)
    print("1. 标准模型对比（自动编码选择）")
    print("=" * 60)

    standard_model_configs = ModelFactory.get_model_configs_by_method()

    for dataset_name, dataset_info in datasets.items():
        run_model_comparison(
            model_configs=standard_model_configs,
            dataset_info=dataset_info,
            dataset_name=dataset_name,
            encoding_type="auto",
            comparison_name="standard",
        )

    # 2. 现代树模型对比（category编码）
    print("\n" + "=" * 60)
    print("2. 现代树模型对比（category编码）")
    print("=" * 60)

    # 仅现代树模型
    modern_tree_models = {
        "XGBoost": xgb.XGBClassifier,
        "LightGBM": lgb.LGBMClassifier,
        "CatBoost": CatBoostClassifier,
    }

    modern_tree_configs = {
        "original": modern_tree_models,
        "undersample": modern_tree_models,
        "oversample": modern_tree_models,
        "smotenc": modern_tree_models,
        "model_balanced": {f"{k}_Balanced": v for k, v in modern_tree_models.items()},
    }

    for dataset_name, dataset_info in datasets.items():
        run_model_comparison(
            model_configs=modern_tree_configs,
            dataset_info=dataset_info,
            dataset_name=dataset_name,
            encoding_type="category",
            comparison_name="modern_tree_category",
        )


if __name__ == "__main__":
    run_all_model_comparisons()
