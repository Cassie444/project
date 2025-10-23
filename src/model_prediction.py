# -*- coding: utf-8 -*-
"""
模型预测主程序
"""

import pandas as pd
import numpy as np
import os
from config import load_and_prepare_data
from src.model_training import ModelPredictor


def predict_with_best_models(dataset_name, X_new, top_n=5):
    """
    使用最佳模型进行预测

    Parameters:
    -----------
    dataset_name : str
        数据集名称
    X_new : DataFrame
        新的特征数据
    top_n : int
        使用前N个最佳模型

    Returns:
    --------
    dict : 预测结果
    """
    predictor = ModelPredictor()

    # 加载评估结果，找到最佳模型
    standard_results_file = f"results_{dataset_name}_standard_comparison.csv"
    modern_results_file = f"results_{dataset_name}_modern_tree_category_comparison.csv"

    best_models = []

    # 从标准对比结果中选择最佳模型
    if os.path.exists(standard_results_file):
        standard_df = pd.read_csv(standard_results_file)
        best_standard = standard_df.head(top_n)
        for _, row in best_standard.iterrows():
            if pd.notna(row.get("model_filepath")):
                best_models.append(
                    {
                        "filepath": row["model_filepath"],
                        "name": f"{row['method']}_{row['base_model']}_{row['encoding']}",
                        "f1_score": row["f1_score"],
                        "comparison_type": "standard",
                    }
                )

    # 从现代树对比结果中选择最佳模型
    if os.path.exists(modern_results_file):
        modern_df = pd.read_csv(modern_results_file)
        best_modern = modern_df.head(top_n)
        for _, row in best_modern.iterrows():
            if pd.notna(row.get("model_filepath")):
                best_models.append(
                    {
                        "filepath": row["model_filepath"],
                        "name": f"{row['method']}_{row['base_model']}_category",
                        "f1_score": row["f1_score"],
                        "comparison_type": "modern_tree",
                    }
                )

    # 按F1分数排序
    best_models = sorted(best_models, key=lambda x: x["f1_score"], reverse=True)[:top_n]

    if not best_models:
        return {"error": f"未找到{dataset_name}的训练好的模型"}

    print(f"使用前{len(best_models)}个最佳模型进行预测:")
    for model in best_models:
        print(f"  - {model['name']} (F1: {model['f1_score']:.3f})")

    # 进行预测
    predictions = {}

    for model_info in best_models:
        try:
            result = predictor.predict_with_saved_model(model_info["filepath"], X_new)
            if "error" not in result:
                predictions[model_info["name"]] = result
                print(f"✓ {model_info['name']} 预测完成")
            else:
                print(f"✗ {model_info['name']} 预测失败: {result['error']}")
        except Exception as e:
            print(f"✗ {model_info['name']} 预测异常: {str(e)}")

    return predictions


def ensemble_predictions(predictions_dict, method="voting"):
    """
    集成多个模型的预测结果

    Parameters:
    -----------
    predictions_dict : dict
        多个模型的预测结果
    method : str
        集成方法：'voting', 'averaging'

    Returns:
    --------
    dict : 集成后的预测结果
    """
    if not predictions_dict:
        return {"error": "没有有效的预测结果"}

    # 提取所有预测概率
    all_probs = []
    all_preds = []
    model_names = []

    for model_name, pred_result in predictions_dict.items():
        if "error" not in pred_result:
            all_probs.append(pred_result["probabilities_calibrated"])
            all_preds.append(pred_result["predictions"])
            model_names.append(model_name)

    if not all_probs:
        return {"error": "没有有效的预测概率"}

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    if method == "averaging":
        # 概率平均
        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    elif method == "voting":
        # 多数投票
        ensemble_preds = np.round(np.mean(all_preds, axis=0)).astype(int)
        ensemble_probs = np.mean(all_probs, axis=0)
    else:
        raise ValueError(f"未知的集成方法: {method}")

    return {
        "ensemble_predictions": ensemble_preds,
        "ensemble_probabilities": ensemble_probs,
        "individual_predictions": predictions_dict,
        "ensemble_method": method,
        "models_used": model_names,
        "num_models": len(model_names),
    }


def predict_on_new_data(dataset_name, new_data_path, output_path=None, top_n=5):
    """
    对新数据进行预测

    Parameters:
    -----------
    dataset_name : str
        数据集名称（用于选择对应的训练模型）
    new_data_path : str
        新数据文件路径
    output_path : str, optional
        预测结果保存路径
    top_n : int
        使用前N个最佳模型
    """
    print(f"对新数据进行预测: {new_data_path}")
    print(f"使用数据集: {dataset_name}")

    # 加载新数据
    try:
        if new_data_path.endswith(".csv"):
            new_data = pd.read_csv(new_data_path)
        elif new_data_path.endswith(".dta"):
            new_data = pd.read_stata(new_data_path)
        else:
            raise ValueError("不支持的文件格式，请使用CSV或DTA文件")

        print(f"新数据shape: {new_data.shape}")

    except Exception as e:
        print(f"加载新数据失败: {e}")
        return

    # 获取特征列（需要与训练时一致）
    datasets = load_and_prepare_data()
    if dataset_name not in datasets:
        print(f"未找到数据集配置: {dataset_name}")
        return

    feature_cols = datasets[dataset_name]["features"]

    # 检查特征列是否存在
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        print(f"新数据缺少以下特征列: {missing_cols}")
        return

    X_new = new_data[feature_cols]
    print(f"特征数量: {len(feature_cols)}")

    # 使用最佳模型进行预测
    predictions = predict_with_best_models(dataset_name, X_new, top_n)

    if "error" in predictions:
        print(f"预测失败: {predictions['error']}")
        return

    # 集成预测结果
    ensemble_result = ensemble_predictions(predictions, method="averaging")

    if "error" in ensemble_result:
        print(f"集成失败: {ensemble_result['error']}")
        return

    # 准备输出结果
    result_df = new_data.copy()
    result_df["ensemble_prediction"] = ensemble_result["ensemble_predictions"]
    result_df["ensemble_probability"] = ensemble_result["ensemble_probabilities"]

    # 添加个别模型的预测结果
    for model_name in ensemble_result["models_used"]:
        if model_name in predictions:
            result_df[f"{model_name}_prediction"] = predictions[model_name][
                "predictions"
            ]
            result_df[f"{model_name}_probability"] = predictions[model_name][
                "probabilities_calibrated"
            ]

    # 保存结果
    if output_path is None:
        output_path = f"predictions_{dataset_name}_{os.path.basename(new_data_path)}"

    result_df.to_csv(output_path, index=False)

    print(f"\n预测完成!")
    print(f"使用了{ensemble_result['num_models']}个模型")
    print(f"集成方法: {ensemble_result['ensemble_method']}")
    print(f"预测的正例比例: {result_df['ensemble_prediction'].mean():.3f}")
    print(f"平均预测概率: {result_df['ensemble_probability'].mean():.3f}")
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    # 示例：对新数据进行预测
    # predict_on_new_data(
    #     dataset_name='dataset1',
    #     new_data_path='path/to/new_data.csv',
    #     top_n=3
    # )

    print("预测模块已就绪")
    print("使用方式:")
    print("1. predict_on_new_data('dataset1', 'new_data.csv') - 对新数据预测")
    print("2. predict_with_best_models('dataset1', X_new) - 使用最佳模型预测")
    print("3. ensemble_predictions(predictions_dict) - 集成多个预测结果")
