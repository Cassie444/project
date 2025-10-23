# -*- coding: utf-8 -*-
"""
模型训练模块
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

from .model_factory import ModelFactory
from .calibration import SmartProbabilityCalibrator


class ModelTrainer:
    """
    模型训练器 - 负责训练模型、校准和保存
    """

    def __init__(self, save_dir="saved_models"):
        """
        Parameters:
        -----------
        save_dir : str
            模型保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_model(
        self,
        model_name,
        method_name,
        X_train,
        y_train,
        pos_weight,
        processor,
        encoding="label",
    ):
        """
        训练单个模型

        Parameters:
        -----------
        model_name : str
            模型名称
        method_name : str
            采样方法名
        X_train : array-like
            训练特征
        y_train : array-like
            训练标签
        pos_weight : float
            正样本权重
        processor : FeatureProcessor
            特征处理器
        encoding : str
            编码类型

        Returns:
        --------
        dict : 包含训练好的模型和校准器的字典
        """
        try:
            # 创建模型
            model = ModelFactory.create_model(
                model_name,
                method_name,
                pos_weight=pos_weight,
                categorical_cols=processor.get_categorical_columns(),
                encoding_type=encoding,
            )

            # 训练模型（根据模型类型选择训练方式）
            if ModelFactory.requires_sample_weight(model_name):
                # 需要sample_weight的模型
                sample_weight = np.where(y_train == 1, pos_weight, 1.0)
                model.fit(X_train, y_train, sample_weight=sample_weight)
            elif ModelFactory.requires_categorical_feature_param(model_name, encoding):
                # 需要指定categorical_feature的LightGBM
                model.fit(
                    X_train,
                    y_train,
                    categorical_feature=processor.get_categorical_columns(),
                )
            else:
                # 标准训练
                model.fit(X_train, y_train)

            # 创建并拟合校准器
            calibrator = SmartProbabilityCalibrator(method="auto", cv=3)
            calibrator.fit(model, X_train, y_train)

            return {
                "model": model,
                "calibrator": calibrator,
                "processor": processor,
                "model_name": model_name,
                "method_name": method_name,
                "encoding": encoding,
                "pos_weight": pos_weight,
                "training_info": {
                    "train_samples": len(y_train),
                    "positive_samples": int(y_train.sum()),
                    "positive_rate": float(y_train.mean()),
                    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            }

        except Exception as e:
            return {"error": str(e)}

    def save_model(self, trained_model_dict, dataset_name):
        """
        保存训练好的模型

        Parameters:
        -----------
        trained_model_dict : dict
            训练好的模型字典
        dataset_name : str
            数据集名称

        Returns:
        --------
        str : 保存的文件路径
        """
        if "error" in trained_model_dict:
            return None

        model_name = trained_model_dict["model_name"]
        method_name = trained_model_dict["method_name"]
        encoding = trained_model_dict["encoding"]

        # 构建文件名
        filename = f"{dataset_name}_{method_name}_{model_name}_{encoding}.pkl"
        filepath = os.path.join(self.save_dir, filename)

        # 保存模型
        with open(filepath, "wb") as f:
            pickle.dump(trained_model_dict, f)

        return filepath

    def load_model(self, filepath):
        """
        加载保存的模型

        Parameters:
        -----------
        filepath : str
            模型文件路径

        Returns:
        --------
        dict : 加载的模型字典
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


class ModelEvaluator:
    """
    模型评估器 - 负责评估模型性能
    """

    @staticmethod
    def evaluate_trained_model(trained_model_dict, X_test, y_test):
        """
        评估训练好的模型

        Parameters:
        -----------
        trained_model_dict : dict
            训练好的模型字典
        X_test : array-like
            测试特征
        y_test : array-like
            测试标签

        Returns:
        --------
        dict : 评估结果
        """
        from .evaluation import evaluate_model
        from .calibration import auto_calibrate_and_predict

        if "error" in trained_model_dict:
            return {"error": trained_model_dict["error"]}

        try:
            model = trained_model_dict["model"]
            calibrator = trained_model_dict["calibrator"]

            # 预测概率
            y_prob_raw = model.predict_proba(X_test)[:, 1]

            # 校准概率
            y_prob_calibrated = calibrator.calibrate_probabilities(y_prob_raw)

            # 计算校准后的ECE
            if calibrator.should_calibrate_flag and y_test is not None:
                ece_after = calibrator._calculate_ece(y_test, y_prob_calibrated)
                calibrator.ece_after = ece_after

            # 获取校准信息
            calibration_info = calibrator.get_calibration_info()

            # 评估模型性能
            model_name = trained_model_dict["model_name"]
            method_name = trained_model_dict["method_name"]
            metrics = evaluate_model(
                y_test, y_prob_calibrated, model_name=f"{method_name}_{model_name}"
            )

            # 添加模型和校准信息
            metrics.update(
                {
                    "method": method_name,
                    "base_model": model_name,
                    "model_type": ModelFactory.get_model_type(model_name),
                    "encoding": trained_model_dict["encoding"],
                    "calibration_method": calibration_info["calibration_method"],
                    "needs_calibration": calibration_info["needs_calibration"],
                    "ece_before": calibration_info["ece_before"],
                    "ece_after": calibration_info["ece_after"],
                    "training_info": trained_model_dict["training_info"],
                }
            )

            return metrics

        except Exception as e:
            return {"error": str(e)}


class ModelPredictor:
    """
    模型预测器 - 用于对新数据进行预测
    """

    def __init__(self, model_trainer=None):
        """
        Parameters:
        -----------
        model_trainer : ModelTrainer, optional
            模型训练器实例
        """
        self.model_trainer = model_trainer or ModelTrainer()

    def predict_with_saved_model(self, model_filepath, X_new):
        """
        使用保存的模型进行预测

        Parameters:
        -----------
        model_filepath : str
            保存的模型文件路径
        X_new : array-like
            新的特征数据

        Returns:
        --------
        dict : 预测结果
        """
        try:
            # 加载模型
            trained_model_dict = self.model_trainer.load_model(model_filepath)

            if "error" in trained_model_dict:
                return {"error": trained_model_dict["error"]}

            # 预测
            return self.predict_with_trained_model(trained_model_dict, X_new)

        except Exception as e:
            return {"error": str(e)}

    def predict_with_trained_model(self, trained_model_dict, X_new):
        """
        使用训练好的模型字典进行预测

        Parameters:
        -----------
        trained_model_dict : dict
            训练好的模型字典
        X_new : array-like
            新的特征数据

        Returns:
        --------
        dict : 预测结果
        """
        try:
            model = trained_model_dict["model"]
            calibrator = trained_model_dict["calibrator"]
            processor = trained_model_dict["processor"]

            # 特征处理
            X_processed = processor.transform(X_new)

            # 预测原始概率
            y_prob_raw = model.predict_proba(X_processed)[:, 1]

            # 校准概率
            y_prob_calibrated = calibrator.calibrate_probabilities(y_prob_raw)

            # 生成预测标签（阈值0.5）
            y_pred = (y_prob_calibrated >= 0.5).astype(int)

            return {
                "predictions": y_pred,
                "probabilities_raw": y_prob_raw,
                "probabilities_calibrated": y_prob_calibrated,
                "model_info": {
                    "model_name": trained_model_dict["model_name"],
                    "method_name": trained_model_dict["method_name"],
                    "encoding": trained_model_dict["encoding"],
                    "calibration_method": calibrator.chosen_method,
                    "training_info": trained_model_dict["training_info"],
                },
            }

        except Exception as e:
            return {"error": str(e)}

    def batch_predict_from_directory(self, models_dir, X_new):
        """
        使用目录中的所有模型进行批量预测

        Parameters:
        -----------
        models_dir : str
            模型保存目录
        X_new : array-like
            新的特征数据

        Returns:
        --------
        dict : 所有模型的预测结果
        """
        results = {}

        if not os.path.exists(models_dir):
            return {"error": f"模型目录不存在: {models_dir}"}

        # 遍历目录中的所有.pkl文件
        for filename in os.listdir(models_dir):
            if filename.endswith(".pkl"):
                filepath = os.path.join(models_dir, filename)
                model_key = filename.replace(".pkl", "")

                prediction_result = self.predict_with_saved_model(filepath, X_new)
                results[model_key] = prediction_result

        return results
