# -*- coding: utf-8 -*-
"""
概率校准模块

概率校准的目的和重要性：
=======================

在二分类问题中，模型输出的概率并不总是能准确反映真实的预测置信度。例如：
- 如果模型预测某个样本为正类的概率是0.8，理想情况下，在所有预测概率为0.8的样本中，
  应该有80%的样本确实是正类。
- 但实际上，很多模型（尤其是树模型和SVM）的概率输出是"未校准"的，
  可能预测0.8的样本中只有60%或90%是正类。

概率校准的应用场景：
==================
1. 当需要预测概率具有真实解释性时（如医疗诊断、风险评估）
2. 当模型输出需要与其他概率性预测结合时
3. 当需要设置置信度阈值进行决策时
4. 当模型用于成本敏感的场景时

两种主要的校准方法：
==================
1. Platt Scaling (sigmoid方法)：
   - 使用逻辑回归将预测概率映射到校准概率
   - 假设校准函数是sigmoid形状
   - 适用于小样本，对训练数据的要求较低
   - 当真实的校准曲线接近sigmoid时效果好

2. Isotonic Regression (isotonic方法)：
   - 使用保序回归，是一种非参数方法
   - 不假设特定的函数形状，更加灵活
   - 适用于较大样本
   - 可以处理更复杂的校准曲线形状

信息泄露的防止：
==============
校准过程中防止信息泄露的关键是确保用于拟合校准器的数据不能用于最终的模型评估。
本模块通过以下方式防止信息泄露：
1. 使用交叉验证获取"未见过"的预测概率来拟合校准器
2. 校准器的拟合和最终评估使用完全不同的数据集
3. 在超参数优化中，校准也在每个CV fold内独立进行
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss


class SmartProbabilityCalibrator:
    """
    智能概率校准器 - 自动选择校准方法并防止信息泄露
    """

    def __init__(self, method="auto", cv=3, auto_decision_threshold=0.1):
        """
        Parameters:
        -----------
        method : str
            校准方法：'auto', 'sigmoid', 'isotonic', 'none'
            - 'auto': 自动选择最佳方法
            - 'sigmoid': Platt scaling
            - 'isotonic': Isotonic regression
            - 'none': 不进行校准
        cv : int
            交叉验证折数
        auto_decision_threshold : float
            自动决策的ECE阈值，超过此值才进行校准
        """
        self.method = method
        self.cv = cv
        self.auto_decision_threshold = auto_decision_threshold
        self.calibrator = None
        self.chosen_method = None
        self.is_fitted = False
        self.should_calibrate_flag = False
        self.ece_before = None
        self.ece_after = None

    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        """
        计算Expected Calibration Error (ECE)
        ECE衡量预测概率与真实频率之间的差异
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _should_calibrate(self, y_true, y_prob):
        """
        判断是否需要校准
        """
        # 检查样本量是否足够
        if len(y_prob) < 100:
            return False

        # 检查预测概率的分布
        if len(np.unique(y_prob)) < 10:  # 概率值太少
            return False

        # 计算ECE
        ece = self._calculate_ece(y_true, y_prob)
        self.ece_before = ece
        return ece > self.auto_decision_threshold

    def _choose_best_method(self, y_true, y_prob):
        """
        自动选择最佳校准方法
        基于交叉验证比较不同方法的性能
        """
        if len(y_prob) < 500:
            # 小样本倾向于使用Platt scaling
            return "sigmoid"

        # 大样本情况下，比较两种方法的性能
        try:
            # 简单的holdout验证来选择方法
            n_test = len(y_prob) // 4
            indices = np.random.permutation(len(y_prob))
            train_idx, test_idx = indices[n_test:], indices[:n_test]

            y_train, y_test = y_true[train_idx], y_true[test_idx]
            prob_train, prob_test = y_prob[train_idx], y_prob[test_idx]

            # 测试sigmoid方法
            sigmoid_cal = LogisticRegression()
            sigmoid_cal.fit(prob_train.reshape(-1, 1), y_train)
            prob_sigmoid = sigmoid_cal.predict_proba(prob_test.reshape(-1, 1))[:, 1]

            # 测试isotonic方法
            isotonic_cal = IsotonicRegression(out_of_bounds="clip")
            isotonic_cal.fit(prob_train, y_train)
            prob_isotonic = isotonic_cal.predict(prob_test)

            # 比较Brier Score（越小越好）
            brier_sigmoid = brier_score_loss(y_test, prob_sigmoid)
            brier_isotonic = brier_score_loss(y_test, prob_isotonic)

            return "sigmoid" if brier_sigmoid < brier_isotonic else "isotonic"

        except Exception:
            # 出错时默认使用sigmoid
            return "sigmoid"

    def fit(self, base_model, X, y):
        """
        拟合校准器
        """
        # 使用交叉验证获取未偏概率（防止信息泄露）
        try:
            y_prob_cv = cross_val_predict(
                base_model, X, y, cv=self.cv, method="predict_proba"
            )[:, 1]
        except Exception:
            # 如果predict_proba失败，尝试decision_function
            try:
                y_scores = cross_val_predict(
                    base_model, X, y, cv=self.cv, method="decision_function"
                )
                # 将decision_function的输出转换为概率
                y_prob_cv = 1 / (1 + np.exp(-y_scores))  # sigmoid变换
            except Exception:
                print("警告：无法获取概率预测，跳过校准")
                self.should_calibrate_flag = False
                self.chosen_method = "none"
                self.is_fitted = True
                return self

        # 判断是否需要校准
        if self.method == "none":
            self.should_calibrate_flag = False
        else:
            self.should_calibrate_flag = self._should_calibrate(y, y_prob_cv)

        if not self.should_calibrate_flag:
            self.calibrator = None
            self.chosen_method = "none"
            self.ece_after = self.ece_before
            self.is_fitted = True
            return self

        # 选择校准方法
        if self.method == "auto":
            self.chosen_method = self._choose_best_method(y, y_prob_cv)
        else:
            self.chosen_method = self.method

        # 创建并拟合校准器
        if self.chosen_method == "sigmoid":
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob_cv.reshape(-1, 1), y)
        elif self.chosen_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_prob_cv, y)

        self.is_fitted = True
        return self

    def calibrate_probabilities(self, y_prob):
        """
        校准概率
        """
        if not self.is_fitted:
            raise ValueError("校准器尚未拟合")

        if not self.should_calibrate_flag or self.calibrator is None:
            return y_prob

        if self.chosen_method == "sigmoid":
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        elif self.chosen_method == "isotonic":
            return self.calibrator.predict(y_prob)
        else:
            return y_prob

    def get_calibration_info(self):
        """
        获取校准信息
        """
        if not self.is_fitted:
            return {
                "calibration_method": "not_fitted",
                "needs_calibration": None,
                "ece_before": None,
                "ece_after": None,
            }

        return {
            "calibration_method": self.chosen_method,
            "needs_calibration": self.should_calibrate_flag,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
        }


def auto_calibrate_and_predict(model, X_train, y_train, X_test, y_test=None):
    """
    自动校准并预测的统一接口

    Parameters:
    -----------
    model : fitted sklearn estimator
        已训练的模型
    X_train : array-like
        训练特征（用于拟合校准器）
    y_train : array-like
        训练标签（用于拟合校准器）
    X_test : array-like
        测试特征
    y_test : array-like, optional
        测试标签（用于计算校准后的ECE）

    Returns:
    --------
    dict : 包含预测概率和校准信息的字典
    """
    # 获取原始预测概率
    y_prob_raw = model.predict_proba(X_test)[:, 1]

    # 创建校准器并拟合
    calibrator = SmartProbabilityCalibrator(method="auto", cv=3)
    calibrator.fit(model, X_train, y_train)

    # 校准概率
    y_prob_calibrated = calibrator.calibrate_probabilities(y_prob_raw)

    # 计算校准后的ECE（如果提供了测试标签）
    if y_test is not None and calibrator.should_calibrate_flag:
        ece_after = calibrator._calculate_ece(y_test, y_prob_calibrated)
        calibrator.ece_after = ece_after
    else:
        calibrator.ece_after = calibrator.ece_before

    # 获取校准信息
    calibration_info = calibrator.get_calibration_info()

    return {
        "y_prob_raw": y_prob_raw,
        "y_prob_calibrated": y_prob_calibrated,
        "calibration_info": calibration_info,
    }


def evaluate_calibration(y_true, y_prob, n_bins=10):
    """
    评估概率校准质量
    """
    # 计算ECE和MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0  # Expected Calibration Error
    mce = 0  # Maximum Calibration Error

    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()

            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += calibration_error * prop_in_bin
            mce = max(mce, calibration_error)

            bin_data.append(
                {
                    "bin_lower": bin_lower,
                    "bin_upper": bin_upper,
                    "accuracy": accuracy_in_bin,
                    "confidence": avg_confidence_in_bin,
                    "count": in_bin.sum(),
                    "calibration_error": calibration_error,
                }
            )

    # 计算其他校准指标
    try:
        brier_score = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)
    except Exception:
        brier_score = np.nan
        logloss = np.nan

    return {
        "ece": ece,  # Expected Calibration Error (越小越好)
        "mce": mce,  # Maximum Calibration Error (越小越好)
        "brier_score": brier_score,  # Brier Score (越小越好)
        "log_loss": logloss,  # Log Loss (越小越好)
        "bin_data": bin_data,  # 每个bin的详细数据
        "needs_calibration": ece > 0.1,  # 是否建议校准
    }
