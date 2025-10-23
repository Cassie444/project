# -*- coding: utf-8 -*-
"""
评估模块
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix, cohen_kappa_score
)


def evaluate_model(y_true, y_prob, threshold=0.5, model_name='Model'):
    """
    评估模型性能（固定阈值0.5）
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_prob : array-like
        预测概率
    threshold : float, default=0.5
        分类阈值
    model_name : str
        模型名称
    
    Returns:
    --------
    dict : 评估指标字典
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # 基础分类指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # 召回率
    
    # 混淆矩阵计算特异性
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp + 1e-12)
    
    # 平衡准确率
    balanced_accuracy = (sensitivity + specificity) / 2.0
    
    # F分数
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    
    # 概率相关指标
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan
        avg_precision = np.nan
    
    # 其他指标
    kappa = cohen_kappa_score(y_true, y_pred)
    pred_positive_rate = np.mean(y_pred)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'f05_score': f05,
        'auc': roc_auc,
        'avg_precision': avg_precision,
        'kappa': kappa,
        'pred_positive_rate': pred_positive_rate,
    }


def calculate_composite_score(metrics, weights=None):
    """
    计算复合评分
    
    Parameters:
    -----------
    metrics : dict
        评估指标字典
    weights : dict, optional
        各指标权重，默认为 {'f1_score': 0.5, 'precision': 0.5}
    
    Returns:
    --------
    float : 复合评分
    """
    if weights is None:
        weights = {'f1_score': 0.5, 'precision': 0.5}
    
    score = 0.0
    for metric, weight in weights.items():
        if metric in metrics:
            score += weight * metrics[metric]
    
    return score