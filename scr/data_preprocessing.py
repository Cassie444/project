# -*- coding: utf-8 -*-
"""
数据预处理模块
"""

import pandas as pd
from config import DATA_PATHS, final_vars, all_features, CONTINUOUS_VARS


def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    df = pd.read_stata(file_path)

    # 创建衍生变量
    df = create_derived_features(df)

    return df


def create_derived_features(df):
    """
    创建衍生特征
    """
    df = df.copy()

    # 症状相关特征
    for i in range(1, 10):
        df[f"symptom_ss{i}"] = df[f"symptom_all_s{i}"]
        df.loc[df[f"symptom_s{i}"] == 1, f"symptom_ss{i}"] = 2

    # 症状汇总
    tmp_symptom = [i for i in df.columns.tolist() if "symptom_s" in i]
    df["is_symptom"] = (df[tmp_symptom].sum(axis=1) > 0).astype(int)

    # 症状-住院复合变量
    df["symptom_hospitalized"] = df["is_symptom"]
    df.loc[
        (df["hospitalized"] == 0) & (df["is_symptom"] == 1), "symptom_hospitalized"
    ] = 1
    df.loc[
        (df["hospitalized"] == 1) & (df["ventilator"] == 0), "symptom_hospitalized"
    ] = 2
    df.loc[
        (df["hospitalized"] == 1) & (df["ventilator"] == 1), "symptom_hospitalized"
    ] = 3

    # 室友相关
    df["roomate"] = df["living_alone"]
    df.loc[df["housemate_symptom"] == 1, "roomate"] = 2

    # 口罩相关
    df["masking2"] = df["not_going_out"]
    df.loc[df["masking"] == 0, "masking2"] = 2

    # 疫苗相关
    df["vax"] = df["vaxed_not_expired"] + 1
    df.loc[(df["vaxed_not_expired"] == 0) & (df["vaxed_but_expired"] == 0), "vax"] = 0

    # ADL相关
    tmp_adl = [i for i in df.columns.tolist() if "ADL_" in i and "IADL" not in i]
    df["is_adl"] = (df[tmp_adl].sum(axis=1) > 0).astype(int)

    tmp_iadl = [i for i in df.columns.tolist() if "IADL" in i]
    df["is_iadl"] = (df[tmp_iadl].sum(axis=1) > 0).astype(int)

    df["adl_iadl"] = df["is_iadl"]
    df.loc[df["is_adl"] == 1, "adl_iadl"] = 2

    # 不健康习惯
    tmp_unhealthy = ["smoking", "overdrinking", "unhealthy_diet"]
    df["unhealthy_habbits"] = (df[tmp_unhealthy].sum(axis=1) > 0).astype(int)

    # 填充缺失值
    df["Chro_disease_any"] = df["Chro_disease_any"].fillna(0)

    return df


def get_datasets():
    """
    获取所有处理后的数据集
    """
    datasets = {}

    for name, path in DATA_PATHS.items():
        df = load_and_preprocess_data(path)
        df_clean = df[final_vars].dropna(subset=all_features).astype(int)
        datasets[name] = {"data": df_clean, "features": all_features}

    return datasets


def get_categorical_feature():
    """
    获取类别特征的索引（除Age外的所有特征）
    """
    categorical_indices = []
    categorical_cols = []
    for i, col in enumerate(all_features):
        if col not in CONTINUOUS_VARS:
            categorical_indices.append(i)
            categorical_cols.append(col)
    return categorical_indices, categorical_cols
