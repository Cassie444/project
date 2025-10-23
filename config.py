# -*- coding: utf-8 -*-
"""
配置文件
"""

import warnings

warnings.filterwarnings("ignore")

# 随机种子
RANDOM_STATE = 101

# 数据路径
DATA_PATHS = {
    "dataset1": "../working_data/dataset1_ml.dta",
    "dataset2": "../working_data/dataset2_ml.dta",
}

# 变量定义
TARGET_VAR = ["positive"]
ID_VARS = ["episode", "primkey", "suspected", "died"]
DEMO_VARS = ["Female", "Married", "Rural", "Age", "urban_res", "Edu_Group"]
BEHAVIOR_VARS = [
    "symptom_hospitalized",
    "roomate",
    "contact_symptom",
    "masking2",
    "isolation",
    "distancing",
    "vax",
    "unhealthy_habbits",
]
DISEASE_ADL_VARS = ["Chro_disease_any", "adl_iadl"]
SYMPTOM_VARS = [f"symptom_ss{i}" for i in range(1, 10)]

all_features = DEMO_VARS + SYMPTOM_VARS + BEHAVIOR_VARS + DISEASE_ADL_VARS
final_vars = TARGET_VAR + all_features

# 连续变量（需要标准化）
CONTINUOUS_VARS = ["Age"]

# 测试集比例
TEST_SIZE = 0.2

# 交叉验证折数
CV_FOLDS = 5

# 默认超参数优化试验次数
DEFAULT_N_TRIALS = 100

# 不平衡数据处理阈值
IMBALANCE_THRESHOLD = 10  # pos_weight > 10时使用特殊处理


def load_and_prepare_data():
    """加载并准备数据集"""
    import pandas as pd

    datasets = {}
    for name, path in DATA_PATHS.items():
        df = pd.read_stata(path)
        # 创建衍生变量
        df = create_derived_features(df)
        df_clean = df[final_vars].dropna(subset=all_features).astype(int)
        datasets[name] = {"data": df_clean, "features": all_features}
    return datasets


def get_categorical_feature(feature_cols):
    """获取类别特征的索引"""
    categorical_features = [col for col in feature_cols if col not in CONTINUOUS_VARS]
    return [i for i, col in enumerate(feature_cols) if col in categorical_features]


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
