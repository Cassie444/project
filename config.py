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
