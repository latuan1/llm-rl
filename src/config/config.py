import os
from datetime import datetime
from enum import Enum

from src.utils.utils import normalize_path


class CLANG_PATH:
    WINDOW = r"C:\Users\Admin\PycharmProjects\llm-rl\clang+llvm-19.1.7-x86_64-pc-windows-msvc\bin"
    UBUNTU = "/usr/lib/x86_64-linux-gnu/libclang-11.so.1"

class MODEL_NAME:
    CODET5_BASE = "codet5-base"

class COMMENT_REMOVAL(Enum):
    AST = 0
    REGREX = 1

class TARGET_SELETCTION_STRATEGIES:  # Cấu hình tiền xử lý thuộc tính TD.
    NONE = "Không "  # Không xử lý
    SORT_BY_TOKEN_AND_CUTOFF = "SORT_BY_TOKEN_AND_CUTOFF"  # Khi tiền xử lý, với một danh sách test case -> sort theo số token tăng dần, và lấy n test case để tổng token <= max_target_length

class MASKING_STRATEGIES(Enum):
    NONE = 0
    RANDOM = 1

max_source_length = 32
max_target_length = 32

REMOVE_COMMENT_MODE = COMMENT_REMOVAL.AST

MASKING_SOURCE = MASKING_STRATEGIES.NONE

MODEL = MODEL_NAME.CODET5_BASE

OPTIMIZE_TARGET_STRATEGY = TARGET_SELETCTION_STRATEGIES.SORT_BY_TOKEN_AND_CUTOFF

PROJECT_PATH = "/workspace/llm-rl"
MAIN_OUTPUT_PATH = normalize_path(f"{PROJECT_PATH}/aka-output")
if not os.path.exists(MAIN_OUTPUT_PATH):
    os.makedirs(MAIN_OUTPUT_PATH)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
OUTPUT_PATH = normalize_path(f"{MAIN_OUTPUT_PATH}/{timestamp}")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

TRAINSET_RAW = normalize_path(f"{PROJECT_PATH}/data/trainset/raw")
TRAINSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_trainset.json")

VALIDATIONSET_RAW = normalize_path(f"{PROJECT_PATH}/data/validation/raw")
VALIDATIONSET_DATA_PATH_PROCESS = normalize_path(f"{OUTPUT_PATH}/processed_validationset.json")

OUTPUT_VALIDATIONSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_validationset.csv")
OUTPUT_TRAINSET_CSV = normalize_path(f"{OUTPUT_PATH}/output_trainingset.csv")