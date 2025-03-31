from src.data.preprocess import preprocess_dataset2
from src.predict.evaluate import load_model_and_dataset, evaluate_model
from src.utils.model_utils import load_model_by_type
from src.utils.mylogger import logger
from src.config.config import MODEL, TRAINSET_DATA_PATH_PROCESS, VALIDATIONSET_DATA_PATH_PROCESS, TRAINSET_RAW, \
    VALIDATIONSET_RAW, TARGET_SELETCTION_STRATEGIES, OPTIMIZE_TARGET_STRATEGY, OUTPUT_VALIDATIONSET_CSV, \
    OUTPUT_TRAINSET_CSV
from src.utils.token_statistics import count_token_for_both, count_tokens


def main():
    logger.info(f"[UET] Load mô hình {MODEL} và tokenizer tương ứng")
    modelObject = load_model_by_type(MODEL)

    # TIEN XU LY DU LIEU
    logger.info(
        f"[UET] Phân tích tập học thô {TRAINSET_RAW} để tạo tập học tinh chế.")
    preprocess_dataset2(TRAINSET_RAW, TRAINSET_DATA_PATH_PROCESS, modelObject.tokenizer,
                        optimize_target_strategy=OPTIMIZE_TARGET_STRATEGY)

    logger.info("")
    logger.info(f"[UET] Phân tích tập test thô {VALIDATIONSET_RAW} để tạo tập test tinh chế")
    preprocess_dataset2(VALIDATIONSET_RAW, VALIDATIONSET_DATA_PATH_PROCESS, modelObject.tokenizer,
                        optimize_target_strategy=TARGET_SELETCTION_STRATEGIES.NONE)
    logger.info("[UET] Tiền xử lý hoàn tất!")

    logger.info("\n")

    # THONG KE
    logger.info(
        f"[UET] Thống kê tổng kích thước source + target TRƯỚC khi CUT")
    count_token_for_both(TRAINSET_DATA_PATH_PROCESS, modelObject.tokenizer, "source_before_cut", "target_before_cut")

    logger.info(
        f"[UET] Thống kê tổng kích thước source + target SAU khi CUT")
    count_token_for_both(TRAINSET_DATA_PATH_PROCESS, modelObject.tokenizer, "source", "target")

    logger.info(
        f"[UET] Thống kê tập học của mô hình ({TRAINSET_DATA_PATH_PROCESS}) TRƯỚC khi CUT")
    count_tokens(TRAINSET_DATA_PATH_PROCESS, modelObject.tokenizer, "source_before_cut", "target_before_cut")

    logger.info(
        f"[UET] Thống kê tập học của mô hình ({TRAINSET_DATA_PATH_PROCESS}) SAU khi CUT")
    count_tokens(TRAINSET_DATA_PATH_PROCESS, modelObject.tokenizer, "source", "target")

    logger.info(
        f"[UET] Thống kê tập test của mô hình ({VALIDATIONSET_DATA_PATH_PROCESS}) TRƯỚC khi CUT")
    count_tokens(VALIDATIONSET_DATA_PATH_PROCESS, modelObject.tokenizer, "source_before_cut", "target_before_cut")

    logger.info(
        f"[UET] Thống kê tập test của mô hình ({VALIDATIONSET_DATA_PATH_PROCESS}) SAU khi CUT")
    count_tokens(VALIDATIONSET_DATA_PATH_PROCESS, modelObject.tokenizer, "source", "target")

    #  Đánh giá mô hình
    logger.info("\n")
    logger.info("[UET] Đánh giá mô hình trên tập validation. Tôi sử dụng trực tiếp mô hình vừa học.")
    _, dataset = load_model_and_dataset(datapath=VALIDATIONSET_DATA_PATH_PROCESS)
    evaluate_model(dataset, modelObject, outputFolder=OUTPUT_VALIDATIONSET_CSV)
    logger.info("[UET] Hoàn tất đánh giá trên tập validation!")

    logger.info("\n")
    logger.info("[UET] Đánh giá mô hình trên tập training. Tôi sử dụng trực tiếp mô hình vừa học.")
    _, dataset = load_model_and_dataset(datapath=TRAINSET_DATA_PATH_PROCESS)
    evaluate_model(dataset, modelObject, outputFolder=OUTPUT_TRAINSET_CSV)
    logger.info("[UET] Hoàn tất đánh giá trên tập training!")

if __name__ == '__main__':
    main()