import csv
import os

from datasets import tqdm, load_dataset

from src.data.preprocess import logger
from src.models.base_model import BaseModel
from src.predict.ast_parser import check_cpp_code_ast

# Load model đã train
def load_model_and_dataset(datapath: str):
    # Load raw test
    logger.info(f"[UET] Đang load dataset từ from %s", datapath)
    dataset = load_dataset("json", data_files=datapath, split="train")
    num_samples = len(dataset)
    logger.info(f"[UET] Số lượng mẫu trong dataset: {num_samples}")

    return dataset

def evaluate_model(dataset, modelObject: BaseModel, outputFolder: str, limit: int = None):
    logger.info("[UET] Đang đánh giá mô hình...")
    logger.info(f"[UET] Số sample đánh giá: {limit}")
    # Xóa file cũ nếu có
    if os.path.exists(outputFolder):
        os.remove(outputFolder)

    with open(outputFolder, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Source", "Expected Target", "Predicted Target", "Check AST"])
        count = 0

        # Tao progress bar
        total = len(dataset) if limit is None else min(limit, len(dataset))
        pbar = tqdm(total=total, desc="Evaluating", unit="sample")

        for i, sample in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            try:
                source_text, predicted = modelObject.generate_from_sample(sample)
                ground_truth = str(sample["target"])
                check_ast = str(check_cpp_code_ast(predicted))

                writer.writerow([source_text, ground_truth, predicted, check_ast])
                f.flush()

                count += 1
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing sample at position {i}: {str(e)}")
                continue

        pbar.close()