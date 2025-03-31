"""
Đếm token trong file JSON dùng để train model.
Đếm source và target
"""
import json
import logging

import numpy as np
from src.utils.mylogger import logger


def count_tokens(json_path, tokenizer, source_str, target_str):
    # Đọc file JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.info(f"Lỗi khi đọc file JSON: {str(e)}")
        return

    logger.info(f"Tổng số mục trong file: {len(data)}")

    # Khởi tạo danh sách để lưu token
    all_source_tokens = []
    all_target_tokens = []

    # Phân tích từng mục
    for i, item in enumerate(data):
        # Đếm token cho source
        source = item.get(source_str, "")
        source_tokens = len(tokenizer.encode(source, add_special_tokens=True))
        all_source_tokens.append(source_tokens)

        # Đếm token cho target
        target = item.get(target_str, "")
        target_tokens = len(tokenizer.encode(target, add_special_tokens=True))
        all_target_tokens.append(target_tokens)

    # Chuyển sang numpy array để tính toán thống kê
    source_tokens_np = np.array(all_source_tokens)
    target_tokens_np = np.array(all_target_tokens)

    # Tính các chỉ số thống kê cơ bản
    source_stats = {
        "min": np.min(source_tokens_np),
        "max": np.max(source_tokens_np),
        "avg": np.mean(source_tokens_np),
        "q25": np.percentile(source_tokens_np, 25),
        "q50": np.percentile(source_tokens_np, 50),
        "q75": np.percentile(source_tokens_np, 75),
        "q90": np.percentile(source_tokens_np, 90),
        "q95": np.percentile(source_tokens_np, 95),
        "q99": np.percentile(source_tokens_np, 99)
    }

    target_stats = {
        "min": np.min(target_tokens_np),
        "max": np.max(target_tokens_np),
        "avg": np.mean(target_tokens_np),
        "q25": np.percentile(target_tokens_np, 25),
        "q50": np.percentile(target_tokens_np, 50),
        "q75": np.percentile(target_tokens_np, 75),
        "q90": np.percentile(target_tokens_np, 90),
        "q95": np.percentile(target_tokens_np, 95),
        "q99": np.percentile(target_tokens_np, 99)
    }

    # In kết quả tổng hợp
    logging.info("===== THỐNG KÊ TOKEN =====")

    logging.info("--- SOURCE TOKENS ---")
    logging.info(f"Min: {source_stats['min']}")
    logging.info(f"Max: {source_stats['max']}")
    logging.info(f"Avg: {source_stats['avg']:.2f}")
    logging.info(f"Q25: {source_stats['q25']}")
    logging.info(f"Q50 (trung vị): {source_stats['q50']}")
    logging.info(f"Q75: {source_stats['q75']}")
    logging.info(f"Q90: {source_stats['q90']}")
    logging.info(f"Q95: {source_stats['q95']}")
    logging.info(f"Q99: {source_stats['q99']}")

    logging.info("--- TARGET TOKENS ---")
    logging.info(f"Min: {target_stats['min']}")
    logging.info(f"Max: {target_stats['max']}")
    logging.info(f"Avg: {target_stats['avg']:.2f}")
    logging.info(f"Q25: {target_stats['q25']}")
    logging.info(f"Q50 (trung vị): {target_stats['q50']}")
    logging.info(f"Q75: {target_stats['q75']}")
    logging.info(f"Q90: {target_stats['q90']}")
    logging.info(f"Q95: {target_stats['q95']}")
    logging.info(f"Q99: {target_stats['q99']}")

    logging.info("===========================")

    return {
        "source_stats": source_stats,
        "target_stats": target_stats
    }


def count_token_for_both(json_path, tokenizer, source_str, target_str):
    """
    Đếm và thống kê tổng token cho source và target trong file JSON
    """
    # Đọc file JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.info(f"Lỗi khi đọc file JSON: {str(e)}")
        return None

    # Khởi tạo danh sách để lưu tổng token mỗi item
    total_tokens_per_item = []

    # Phân tích từng mục
    for item in data:
        # Đếm token cho source
        source = item.get(source_str, "")
        source_tokens = len(tokenizer.encode(source, add_special_tokens=True))

        # Đếm token cho target
        target = item.get(target_str, "")
        target_tokens = len(tokenizer.encode(target, add_special_tokens=True))

        # Tổng token mỗi item
        total_tokens_per_item.append(source_tokens + target_tokens)

    # Chuyển sang numpy array để tính toán thống kê
    total_tokens_np = np.array(total_tokens_per_item)

    # Tổng token
    total_tokens = np.sum(total_tokens_np)

    # Thống kê tổng token
    total_tokens_stats = {
        "min": np.min(total_tokens_np),
        "max": np.max(total_tokens_np),
        "avg": np.mean(total_tokens_np),
        "q25": np.percentile(total_tokens_np, 25),
        "q50": np.percentile(total_tokens_np, 50),
        "q75": np.percentile(total_tokens_np, 75),
        "q90": np.percentile(total_tokens_np, 90),
        "q95": np.percentile(total_tokens_np, 95),
        "q99": np.percentile(total_tokens_np, 99)
    }

    # In ra thông tin
    for key, value in total_tokens_stats.items():
        logger.info(f"{key.upper()}: {value}")

    # Trả về từ điển thống kê
    return {
        "total_tokens": total_tokens,
        "total_tokens_per_item_stats": total_tokens_stats
    }
