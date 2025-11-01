"""Logging system để theo dõi pipeline."""
import logging
import os
from datetime import datetime


def setup_logger(name="pipeline", log_dir=None):
    """
    Setup logger với file output.
    
    Args:
        name: Tên logger
        log_dir: Thư mục lưu log (mặc định: data/logs/pipeline)
    
    Returns:
        logging.Logger: Logger instance
    """
    if log_dir is None:
        log_dir = os.path.join("data", "logs", "pipeline")
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Xóa handlers cũ (nếu có)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"📝 Log file: {log_file}")
    return logger
