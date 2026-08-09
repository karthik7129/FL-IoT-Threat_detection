import logging
import os
from typing import Optional

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file))

    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
    return logging.getLogger()
