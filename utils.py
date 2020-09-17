import logging
from os import path, getpid

format = "[%(process)d] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logger = logging.getLogger()


def get_ram_usage():
    """Get current process physical memory usage in GB"""
    
    process = psutil.Process(getpid())
    ram_gb = round(process.memory_info().rss / 1024.**3, 2)
    return ram_gb


def log_ram():
    """Log RAM usage with a global logger"""

    logger.info(f"Current memory usage: {get_ram_usage()} GB")
