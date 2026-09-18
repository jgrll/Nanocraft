from loguru import logger
from datetime import datetime
import glob
import os

nkeep = 3
log_filename = f"nanocraft_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logger.remove()
logger.add(
    log_filename,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>"
)

log_files = sorted(glob.glob("nanocraft_*.log"), key=os.path.getmtime, reverse=True)
for old_file in log_files[nkeep:]:
    os.remove(old_file)

logger.info(f"Logger initialized: {log_filename}")
