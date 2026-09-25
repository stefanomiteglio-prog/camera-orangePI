import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("treeeyes")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/treeeyes.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)
