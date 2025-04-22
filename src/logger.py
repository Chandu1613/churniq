import logging
import os

os.makedirs('Logs', exist_ok=True)

logging.basicConfig(
    filename="/home/user/churniq/Logs/log_messages.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_message(message):
    logging.info(message)