import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs"):
    """Sets up logging to console and file."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"deep_prompt_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("DeepPromptSystem")

def save_to_md(content, filename="generated_prompt.md"):
    """Saves the content to a markdown file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        logging.info(f"Successfully saved output to {filename}")
    except Exception as e:
        logging.error(f"Failed to save output to {filename}: {e}")
