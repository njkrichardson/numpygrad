import logging
from pathlib import Path

DEFAULT_STREAM_LOG_LEVEL = logging.INFO

PROJECT_DIR = Path(__file__).parent.parent.parent.resolve()

MEDIA_DIR = PROJECT_DIR / "media"

AUTO_MKDIR = (MEDIA_DIR,)

for dir in AUTO_MKDIR:
    dir.mkdir(exist_ok=True)
