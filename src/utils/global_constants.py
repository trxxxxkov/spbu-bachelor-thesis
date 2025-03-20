"""Global constants and constant structures, including parameters for all
experiments mentioned in the thesis"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "src", "data")
