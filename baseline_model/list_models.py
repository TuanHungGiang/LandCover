import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_suite.models import MODEL_NAMES, model_info

print(json.dumps([model_info(name) for name in MODEL_NAMES], indent=2))
