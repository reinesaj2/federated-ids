import sys
from pathlib import Path

# Add the project root directory to sys.path to allow imports from the root
# and scripts/ packages when running tests from the tests/ directory.
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
