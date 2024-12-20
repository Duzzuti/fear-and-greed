import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import utils
utils.fetch_all("data/", "2000-01-01")