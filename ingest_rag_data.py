import sys
import json
import logging
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(str(Path(__file__).parent.parent))

from config import Config


import argparse

# Initialize parser
parser = argparse.ArgumentParser(description="Process some command-line arguments.")

# Add arguments
parser.add_argument("--file", type=str, required=False, help="Enter file path to ingest")
parser.add_argument("--dir", type=str, required=False, help="Enter directory path of files to ingest")

# Parse arguments
args = parser.parse_args()

# Load configuration
config = Config()


