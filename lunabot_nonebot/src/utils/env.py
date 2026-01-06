from os import getenv
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = getenv('CONFIG_DIR') or "config/"
DATA_DIR = getenv('DATA_DIR') or "data/"
SEKAI_API_BASE_PATH = getenv('SEKAI_API_BASE_PATH')
SEKAI_ASSET_BASE_PATH = getenv('SEKAI_ASSET_BASE_PATH')
CONFIG_UPDATE_CHECK_INTERVAL = float(getenv('CONFIG_UPDATE_CHECK_INTERVAL') or 3.0)