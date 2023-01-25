from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'raw_data'
PREPROCESSING_DIR = ROOT_DIR / 'preprocessing'
HOLD_FEATURES_PATH = DATA_DIR / 'HoldFeature2016.csv'
HOLD_FEATURES_LEFT_HAND_PATH = DATA_DIR / 'HoldFeature2016LeftHand.csv'
HOLD_FEATURES_RIGHT_HAND_PATH = DATA_DIR / 'HoldFeature2016RightHand.csv'
SCRAPE_DATA_PATH = DATA_DIR / 'moonGen_scrape_2016_cp'
