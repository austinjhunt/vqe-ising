import os 
LOG_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_FOLDER_PATH, exist_ok=True) 