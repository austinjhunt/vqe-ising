import logging 
from conf import LOG_FOLDER_PATH

class Base:
    def __init__(self, name: str = '', verbose: bool = False): 
            self.name = name
            self.verbose = verbose 
            self.setup_logging() 
    
    def setup_logging(self):
        """ set up self.logger for Driver logging """ 
        self.logger = logging.getLogger(self.name)
        format = "[%(prefix)s - %(filename)s:%(lineno)s - %(funcName)3s() ] %(message)s"
        formatter = logging.Formatter(format)
        handlerStream = logging.StreamHandler()
        handlerStream.setFormatter(formatter) 
        self.logger.addHandler(handlerStream)  
        
        handlerFile = logging.FileHandler(f'{LOG_FOLDER_PATH}/{self.name}.log')
        handlerFile.setFormatter(formatter) 
        self.logger.addHandler(handlerFile)  
        if self.verbose:
            self.logger.setLevel(logging.DEBUG) 
        else:
            self.logger.setLevel(logging.INFO)

    def debug(self, msg):
        self.logger.debug(msg, extra={'prefix': self.name}, stacklevel=2)

    def info(self, msg):
        self.logger.info(msg, extra={'prefix': self.name}, stacklevel=2)

    def error(self, msg):
        self.logger.error(msg, extra={'prefix': self.name}, stacklevel=2)

