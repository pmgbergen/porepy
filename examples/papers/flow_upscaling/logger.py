import logging, sys

class Logger(object):

    def __init__(self, file_name="log.txt", stream=sys.stdout, level=logging.DEBUG):

        fmt = "%(asctime)s %(levelname)-8s %(message)s"
        date_fmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

        # file handler
        handler = logging.FileHandler(file_name, mode="w")
        handler.setFormatter(formatter)

        # screen handler
        screen_handler = logging.StreamHandler(stream=stream)
        screen_handler.setFormatter(formatter)

        self.logger = logging.getLogger()
        self.logger.setLevel(level)

        self.logger.addHandler(handler)
        self.logger.addHandler(screen_handler)

        # string to put before each message
        self.pre = ""

    def info(self, message):
        self.logger.info(self.pre + message)

    def add_tab(self):
        self.pre += "\t"

    def remove_tab(self):
        self.pre = self.pre[:-1]

logger = Logger()
