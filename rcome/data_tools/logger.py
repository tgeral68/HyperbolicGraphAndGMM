from rcome.data_tools import config
import os

class JSONLogger(config.ConfigurationFile):
    def __init__(self, filepath, mod="fill"):
        super(JSONLogger, self).__init__(filepath, mod=mod)
        if(mod!="continue" and mod!="fill"):
            self.clear()
    def append(self, dictionary_like):
        for k, v in dictionary_like.items():
            self.__setitem__(k, v)
    