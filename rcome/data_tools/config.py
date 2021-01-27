import io
import os
import json

class ConfigurationFile(object):
    def __init__(self, filepath, mod="fill"):
        self.filepath = filepath
        self.mod = mod
        self.content = {}
        # create the file if does not exist
        if(not os.path.exists(self.filepath)):
            print("Creating the file at ", self.filepath)

            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            self.clear()

        # update the object in reading the file
        self.update()

    # clear all the config file 
    def clear(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        self.content = {}
        ConfigurationFile._writing_json(self.filepath, {}, mod="w+")
        self.last_correct_save = {}
    # return a list containg all the config file existing keys 
    def keys(self):
        return [k for k in self.content]

    # update the config file
    def update(self):
        file_content = ConfigurationFile._reading_json(self.filepath)
        self.last_correct_save = file_content
        self.content = dict(file_content, **self.content)
        try:
            ConfigurationFile._writing_json(self.filepath, self.content)
        except Exception:
            ConfigurationFile._writing_json(self.filepath,  self.last_correct_save)
            print("Error writing JSON")
            raise Exception
           
    
    def set_mod(self, mod=""):
        self.mod = mod

    def __getitem__(self, index ):

        if(index not in self.content and self.mod=="fill"):
            value = input("A value is necessary for variable "+str(index)+"\n value : ")
            self.content[index] = value
            self.update()
        return  self.content[index] if(index in self.content) else None

    def __setitem__(self, index, value):
        self.content[index] = value
        self.update()

    def __str__(self):
        _cf_str = ""
        max_len = max([0]+[len(k) for k in self.content])
        for k, v in self.content.items():
            _cf_str += (k+(" "*(max_len - len(k)))+" : "+str(v)) + '\n'
        return _cf_str

    def __repr__(self):
        return self.__str__()

    def __contains__(self, b):
        return b in self.content

    @staticmethod
    def _reading_json(filepath):
        with io.open(filepath, 'r') as file_sconf:
            return json.load(file_sconf)

    @staticmethod
    def _writing_json(filepath, dictionary, mod='w'):
        with io.open(filepath, mod) as file_sconf:
            file_sconf.write(json.dumps(dictionary, sort_keys=True, indent=4))
        
