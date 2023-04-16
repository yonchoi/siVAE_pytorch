from typing import Mapping, Optional, Sequence, Union
import json

class Configuration():

    def __init__(self,**kwargs):

        self.__dict__.update(**kwargs)

    def to_json(self,filename):

        with open(filename, "w") as outfile:
            json.dump(self.__dict__, outfile)
