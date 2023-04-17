from typing import Mapping, Optional, Sequence, Union
import json

class Configuration():

    def __init__(self,**kwargs):

        self.__dict__.update(**kwargs)

    def to_json(self,filename):

        with open(filename, "w") as outfile:
            json.dump(self.__dict__, outfile)

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.
        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.
        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        # if use_diff is True:
        #     config_dict = self.to_diff_dict()
        # else:
        config_dict = self.__dict__
        config_dict = {k:v for k,v in config_dict.items() if isinstance(v,str)}
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __index__(self, kwarg):
        return self.__dict__[kwarg]
