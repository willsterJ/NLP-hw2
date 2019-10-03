from abc import ABC, abstractmethod
import load_data
import config
import re
import numpy as np

# abstract class
class Model:
    def __init__(self,data):
        self.data = data

    def generate_features(self):
        pass
    def objective_function(self):

class Unigram(Model):
    def __init__(self, data):
        super().__init__(data)

    def generate_features(self):
        """
        :return: dictionary of character keys
        """
        data = self.data
        char_dict = {}

        # parse each character into a set
        for item in data:
            s = str(item[1])
            s = re.sub(r'[^\w]', '', s)
            l = list(s)
            for c in l:
                char_dict[c] = 0  # add character to set
        return char_dict





data = load_data.LoadData()
data = data.read_file(config.Config.train_data)
model = Unigram(data)
model.generate_features()
