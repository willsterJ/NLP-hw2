from model import Model
import re
import numpy as np
from Data_Point import Data_Point


class Unigram(Model):
    def __init__(self, data):
        super().__init__(data)

    def find_features_and_labels(self):
        """
        :return: pair dictionaries of character keys and labels
        """
        data = self.data
        char_dict = {}
        char_index = 0
        label_dict = {}  # stores label and their corresponding index (i.e. drugs:0, company:1, ...)
        label_count = 0

        input_list = []  # list of dict of chars stores the char features per input and corresponding output

        # parse each character into a set

        count = 0  # TODO debug

        for item in data:
            count += 1
            if count > 10000:
                break

            input = str(item[1])
            input = re.sub(r'[^\w]', '', input)  # remove spaces
            input_l = list(input)
            label = str(item[0])

            input_char_dict = {}

            data_point = Data_Point()

            for c in input_l:
                if c not in input_char_dict:
                    input_char_dict[c] = 1  # add char to 'local' char dict
                else:
                    input_char_dict[c] += 1

                if c not in char_dict:
                    char_dict[c] = char_index  # add character to 'global' char dict
                    char_index += 1

            if label not in label_dict.keys():
                label_dict[label] = label_count  # also add to label dict
                label_count += 1

            data_point.true_label = label
            data_point.features_dict = input_char_dict
            self.data_points_list.append(data_point)

        self.feature_dict = char_dict
        self.label_dict = label_dict
