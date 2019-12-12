from model import Model
import re
import numpy as np
from Data_Point import Data_Point


class Unigram(Model):
    def __init__(self, data, valid_set, test_set):
        super().__init__(data, valid_set, test_set)
        self.find_features_and_labels()
        self.valid_data_points_list = self.validation_test_features(valid_set)
        self.test_data_points_list = self.validation_test_features(test_set)

    def find_features_and_labels(self):
        """
        """
        data = self.data
        char_dict = {}
        char_index = 0
        label_dict = {}  # stores label and their corresponding index (i.e. drugs:0, company:1, ...)
        label_count = 0

        # parse each character into a set
        for item in data:
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

            data_point.true_label_index = label_dict[label]  # TODO changed label to label_index
            data_point.features_dict = input_char_dict
            self.data_points_list.append(data_point)

        self.feature_dict = char_dict
        self.label_dict = label_dict

    def validation_test_features(self, data_set):
        output_list = []
        for item in data_set:
            input = str(item[1])
            input = re.sub(r'[^\w]', '', input)  # remove spaces
            input_l = list(input)
            label = str(item[0])

            input_char_dict = {}

            data_point = Data_Point()

            for c in input_l:
                if c in self.feature_dict:
                    if c not in input_char_dict:
                        input_char_dict[c] = 1
                    else:
                        input_char_dict[c] += 1

            data_point.features_dict = input_char_dict
            if label != '':
                data_point.true_label_index = self.label_dict[label]
            output_list.append(data_point)
        return output_list

