from model import Model
import re
import numpy as np
from Data_Point import Data_Point


class BiTrigram(Model):
    def __init__(self, data, valid_set, test_set):
        super().__init__(data, valid_set, test_set)
        self.N = 3
        self.feature_list = []

        self.find_features_and_labels()
        self.find_validation_features()

    def find_features_and_labels(self):
        data = self.data
        # vars
        global_feat_dict = {}
        global_feat_index = 0
        label_dict = {}
        label_index = 0

        for item in data:  # for each item...
            # get data
            label = item[0]
            input = str(item[1])

            local_feat_dict = {}  # local dict to store features for each item

            input_list = list(input)  # convert string to list of characters
            # find all n-grams of the string
            for i in range(0, len(input_list) - self.N + 1):
                s = input_list[i]
                if s == ' ':
                    continue
                for j in range(1, self.N):
                    if input_list[j] == ' ':  # stop when whitespace encountered
                        break
                    s = s + '' + input_list[i + j]

                    if s not in local_feat_dict:
                        local_feat_dict[s] = 1
                    else:
                        local_feat_dict[s] += 1

                    if s not in global_feat_dict:
                        global_feat_dict[s] = global_feat_index
                        global_feat_index += 1

            if label not in label_dict:
                label_dict[label] = label_index
                label_index += 1

            data_point = Data_Point()
            data_point.true_label_index = label_dict[label]
            data_point.features_dict = local_feat_dict
            self.data_points_list.append(data_point)

        self.feature_dict = global_feat_dict
        self.label_dict = label_dict

    def find_validation_features(self):
        for item in self.valid_set:  # for each item...
            # get data
            label = item[0]
            input = str(item[1])

            local_feat_dict = {}  # local dict to store features for each item

            input_list = list(input)  # convert string to list of characters
            # find all n-grams of the string
            for i in range(0, len(input_list) - self.N + 1):
                s = input_list[i]
                if s == ' ':
                    continue
                for j in range(1, self.N):
                    if input_list[j] == ' ':  # stop when whitespace encountered
                        break
                    s = s + '' + input_list[i + j]

                    if s not in local_feat_dict:
                        local_feat_dict[s] = 1
                    else:
                        local_feat_dict[s] += 1

            data_point = Data_Point()
            data_point.true_label_index = self.label_dict[label]
            self.valid_data_points_list.append(data_point)