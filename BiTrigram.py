from model import Model
import re
import numpy as np
from Data_Point import Data_Point


class BiTrigram(Model):
    def __init__(self, data):
        super().__init__(data)

    def find_features_and_labels(self):
        data = self.data

        global_feat_dict = {}
        global_feat_index = 0
        label_dict = {}
        label_index = 0

        for item in data:
            label = item[0]
            input = str(item[1])

            local_feat_dict = {}

            for i in range(0, len(input) - 2):
                trigram = input[i:i + 3]
                if trigram not in local_feat_dict:
                    local_feat_dict[trigram] = 1
                else:
                    local_feat_dict[trigram] += 1

                if trigram not in global_feat_dict:
                    global_feat_dict[trigram] = global_feat_index
                    global_feat_index += 1

                if label not in label_dict:
                    label_dict[label] = label_index
                    label_index += 1

            data_point = Data_Point()
            data_point.true_label = label
            data_point.features_dict = local_feat_dict
            self.data_points_list.append(data_point)

        self.feature_dict = global_feat_dict
        self.label_dict = label_dict