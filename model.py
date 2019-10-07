from abc import ABC, abstractmethod
from typing import Any, Union

import load_data
import config
import re
import numpy as np
import math
import sys


# parent class
class Model:
    def __init__(self, data):
        self.data = data  # numpy matrix [[label, name]]
        self.feature_dict = {}  # feature_name: local_index
        self.label_dict = {}  # true_output: index_output_vector
        self.input_matrix = np.zeros((1, 1))  # dim: input size x (feature size * num of labels)
        self.weight_vec = np.zeros((1,1))  # dim: feature_size * num_labels
        self.denominator_sum = 0  # normalizer to be used in objective function

    def find_features_and_labels(self):
        pass

    def get_column_to_feature_index(self, in_index, label):  # TODO remove label param
        """
        Since the matrix's column space is num_of_features * num_of_labels, this function gets the true index of features
        as if the matrix's column space is scaled down to only num_of_features
        """
        out_index = in_index % len(self.feature_dict)  # mod w.r.t total features gives us actual feature index

        # check to see if actual feature index agrees with out_index
        true_label_index = self.label_dict[label]
        if not true_label_index == out_index:
            print('ERROR - true_label_index is not the same as out_index')
            exit(1)

        return out_index

    def get_feature_to_column_index(self, in_index, label):
        label_index = self.label_dict[label]
        features_size = len(self.feature_dict)
        out_index = label_index * features_size + in_index
        return out_index

    def compute_score(self, x_vec, w_vec):
        x_vec = np.copy(x_vec)
        w_vec = np.copy(w_vec)

        return math.exp(np.dot(w_vec.transpose(), x_vec))

    def objective_function_denominator_sum(self):
        row_dim, col_dim = self.input_matrix.shape

        d_sum = 0
        for i in range(0, col_dim-1):
            feature_vec = self.input_matrix[i]
            d_sum += math.exp(np.dot(self.weight_vec.transpose(), feature_vec))

        self.denominator_sum = d_sum

    def objective_function(self, weights_vec, x_mat, y_vec, denominator):
        w_vec = np.copy(weights_vec)
        x_mat = np.copy(x_mat)
        y_vec = np.copy(y_vec)

        if not w_vec.shape[0] == x_mat.shape[1]:
            print('weights and x matrix dimensions do not match')
            exit(1)

        obj_func = 0
        for i in range(0, x_mat.shape[0]):  # for each row of input matrix
            feature_vec = x_mat[i]
            numerator = self.compute_score(feature_vec, w_vec)
            denominator = self.denominator_sum
            obj_func += math.log(numerator / denominator)

        return obj_func

    def regularization(self, obj_func, weights_vec, lamb):
        w_vec = np.copy(weights_vec)
        norm = math.sqrt(np.dot(w_vec.transpose(), w_vec))

        return obj_func - lamb * norm

    def get_predicted_class(self):
        y_pred = np.zeros((1, len(self.label_dict)))

        max = -(sys.maxsize - 1)
        max_index = 0

        for

    '''
    def gradient(self, x_vec, w_vec, lamb):

        RIGHT = 0
        for i in range(0, self.input_matrix.shape[0]):
    '''



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

        input_count = 0
        input_list = []  # list of dict of chars stores the char features per input and corresponding output

        # parse each character into a set
        for item in data:
            input = str(item[1])
            input = re.sub(r'[^\w]', '', input)
            input_l = list(input)
            label = str(item[0])

            input_char_dict = {}

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

            input_list.append((input_char_dict, label))

        self.feature_dict = char_dict
        self.label_dict = label_dict
        return char_dict, label_dict, input_list

    def generate_input_matrix(self):
        (feature_dict, label_dict, input_list) = self.find_features_and_labels()
        num_of_inputs = len(self.data)
        column_dim = len(feature_dict) * len(label_dict)  # feature vect of classes. See class notes
        output_dim = len(label_dict)

        input_matrix = np.zeros((num_of_inputs, column_dim))  # create matrix of dim inputs x features

        print(label_dict)
        print(len(feature_dict))
        print(input_matrix.shape)

        for i, (features, label) in enumerate(input_list):
            # features: dict{feature: count}
            # label: string

            label_index = label_dict[label]  # extract output index

            feature_vec = input_matrix[i]

            for key, value in features.items():  # for each {feature: count} entry
                index_of_feature = self.feature_dict[key]  # get associated index of global feature
                column_index = self.get_feature_to_column_index(index_of_feature, label)

                # update feature vector of each input
                feature_vec[column_index] = value

        self.input_matrix = input_matrix
        self.weight_vec = np.random.randint(low=0, high=10, size=column_dim)  # create weight vec of size column dim

        return input_matrix







data = load_data.LoadData()
data = data.read_file(config.Config.train_data)
model = Unigram(data)
model.generate_input_matrix()
