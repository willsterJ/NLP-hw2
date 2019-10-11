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
        self.input_list = []  # [[{local_char dict}, label]]
        self.predicted_labels = []
        self.true_labels = []
        self.input_matrix = None  # dim: input size x (feature size * num of labels)
        self.weight_matrix = None  # dim: feature_size * num_labels
        self.INPUT_DIM = 0
        self.FEATURE_DIM = 0
        self.OUTPUT_DIM = 0
        self.denominator_true_sum = 0  # normalizer to be used in objective function
        self.denominator_predicted_sum = 0  # used in gradient

        self.lamb = config.Config.lamb


    def find_features_and_labels(self):
        pass

    def get_column_to_feature_index(self, in_index, label):
        # DEPRECATED
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
        # DEPRECATED

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
        for i in range(0, row_dim-1):
            feature_vec = self.input_matrix[i]
            weight_vec = self.weight_matrix[self.get_true_label_index(i)]
            d_sum += self.compute_score(feature_vec, weight_vec)

        self.denominator_true_sum = d_sum

    def objective_function(self, weights_mat, x_mat):
        w_mat = np.copy(weights_mat)
        x_mat = np.copy(x_mat)

        if not w_mat.shape[1] == x_mat.shape[1]:
            print('weights and x matrix dimensions do not match')
            exit(1)

        obj_func = 0
        for i in range(0, x_mat.shape[0]):  # for each row of input matrix
            feature_vec = x_mat[i]
            label_index = self.get_true_label_index(i)
            weight_vec = w_mat[label_index]

            numerator = self.compute_score(feature_vec, weight_vec)
            denominator = self.denominator_true_sum
            obj_func += math.log(numerator / denominator)

        return obj_func

    def regularization(self, obj_func, weights_mat, lamb):
        norm = np.linalg.norm(weights_mat)

        return obj_func - (lamb * (norm ** 2))

    def get_true_label_index(self, input_index):
        """
        Get the true label from self.input_list given input data index
        :param input_index:
        :return: label_index
        """
        true_label = self.input_list[input_index][1]
        label_index = self.label_dict[true_label]
        return label_index

    def get_predicted_label(self, x_vec):
        """
        The predicted class is the feature class vector with the highest score
        :param x_vec:
        :return: max_key
        """
        max = -(sys.maxsize - 1)
        max_key = None

        for key, value in self.label_dict.items():  # for each class feature vector
            score = self.compute_score(x_vec, self.weight_matrix)
            if score > max:
                max = score
                max_key = key

        return max_key

    def maximum_entropy(self, weight_vec, feature_vec):
        w_vec = np.copy(weight_vec)
        x_vec = np.copy(feature_vec)

        return self.compute_score(w_vec, x_vec) / self.denominator_predicted_sum

    def maximum_entropy_denominator_sum(self):
        row_dim, col_dim = self.input_matrix.shape

        d_sum = 0
        for i in range(0, row_dim-1):
            feature_vec = self.input_matrix[i]
            weight_vec = self.weight_matrix[self.label_dict[self.predicted_labels[i]]]
            d_sum += self.compute_score(feature_vec, weight_vec)

        self.denominator_predicted_sum = d_sum

    def compute_gradient(self):
        partial_gradients = np.zeros((self.OUTPUT_DIM, self.FEATURE_DIM))

        left_sum = np.zeros((1, self.FEATURE_DIM))
        for i in range(0, self.OUTPUT_DIM):  # for each partial gradient (i.e. weight vector)
            for j in range(0, self.INPUT_DIM):  # for each input
                feature_vec = self.input_matrix[j]
                true_label = self.true_labels[j]
                true_label_index = self.label_dict[true_label]

                if true_label_index == i:  # if derivative w.r.t current class is the same as true class of input
                    left_sum = np.add(left_sum, feature_vec)

            partial_gradients[i] = left_sum

        right_sum = np.zeros((1, self.FEATURE_DIM))
        for i in range(0, self.INPUT_DIM):
            predicted_label = self.predicted_labels[i]
            predicted_label_index = self.label_dict[predicted_label]

            feature_vec = self.input_matrix[i]
            weight_vec = self.weight_matrix[predicted_label_index]
            right_sum = np.add(right_sum, self.maximum_entropy(weight_vec, feature_vec) * feature_vec)

        # TODO finish the gradient calculation

        partial_gradients = left_sum - right_sum - (2 * self.lamb          )

        return partial_gradients

    def gradient_ascent(self):
        epsilon = config.Config.epsilon
        lr = config.Config.learning_rate
        t = 0
        diff = sys.maxsize
        curr_weights = self.weight_matrix

        while t == 0 or diff > epsilon:
            t += 1
            prev_weights = curr_weights
            curr_weights = prev_weights + (lr * self.compute_gradient())
            lr = lr / math.sqrt(t)
            diff = np.linalg.norm(np.subtract(curr_weights, prev_weights))


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
            self.true_labels.append(label)

        self.feature_dict = char_dict
        self.label_dict = label_dict
        self.input_list = input_list
        return char_dict, label_dict, input_list

    def generate_input_matrix(self):
        (feature_dict, label_dict, input_list) = self.find_features_and_labels()
        self.INPUT_DIM = len(self.data)
        row_dim = self.INPUT_DIM
        self.FEATURE_DIM = len(feature_dict)  # feature vect of classes. See class notes
        column_dim = self.FEATURE_DIM
        self.OUTPUT_DIM = len(self.label_dict)

        input_matrix = np.zeros((row_dim, column_dim))  # create matrix of dim inputs x features

        print(label_dict)
        print(len(feature_dict))
        print(input_matrix.shape)

        for i, (features, label) in enumerate(input_list):  # for each input data
            # features: dict{feature: count}
            # label: string

            feature_vec = input_matrix[i]

            for key1, value1 in features.items():  # for each {feature: count} entry
                index_of_feature = self.feature_dict[key1]  # get associated column index from global feature
                feature_vec[index_of_feature] = value1

                '''
                # DEPRECATED. Use this for 1D weight that divides weights between classes
                for key2, value2 in label_dict.items():  # {class: index}
                    # duplicate feature vectors so that each vector has valid features for each class
                    # ex: if x = [1 2 3 4 ...], then for each class label, create a feature vector from x
                    # f1 = [1 2 3 4 0 ... 0], f2 = [0 ... 0 1 2 3 4 0 ... 0], ...

                    feature_vec = input_matrix[i + value2]
                    column_index = self.get_feature_to_column_index(index_of_feature, key2)

                    feature_vec[column_index] = value1
                '''

        self.input_matrix = input_matrix
        self.weight_matrix = np.random.randint(low=0, high=5, size=(self.OUTPUT_DIM, self.FEATURE_DIM))

        return input_matrix







data = load_data.LoadData()
data = data.read_file(config.Config.train_data)
model = Unigram(data)
print(model.generate_input_matrix())

