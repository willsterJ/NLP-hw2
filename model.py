from abc import ABC, abstractmethod
from typing import Any, Union

import load_data
import config
import re
import numpy as np
import math
import sys
import matplotlib.pyplot as plt


# parent class
class Model:
    def __init__(self, data):
        self.data = data  # numpy matrix [[label, name]]
        self.feature_dict = {}  # feature_name: feature_vec index
        self.label_dict = {}  # true_output: label_index

        self.data_points_list = []  # list of Data_Point

        self.input_matrix = None  # dim: input size x (feature size * num of labels)
        self.weight_matrix = None  # dim: feature_size * num_labels
        self.INPUT_DIM = 0
        self.FEATURE_DIM = 0
        self.OUTPUT_DIM = 0

        self.lamb = config.Config.lamb  # user-specified parameter

        # to be used for plotting objective function
        self.plot_data_x = []
        self.plot_data_y = []

    def find_features_and_labels(self):  # subclass implemented function
        pass

    def generate_input_matrix(self):
        """
        Creates the design matrix using the feature and label data generated from subclass. Iterate over list of
        data_points and update their values. Also create a random weight matrix
        :return:
        """
        self.INPUT_DIM = len(self.data)
        row_dim = self.INPUT_DIM
        self.FEATURE_DIM = len(self.feature_dict)  # feature vect of classes. See class notes
        column_dim = self.FEATURE_DIM
        self.OUTPUT_DIM = len(self.label_dict)

        input_matrix = np.zeros((row_dim, column_dim), dtype=np.double)  # create matrix of dim inputs x features

        print(self.label_dict)
        print(input_matrix.shape)

        for i, data_point in enumerate(self.data_points_list):  # for each input data
            data_point.index = i
            data_point.features_vec = input_matrix[i]
            features = data_point.features_dict

            for key1, value1 in features.items():  # for each {feature: count} entry
                index_of_feature = self.feature_dict[key1]  # get associated column index from global feature
                data_point.features_vec[index_of_feature] = value1

        self.input_matrix = input_matrix

        np.random.seed(0)
        self.weight_matrix = np.random.randint(low=0, high=1, size=(self.OUTPUT_DIM, self.FEATURE_DIM)).astype("float")

        return input_matrix

    def compute_score(self, x_vec, w_vec):
        '''
        computes the dot product between a weight vec and an input vec
        :param x_vec:
        :param w_vec:
        :return:
        '''
        return np.dot(w_vec.transpose(), x_vec).astype(np.double)

    def compute_all_predicted_labels(self):
        """
        computes and updates predicted labels for each input
        :return:
        """
        for i, data_point in enumerate(self.data_points_list):
            data_point.pred_label = self.update_predicted_label_from_index(i)

    def update_predicted_label_from_index(self, input_index):
        """
        From input data index, update its associated predicted label
        :param input_index:
        :return:
        """
        x_vec = self.input_matrix[input_index]
        max = -(sys.maxsize - 1)
        max_key = None  # label name

        # find max score
        for key, value in self.label_dict.items():  # for each {class name: class index}
            score = self.compute_score(x_vec, self.weight_matrix[value])  # each class weight vector
            if score > max:
                max = score
                max_key = key

        return max_key

    def maximum_entropy(self, weight_vec, feature_vec, denominator):
        """
        Find the maximum entropy of a input data with its associated weight vec
        :param weight_vec:
        :param feature_vec:
        :param denominator:
        :return:
        """

        return math.exp(self.compute_score(weight_vec, feature_vec) / denominator)

    def maximum_entropy_denominator_sum(self):
        """
        Computes the denominator sum
        :return:
        """
        d_sum = 0
        for i, data_point in enumerate(self.data_points_list):
            feature_vec = self.input_matrix[i]
            weight_vec = self.weight_matrix[self.label_dict[data_point.pred_label]]
            d_sum += math.exp(self.compute_score(feature_vec, weight_vec))

        return d_sum

    def compute_gradient(self):
        """
        Compute the gradient. Stores all partial gradients in a matrix, with each row corresponding to a class weight's
        partial derivative
        :return:
        """
        self.compute_all_predicted_labels()  # update all predicted labels
        partial_gradients = np.zeros((self.OUTPUT_DIM, self.FEATURE_DIM))

        right_sum = np.zeros((1, self.FEATURE_DIM))
        pred_denominator = self.maximum_entropy_denominator_sum()
        # compute expectation quantity from what the model predicts (using predicted labels)
        for i, data_point in enumerate(self.data_points_list):
            pred_label_index = self.label_dict[data_point.pred_label]

            feature_vec = self.input_matrix[i]
            weight_vec = self.weight_matrix[pred_label_index]
            right_sum = np.add(right_sum, self.maximum_entropy(weight_vec, feature_vec, pred_denominator) * feature_vec)

        left_sum = np.zeros((1, self.FEATURE_DIM))
        # compute total feature count over examples with true y class
        for i in range(0, self.OUTPUT_DIM - 1):  # for each partial gradient (i.e. weight vector)
            for j, data_point in enumerate(self.data_points_list):  # for each input
                feature_vec = self.input_matrix[j]
                true_label_index = self.label_dict[data_point.true_label]

                if true_label_index == i:  # if derivative w.r.t current class is the same as true class of input
                    left_sum = np.add(left_sum, feature_vec)

            partial_gradients[i] = left_sum - right_sum - (2 * self.lamb * self.weight_matrix[i])

        return partial_gradients

    def gradient_ascent(self):
        """
        Gradient ascent routine
        :return:
        """
        epsilon = config.Config.epsilon
        lr = config.Config.learning_rate
        t = 0
        diff = sys.maxsize

        while t == 0 or diff > epsilon:
            print("%d: %f" % (t, diff))
            # for plotting use
            self.plot_data_x.append(t)
            self.plot_data_y.append(self.objective_function())

            t += 1
            prev_weights = self.weight_matrix
            curr_weights = prev_weights + (lr * self.compute_gradient())
            lr = lr / math.sqrt(t)
            diff = np.linalg.norm(np.subtract(curr_weights, prev_weights))

            self.weight_matrix = curr_weights

    def compute_accuracy(self):
        count = 0
        for i, data_point in enumerate(self.data_points_list):
            if data_point.pred_label == data_point.true_label:
                count += 1

        print("Accuracy: %f" %(count / len(self.data_points_list)))

    def objective_function(self):
        """
        compute the objective function of the model
        :return:
        """
        denominator = self.objective_function_denominator_sum()

        obj_func = 0
        for i, data_point in enumerate(self.data_points_list):  # for each row of input matrix

            #if i == 2279:
            #    print('hwere')

            feature_vec = self.input_matrix[i]
            label_index = self.label_dict[data_point.true_label]
            weight_vec = self.weight_matrix[label_index]

            score = self.compute_score(feature_vec, weight_vec)
            numerator = math.exp(score)

            # TODO DEBUG: numerical underflow...

            quotient = numerator / denominator
            if (quotient <= 0):
                quotient = 1e-20

            try:
                obj_func += math.log(quotient)
            except ValueError:
                print("i: %d" %i)

        return obj_func

    def objective_function_denominator_sum(self):
        d_sum = 0
        for i, data_point in enumerate(self.data_points_list):
            feature_vec = self.input_matrix[i]
            weight_vec = self.weight_matrix[self.label_dict[data_point.true_label]]
            d_sum += math.exp(self.compute_score(feature_vec, weight_vec))

        return d_sum

    def regularization(self, obj_func, weights_mat, lamb):
        norm = np.linalg.norm(weights_mat)

        return obj_func - (lamb * (norm ** 2))

    def plot_objective_function(self):
        plt.plot(self.plot_data_x, self.plot_data_y)



