import config
import numpy as np
import math
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Data_Point import Data_Point
import time
import threading


# parent class
class Model:
    def __init__(self, data, valid_set, test_set):
        self.data = data  # numpy matrix [[label, name]]
        self.valid_set = valid_set
        self.test_set = test_set
        self.feature_dict = {}  # feature_name: feature_vec index
        self.label_dict = {}  # true_output: label_index

        self.data_points_list = []  # list of Data_Point
        self.valid_data_points_list = []
        self.test_data_points_list = []

        self.input_matrix = None  # dim: input size x (feature size * num of labels)
        self.weight_matrix = None  # dim: feature_size * num_labels
        self.valid_matrix = None
        self.INPUT_DIM = 0
        self.FEATURE_DIM = 0
        self.OUTPUT_DIM = 0

        self.lamb = config.Config.lamb  # user-specified parameter

        # to be used for plotting objective function
        self.plot_data_x = []
        self.plot_data_y = []

    def find_features_and_labels(self):  # subclass implemented function
        pass

    def combine_features_from_models(self, model1, model2):
        """
        method that takes in 2 models and combines them into 1
        """
        # get data point's local features
        data_point_list1 = model1.data_points_list
        data_point_list2 = model2.data_points_list

        # combine data point attributes
        for i in range(0, len(data_point_list1)):
            data_point = Data_Point()
            local_features_1 = data_point_list1[i].features_dict
            local_features_2 = data_point_list2[i].features_dict
            for key1, val1 in local_features_1.items():
                data_point.features_dict[key1] = val1
            for key2, val2 in local_features_2.items():
                if key2 not in data_point.features_dict:
                    data_point.features_dict[key2] = val2

            data_point.true_label_index = data_point_list1[i].true_label_index
            data_point.index = i
            self.data_points_list.append(data_point)

        # combine feature to col map index
        self.feature_dict = model1.feature_dict
        col_size = len(model1.feature_dict)

        add_col_ind = 0
        for key2, val2 in model2.feature_dict.items():
            if key2 not in self.feature_dict:
                self.feature_dict[key2] = col_size + add_col_ind
                add_col_ind += 1

        # label should be the same
        self.label_dict = model1.label_dict


    def generate_input_matrix(self):
        """
        Creates the design matrix using the feature and label data generated from subclass. Iterate over list of
        data_points and update their values. Also create a random weight matrix
        """
        self.input_matrix = self.create_matrix(self.data_points_list)

        np.random.seed(0)
        self.weight_matrix = np.zeros((self.OUTPUT_DIM, self.FEATURE_DIM), dtype=float)

        self.valid_matrix = self.create_matrix(self.valid_data_points_list)

    def create_matrix(self, data_point_list,):
        self.INPUT_DIM = len(data_point_list)
        row_dim = self.INPUT_DIM
        self.FEATURE_DIM = len(self.feature_dict)  # feature vect of classes. See class notes
        column_dim = self.FEATURE_DIM
        self.OUTPUT_DIM = len(self.label_dict)

        input_matrix = np.zeros((row_dim, column_dim), dtype=np.double)  # create matrix of dim inputs x features

        print(self.label_dict)
        print(input_matrix.shape)

        for i, data_point in enumerate(data_point_list):  # for each input data
            data_point.index = i
            data_point.features_vec = input_matrix[i]
            features = data_point.features_dict

            for key1, value1 in features.items():  # for each {feature: count} entry
                index_of_feature = self.feature_dict[key1]  # get associated column index from global feature
                data_point.features_vec[index_of_feature] = value1

        return input_matrix

    def compute_score(self, w_vec, x_vec):
        """
        computes the dot product between a weight vec and an input vec
        """
        return np.dot(w_vec.transpose(), x_vec).astype(np.double)

    def maximum_entropy(self, input_index, partial_index):
        """
        Find the maximum entropy of a input data given its index
        """
        weight_vec = self.weight_matrix[partial_index]  # weight associated with predicted label
        feature_vec = self.input_matrix[input_index]

        # TODO debug overflow
        numerator = math.exp(self.compute_score(weight_vec, feature_vec))
        # numerator = np.exp(self.compute_score(weight_vec, feature_vec), dtype=np.float64)

        denominator = 0
        for i in range(0, self.OUTPUT_DIM):  # normalize over all weights for that input feature vector
            w_vec = self.weight_matrix[i]
            score = self.compute_score(w_vec, feature_vec)
            # val = np.exp(score, dtype=np.float64)
            val = math.exp(score)
            denominator += val

        try:
            quotient = numerator / float(denominator)
        except ValueError:
            print('input_i: %d , denom: %d' % (input_index, denominator))
        except RuntimeWarning:
            exit(1)

        return quotient

    def compute_gradient(self):
        """
        Compute the gradient. Stores all partial gradients in a matrix, with each row corresponding to a class weight's
        partial derivative
        :return:
        """
        partial_gradients = np.zeros((self.OUTPUT_DIM, self.FEATURE_DIM))

        left_sum = np.zeros((1, self.FEATURE_DIM))
        right_sum = np.zeros((1, self.FEATURE_DIM))

        # set up multi-threading
        num_threads = config.Config.num_threads
        threads = [None] * num_threads
        left_result = {}  # shared dict for all threads
        right_result = {}

        # compute total feature count over examples with true y class
        for partial_index in range(0, self.OUTPUT_DIM):  # for each partial gradient (i.e. weight vector)
            # reset accumulators
            for i in range(num_threads):
                left_result[i] = np.zeros((1, self.FEATURE_DIM))
                right_result[i] = np.zeros((1, self.FEATURE_DIM))
            # initialize threads
            for i in range(num_threads):
                # set up start and ending indices for each thread's access to input matrix
                [start_ind, end_ind] = self.threading_find_start_end_index(i, len(self.input_matrix), num_threads)
                threads[i] = threading.Thread(target=self.compute_gradient_threading,
                                              args=(partial_index, start_ind, end_ind, left_result, right_result),
                                              name=str(i))
                threads[i].start()
                threads[i].join()
            # add results from threads
            for i in range(num_threads):
                right_sum = np.add(right_sum, right_result[i])
                left_sum = np.add(left_sum, left_result[i])

            partial_gradients[partial_index] = left_sum - right_sum - (2 * self.lamb * self.weight_matrix[partial_index])

        partial_gradients = partial_gradients / self.INPUT_DIM

        return partial_gradients

    def compute_gradient_threading(self, partial_index, start_ind, end_ind, left_result, right_result):
        thread_id = int(threading.current_thread().name)
        right_acc = right_result[thread_id]  # get thread_id
        left_acc = left_result[thread_id]

        # compute expectation quantity from what the model predicts (using predicted labels)
        for input_index in range(start_ind, end_ind):
            feature_vec = self.input_matrix[input_index]
            max_ent = self.maximum_entropy(input_index, partial_index)

            right_acc = np.add(right_acc, max_ent * feature_vec)

        for input_index in range(start_ind, end_ind):  # for each input
            feature_vec = self.input_matrix[input_index]
            true_label_index = self.data_points_list[input_index].true_label_index

            if true_label_index == partial_index:  # if derivative w.r.t current class is the same as true class of input
                left_acc = np.add(left_acc, feature_vec)

        # update result
        right_result[thread_id] = right_acc
        left_result[thread_id] = left_acc

    def gradient_ascent(self):
        """
        Gradient ascent routine
        :return:
        """
        epsilon = config.Config.epsilon
        lr_0 = config.Config.learning_rate
        lr = lr_0
        t = 0
        diff = 0

        start_time = time.time()

        while t == 0 or diff > epsilon and t <= 500:

            t += 1
            prev_weights = self.weight_matrix
            curr_weights = np.add(prev_weights, (lr * self.compute_gradient()))
            lr = lr_0 / math.sqrt(t)
            diff = np.linalg.norm(np.subtract(curr_weights, prev_weights))

            self.weight_matrix = curr_weights

            obj = self.objective_function_helper()
            #training_accuracy = self.compute_accuracy(self.data_points_list, self.input_matrix)
            validation_accuracy = self.compute_accuracy(self.valid_data_points_list, self.valid_matrix)
            # for plotting use
            # print("%d: diff=%f, obj=%f, train_acc=%f, valid_acc=%f" % (t, diff, obj, training_accuracy, validation_accuracy))
            print("%d: diff=%f, obj=%f, valid_acc=%.16f" % (t, diff, obj, validation_accuracy))

            self.plot_data_x.append(t)
            self.plot_data_y.append(obj)

        end_time = time.time()
        print('total time = ' + str(start_time - end_time))

    def compute_accuracy(self, data_points_list, matrix):
        threads = [None] * config.Config.num_threads

        for i in range(len(threads)):
            [start_ind, end_ind] = self.threading_find_start_end_index(i, len(data_points_list), len(threads))
            threads[i] = threading.Thread(target=self.compute_all_predicted_labels(data_points_list, matrix, start_ind, end_ind))
            threads[i].start()
            threads[i].join()

        count = 0
        for i, data_point in enumerate(data_points_list):
            if data_point.pred_label_index == data_point.true_label_index:
                count += 1

        return count / len(data_points_list)

    def compute_all_predicted_labels(self, data_points_list, matrix, start_ind, end_ind):
        """
        computes and updates predicted labels for each input
        :return:
        """
        for i in range(start_ind, end_ind):
            data_points_list[i].pred_label_index = self.update_predicted_label_from_index(i, matrix)

    def update_predicted_label_from_index(self, input_index, matrix):
        """
        From input data index, update its associated predicted label
        :param input_index:
        :return:
        """
        feature_vect = matrix[input_index]
        feature_vect = np.expand_dims(feature_vect, axis=1)
        output_vect = np.dot(self.weight_matrix, feature_vect)
        max_ind = np.argmax(output_vect)
        return max_ind

    def objective_function_helper(self):
        threads = [None] * config.Config.num_threads
        obj_sums = {}

        for i in range(len(threads)):
            [start_ind, end_ind] = self.threading_find_start_end_index(i, len(self.input_matrix), len(threads))
            obj_sums[i] = 0
            threads[i] = threading.Thread(target=self.objective_function,
                                          args=(start_ind, end_ind, obj_sums), name=str(i))
            threads[i].start()
            threads[i].join()

        obj_sum = 0
        for i in range(config.Config.num_threads):
            obj_sum += obj_sums[i]

        return obj_sum

    def objective_function(self, start_ind, end_ind, obj_sums):
        """
        compute the objective function of the model
        :return:
        """
        for i in range(start_ind, end_ind):  # for each row of input matrix
            feature_vec = self.input_matrix[i]
            label_index = self.data_points_list[i].true_label_index
            weight_vec = self.weight_matrix[label_index]

            score = self.compute_score(weight_vec, feature_vec)
            numerator = math.exp(score)

            denominator = 0
            for j in range(0, self.OUTPUT_DIM):
                w_vec = self.weight_matrix[j]
                denominator += math.exp(self.compute_score(w_vec, feature_vec))

            quotient = numerator / float(denominator)

            try:
                obj_sums[int(threading.current_thread().name)] += math.log(quotient)
            except ValueError:
                print("input: %d, numerator: %f, denom: %f" % (i, numerator, denominator))
                exit(1)

    def regularization(self, obj_func, weights_mat, lamb):
        norm = np.linalg.norm(weights_mat)

        return obj_func - (lamb * (norm ** 2))

    def plot_objective_function(self, name):
        fig = plt.figure()
        plt.title(name)
        plt.xlabel("t")
        plt.ylabel("L")
        plt.plot(self.plot_data_x, self.plot_data_y)
        if not os.path.exists("./output"):
            os.mkdir("./output")
        else:
            plt.savefig("./output/%s.png" % name)

    def threading_find_start_end_index(self, i, size, num_threads):
        step = size // num_threads
        remainder = size % num_threads
        start_ind = step * i
        if i == num_threads - 1:  # if last thread, give it remainder
            end_ind = start_ind + step + remainder
        else:
            end_ind = start_ind + step
        return start_ind, end_ind


