import config
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
        #self.weight_matrix = np.random.randint(low=0, high=5, size=(self.OUTPUT_DIM, self.FEATURE_DIM)).astype("double")
        # self.weight_matrix = np.random.rand(self.OUTPUT_DIM, self.FEATURE_DIM)
        self.weight_matrix = np.zeros((self.OUTPUT_DIM, self.FEATURE_DIM), dtype=float)
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
        # compute total feature count over examples with true y class
        for partial_index in range(0, self.OUTPUT_DIM):  # for each partial gradient (i.e. weight vector)
            # compute expectation quantity from what the model predicts (using predicted labels)
            for input_index, data_point in enumerate(self.data_points_list):
                feature_vec = self.input_matrix[input_index]
                max_ent = self.maximum_entropy(input_index, partial_index)

                right_sum = np.add(right_sum, max_ent * feature_vec)

            for input_index, data_point in enumerate(self.data_points_list):  # for each input
                feature_vec = self.input_matrix[input_index]
                true_label_index = self.label_dict[data_point.true_label]

                if true_label_index == partial_index:  # if derivative w.r.t current class is the same as true class of input
                    left_sum = np.add(left_sum, feature_vec)

            #right_sum = (1/self.INPUT_DIM) * right_sum
            #left_sum = (1 / self.INPUT_DIM) * left_sum
            partial_gradients[partial_index] = left_sum - right_sum - (2 * self.lamb * self.weight_matrix[partial_index])
            #partial_gradients = (1/self.INPUT_DIM) * partial_gradients

        partial_gradients = partial_gradients / self.INPUT_DIM

        #TODO debug
        for i in range(0, self.OUTPUT_DIM):
            #print(partial_gradients[i])
            max = np.max(np.array(partial_gradients[i]))
            min = np.min(np.array(partial_gradients[i]))
            print("\t w_%d: max=%f, min=%f" % (i, max, min))

        return partial_gradients

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

        while t == 0 or diff > epsilon:
            # for plotting use
            print("%d: %s" % (t, diff))
            #self.plot_data_x.append(t)
            #self.plot_data_y.append(obj)

            t += 1
            prev_weights = self.weight_matrix
            curr_weights = np.add(prev_weights, (lr * self.compute_gradient()))
            lr = lr_0 #/ (self.INPUT_DIM) # * math.sqrt(t))
            diff = np.linalg.norm(np.subtract(curr_weights, prev_weights))
            #print("%d: %s" % (t, diff))

            self.weight_matrix = curr_weights

            self.compute_accuracy()
            print("obj: %f" % self.objective_function())

    def compute_accuracy(self):
        self.compute_all_predicted_labels()  # update all predicted labels

        '''
        for i, data_point in enumerate(self.data_points_list):
            print("true: %s, pred: %s" % (data_point.true_label, data_point.pred_label))
        '''

        count = 0
        for i, data_point in enumerate(self.data_points_list):
            if data_point.pred_label == data_point.true_label:
                count += 1

        print("Accuracy: %f" %(count / len(self.data_points_list)))

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
        max_val = -(sys.float_info.max - 1)
        max_key = None  # label name

        # find max score
        for key, value in self.label_dict.items():  # for each {class name: class index}
            score = self.compute_score(self.weight_matrix[value], x_vec)  # each class weight vector
            if score > max_val:
                max_val = score
                max_key = key

        return max_key

    def objective_function(self):
        """
        compute the objective function of the model
        :return:
        """
        obj_func = 0
        for i, data_point in enumerate(self.data_points_list):  # for each row of input matrix

            if i == 101:
                print("")

            feature_vec = self.input_matrix[i]
            label_index = self.label_dict[data_point.true_label]
            weight_vec = self.weight_matrix[label_index]

            score = self.compute_score(weight_vec, feature_vec)
            numerator = math.exp(score)

            denominator = 0
            for j in range(0, self.OUTPUT_DIM):
                w_vec = self.weight_matrix[j]
                denominator += math.exp(self.compute_score(w_vec, feature_vec))

            quotient = numerator / float(denominator)

            try:
                obj_func += math.log(quotient)
            except ValueError:
                print("input: %d, numerator: %f, denom: %f" % (i, numerator, denominator))
                exit(1)

        return float(obj_func)

    def regularization(self, obj_func, weights_mat, lamb):
        norm = np.linalg.norm(weights_mat)

        return obj_func - (lamb * (norm ** 2))

    def plot_objective_function(self):
        fig = plt.figure()
        plt.title("Unigram")
        plt.xlabel("t")
        plt.ylabel("L")
        plt.plot(self.plot_data_x, self.plot_data_y)
        plt.show()



