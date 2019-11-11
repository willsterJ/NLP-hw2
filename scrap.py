import numpy as np
import math
import re

x = np.zeros((2, 2))

print(x)

v = x[0]
v[0] = 5

print(x)

print(math.log(math.exp(2)))

x = .0000001
y = '%.2E' % x

print(y)

x = -2.3e72 + 10
print(math.exp(x))

s = "Hexylllo therexl llc"
z = re.search(".*llc$", s)
if z:
    print(z)

dicto = {"king"}
if "king" in dicto:
    print("'s")

if "xy" in s:
    print("yes")



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


def log_of_maximum_entropy(self, input_index):
    """
    Instead of calculating maximum entropy directly, take the log of both sides. Now we get on right-hand side:
    score(weight_vec, feature_vec) - PRODUCT
    :param input_index:
    :return:
    """
    data_point = self.data_points_list[input_index]
    pred_label_index = self.label_dict[data_point.pred_label]
    weight_vec = self.weight_matrix[pred_label_index]  # weight associated with predicted label
    feature_vec = self.input_matrix[input_index]

    sum = 0
    for i in range(0, self.OUTPUT_DIM):  # for each class label weight vec
        w_vec = self.weight_matrix[i]
        # score = self.compute_score(w_vec, feature_vec)
        sum += np.exp(w_vec)
    sum = self.compute_score(sum, feature_vec)

    try:  # TODO DEBUG sum = 0
        log_sum = math.log(sum)
    except ValueError:
        print('problem at i: %d' % input_index)
        exit(1)

    return self.compute_score(weight_vec, feature_vec) - log_sum
