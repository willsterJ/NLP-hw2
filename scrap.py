import numpy as np
import math

x = np.zeros((2,2))

print(x)

v = x[0]
v[0] = 5

print(x)

print(math.log(math.exp(2)))

x = .0000001
y = '%.2E' % x

print(y)


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