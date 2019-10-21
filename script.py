from Unigram import Unigram
from load_data import LoadData
import config
from Data_Point import Data_Point
import matplotlib.pyplot as plt

data = LoadData()
data = data.read_file(config.Config.train_data)
model = Unigram(data)
model.find_features_and_labels()
input_matrix = model.generate_input_matrix()

test_point = model.data_points_list[0]
print(test_point.true_label)


model.gradient_ascent()


model.compute_accuracy()

model.plot_objective_function()
