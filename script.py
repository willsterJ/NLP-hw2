from Unigram import Unigram
from BiTrigram import BiTrigram
from CustomModel import CustomModel
from load_data import LoadData
import config
from Data_Point import Data_Point
import matplotlib.pyplot as plt

data = LoadData()
data = data.read_file(config.Config.train_data)
model = Unigram(data)
# model = BiTrigram(data)
# model = CustomModel(data)
model.find_features_and_labels()
input_matrix = model.generate_input_matrix()

model.gradient_ascent()

model.compute_accuracy()

#model.plot_objective_function()
