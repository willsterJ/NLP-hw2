from Unigram import Unigram
from load_data import LoadData
import config
from Data_Point import Data_Point

data = LoadData()
data = data.read_file(config.Config.train_data)
model = Unigram(data)
model.find_features_and_labels()
input_matrix = model.generate_input_matrix()

objective_function = model.objective_function()
print(objective_function)

test_point = model.data_points_list[0]
print(test_point.true_label)

print(model.get_predicted_label_from_index(0))

model.get_all_predicted_labels()
