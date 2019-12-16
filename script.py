from Unigram import Unigram
from BiTrigram import BiTrigram
from CustomModel import CustomModel
from model import Model
from IOModule import IOModule
from config import Config
from Data_Point import Data_Point
import argparse
import os

parser = argparse.ArgumentParser(description='input args')
parser.add_argument('--model', type=str, required=True,
                  choices=['unigram', 'ngram', 'custom', 'best'], default='unigram', help='specify model to be run')
parser.add_argument('--lr', type=float, default=0.5, help='specify learning rate')
parser.add_argument('--lamb', type=float, default=0.01, help='specify lamb normalizer')
parser.add_argument('--epsilon', type=float, default=0.00005, help='specify epsilon threshold')
parser.add_argument('--t', type=int, default=500, help='specify number of epochs')

args = parser.parse_args()
print(args)

print(os.cpu_count())
Config.num_threads = os.cpu_count()

Config.epsilon = args.epsilon
Config.learning_rate = args.lr
Config.lamb = args.lamb
Config.t = args.t

data = IOModule()
data_set = data.read_file(Config.train_data)
valid_set = data.read_file(Config.validate_data)
test_set = data.read_file(Config.test_data)

if args.model == 'unigram':
    model = Unigram(data_set, valid_set, test_set)
elif args.model == 'ngram':
    model = BiTrigram(data_set, valid_set, test_set)
elif args.model == 'custom':
    model = CustomModel(data_set, valid_set, test_set)
elif args.model == 'best':
    model1 = BiTrigram(data_set, valid_set, test_set)
    model2 = CustomModel(data_set, valid_set, test_set)
    model = Model(data_set, valid_set, test_set)
    model.combine_features_from_models(model1, model2)

model.generate_input_matrix()

model.gradient_ascent()

model.plot_output(args.model)

model.compute_test_predictions()
