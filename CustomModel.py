from model import Model
import re
import numpy as np
from Data_Point import Data_Point


class CustomModel(Model):
    def __init__(self, data, valid_set, test_set):
        super().__init__(data, valid_set, test_set)
        self.suffix_l = ["ac", "an", "alt", "ate", "al", "ax", "at", "ar", "air", "aine", "am", "cin",
                        "ec", "erm", "ex", "en", "el", "ene", "hyd", "ium", "ide", "is", "il", "ine", "it", "ir", "ix",
                        "ist", "in", "lin", "mic", "phen", "ound", "om", "ox", "ot", "ole", "ol", "ort", "oda", "ra",
                        "tic", "yl", "ym",

                        "llc", "l.l.c", "corp", "corp.", "inc", "inc.", "ltd", "ltd.", "lp", "l.p", "co", "co."]

        self.word_l = ["symptom", "cough", "allergy", "gel", "liquid", "children", "maximum", "fungal", "powder",
                           "strength", "cold", "flu", "ointment", "nasal",

                           "by", "or", "in", "an", "a", "for", "and", "I", "as", "to", "of", "at", "la", "le", "las", "the", "with", "me",
                           "she", "he", "between", "vs.", "de", "her", "him", "love", "der", "von", "el", "tu", "en", "film",
                           "movie", "from", "into", "going", "you",

                           "government", "managed", "fund", "funding", "trust", "holding",  "shareholder", "income",
                            "corporation", "incorporated", "securities", "companies", "services", "marketing",
                           "group", "international", "technologies", "capital", "equities", "power", "financial", "limited",
                           "bank", "banco", "laboratories"]
        self.special_l = ["-", "'s", ":", "!", "&"]

        self.find_features_and_labels()
        self.find_validation_features()

    def find_features_and_labels(self):
        data = self.data

        global_feat_dict = {}
        global_feat_index = 0
        label_dict = {}
        label_index = 0

        for item in data:
            label = item[0]
            input = str(item[1])

            local_feat_dict = {}

            discovered_features = []

            discovered_features.extend(self.features_suffixes(input))
            discovered_features.extend(self.features_words(input))
            discovered_features.extend(self.features_special(input))

            for f in discovered_features:
                if f not in local_feat_dict:
                    local_feat_dict[f] = 1
                else:
                    local_feat_dict[f] += 1

                if f not in global_feat_dict:
                    global_feat_dict[f] = global_feat_index
                    global_feat_index += 1

            if label not in label_dict:
                label_dict[label] = label_index
                label_index += 1

            data_point = Data_Point()
            data_point.true_label_index = label_dict[label]
            data_point.features_dict = local_feat_dict
            self.data_points_list.append(data_point)

        self.feature_dict = global_feat_dict
        self.label_dict = label_dict

    def features_suffixes(self, input):
        output_list = []
        input_l = input.split(" ")

        for word in input_l:
            for suffix in self.suffix_l:
                reg = ".*" + suffix + "$"
                z = re.search(reg, word)
                if z:
                    output_list.append(suffix)
        return output_list

    def features_words(self, input):
        output_l = []
        input_l = input.split(" ")
        for word in input_l:
            for w in self.word_l:
                if w in word:
                    output_l.append(w)
        return output_l

    def features_special(self, input):
        output_l = []
        for spec in self.special_l:
            if spec in input:
                output_l.append(spec)
        return output_l

    def find_validation_features(self):
        data = self.valid_set

        for item in data:
            label = item[0]
            input = str(item[1])

            local_feat_dict = {}

            discovered_features = []

            discovered_features.extend(self.features_suffixes(input))
            discovered_features.extend(self.features_words(input))
            discovered_features.extend(self.features_special(input))

            for f in discovered_features:
                if f in self.feature_dict:  # if feature in global feat dict
                    if f not in local_feat_dict:
                        local_feat_dict[f] = 1
                    else:
                        local_feat_dict[f] += 1

            data_point = Data_Point()
            data_point.true_label_index = self.label_dict[label]
            data_point.features_dict = local_feat_dict
            self.valid_data_points_list.append(data_point)


