"""
loads the .txt data file and stores it into a matrix
"""
import config
import os
import re
import numpy as np


class IOModule:

    def read_file(self, path):
        """
        reads input file, and converts it to a numpy matrix of [label:string, data:string]
        :return: numpy matrix [[string, string]]
        """

        if not os.path.exists(path):
            print("ERROR: " + path + " path does not exist!")
            exit(1)

        string_list = []

        # open and store lines in list
        with open(path, encoding='ISO-8859-1', mode='r') as fp:
            line = fp.readline()
            while line:
                string_list.append(line.rstrip())  # rstrip removes trailing \n
                line.strip()
                try:
                    line = fp.readline()
                except IOError:
                    pass

        # parse each line to object
        a = np.zeros((len(string_list), 2), dtype=object)

        for i, item in enumerate(string_list):
            item = string_list[i]
            parse = re.split(r'\t+', item)
            a[i][0] = parse[0]
            a[i][1] = parse[1]

        return a

    def write_output(self, data_set, dest_path):
        print('writing to ' + dest_path)
        with open(dest_path, 'w') as fp:
            for data in data_set:
                s = str(data[0]).ljust(8)
                s = s + str(data[1])
                fp.write('%s\n' % s)
        print('done.')
