import numpy as np
import math
import re
import threading

class Test:

    def __init__(self):
        self.num_threads = 4
        self.threads = [None] * self.num_threads
        self.D = {}

    def update_D(self, D, index, thread):
        D[index] = thread

    def exec(self):
        threads = self.threads
        D = {}
        for i in range(self.num_threads):
            threads[i] = threading.Thread(target=self.update_D, args=(D, i, i, ))
            threads[i].start()
            threads[i].join()
        print(D)

t = Test()
t.exec()