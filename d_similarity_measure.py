# Author: Xiuxia Du
# January 2021

import numpy as np
from numpy import linalg as LA

class simimarity_measure:

    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def get_euclidean(self):
        d = LA.norm(self.x1 - self.x2, ord=2)
        return d

    def get_cosine(self):
        d = np.dot(self.x1, self.x2) / (LA.norm(self.x1, ord=2) * LA.norm(self.x2, ord=2))

        return d

