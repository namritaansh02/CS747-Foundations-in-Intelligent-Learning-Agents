import random
from cv2 import EMD
import numpy as np

class MDP():
    def __init__(self, S, A, T, R, gamma, endstates):
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.discount = gamma 
        if not endstates:
            self.mdptype = 'episodic'
        else:
            self.mdptype = 'continuing'
        self.endstates = endstates