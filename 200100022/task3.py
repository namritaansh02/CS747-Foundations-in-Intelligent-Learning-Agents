"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        self.time = 0
        self.emprical_mean = np.zeros(num_arms, float)
        self.pulls = np.zeros(num_arms)
        self.epsilon = 0.0003

        for i in range(1, num_arms):
            self.emprical_mean[i] += self.emprical_mean[i-1] + (1/num_arms)

    def give_pull(self):
        if np.random.random()<self.epsilon:
            arm_index = np.random.randint(self.num_arms)
        else:
            arm_index = np.argmax(self.emprical_mean)

        self.pulls[arm_index] += 1
        return arm_index
    
    def get_reward(self, arm_index, reward):
        n = self.pulls[arm_index]
        mean = self.emprical_mean[arm_index]
        self.emprical_mean[arm_index] = mean*((n-1)/n) + reward*(1/n)
