"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
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

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

def KL(x, y):
    return x*math.log(x/(y+1e-9) + 1e-9) + (1-x)*math.log((1-x)/(1-y+1e-9) + 1e-9)

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)

        self.pulls = np.zeros(num_arms)
        self.ucb_values = np.zeros(num_arms)
        self.emprical_mean = np.zeros(num_arms)
        self.time = 0
    
    def give_pull(self):
        if self.time<self.num_arms:
            arm_index = self.time
        else:
            arm_index = np.argmax(self.ucb_values)
        
        self.time += 1
        self.pulls[arm_index] += 1

        return arm_index
    
    def get_reward(self, arm_index, reward):
        for arm in range(self.num_arms):
            n = self.pulls[arm]
            t = self.time

            mean = self.emprical_mean[arm]
            if arm == arm_index:
                self.emprical_mean[arm] = mean*((n-1)/n) + reward*(1/n)
            
            self.ucb_values[arm] = self.emprical_mean[arm] + math.sqrt(2*np.log(t)*(1/(n+1e-9)))

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)

        self.pulls = np.zeros(num_arms)
        self.emprical_mean = np.zeros(num_arms)
        self.klucb_values = np.zeros(num_arms)
        self.time = 0
    
    def give_pull(self):
        if self.time<self.num_arms:
            arm_index = self.time
        else:
            arm_index = np.argmax(self.klucb_values)
        
        self.time += 1
        self.pulls[arm_index] += 1
        return arm_index

    
    def get_reward(self, arm_index, reward):
        # update kl_ucb^{t}_{a} for all arms 
        for arm in range(self.num_arms):
            n = self.pulls[arm]
            t = self.time 
            mean = self.emprical_mean[arm]
            if arm == arm_index:
                self.emprical_mean[arm] = ((n-1)/n)*mean + reward*(1/n)
            c = 3

            mean = self.emprical_mean[arm]
            upbound = (math.log(t) + c*math.log(math.log(t) + 1e-9))
            
            l = self.emprical_mean[arm]
            r = 1

            while 0.05<r-l:
                mid = l + (r-l)/2
                val = n*KL(mean, mid)
                if val<=upbound:
                    l = mid 
                else:
                    r = mid 

            self.klucb_values[arm] = l 


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)

        self.success = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
    
    def give_pull(self):
        values = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            values[i] = np.random.beta(self.success[i]+1, self.failures[i]+1)

        return np.argmax(values)
    
    def get_reward(self, arm_index, reward):
        if reward:
            self.success[arm_index] += 1
        else:
            self.failures[arm_index] += 1
