"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"

        self.success = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
    
    def give_pull(self):
        # START EDITING HERE

        arm_pullcount = np.zeros(self.num_arms)

        for i in range(self.batch_size):
            sampled_value = np.zeros(self.num_arms)
            for i in range(self.num_arms):
                sampled_value[i] = np.random.beta(self.success[i]+1, self.failures[i]+1)
            arm_pullcount[np.argmax(sampled_value)] += 1

        arm = []
        pulls = []
        for i in range(self.num_arms):
            if arm_pullcount[i]>0:
                arm.append(i)
                pulls.append(int(arm_pullcount[i]))

        return arm, pulls
        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE

        for arm in arm_rewards:
            rewards = arm_rewards[arm]
            for reward in rewards:
                if reward:
                    self.success[int(arm)] += 1
                else:
                    self.failures[int(arm)] += 1
        # END EDITING HERE
