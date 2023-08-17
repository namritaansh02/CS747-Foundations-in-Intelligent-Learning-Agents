import random
import numpy as np
import pulp as pulp

import time 
np.random.seed(42)

class Policy():
    def __init__(self, S):
        self.S = S
        self.value = np.zeros((S))
        self.action = np.zeros((S), int)

class Algorithm():
    def __init__(self, algo_str, eval_policy = False):
        self.evaluate = None
        if eval_policy:
            self.evaluate = self.evaluate_policy
        elif algo_str == 'vi':
            self.evaluate = self.evaluate_vi
        elif algo_str == 'lp':
            self.evaluate = self.evaluate_lp
        elif algo_str == 'hpi' or algo_str == 'default':
            self.evaluate = self.evaluate_hpi

    def print_F(self, F):
        for i in range(F.value.shape[0]):
            print(str(F.value[i])+'000000', F.action[i])

    def evaluate_policy(self, mdp, F):
        F.value = self.get_values(F, mdp)
        self.print_F(F)
        return F

    def bellman_optimality_operator(self, F, s, mdp):
        value_action = np.zeros((mdp.A))
        for a in range(mdp.A):
            value_action[a] = np.dot(mdp.T[s][a], mdp.R[s][a] + mdp.discount*F.value)
        v = np.max(value_action)
    
        return v
    
    def bellman_optimality_operator2(self, F, s, mdp):
        return np.max(np.diag(( mdp.T[s] * (mdp.R[s]+mdp.discount*F.value)).sum(-1)))

    def evaluate_vi(self, mdp):
        F = Policy(mdp.S)
        
        while True:
            Fv = np.copy(F.value)
            for s in range(mdp.S):
                F.value[s] = self.bellman_optimality_operator2(F, s, mdp)
            
            if np.max(abs(F.value-Fv))<1e-40:
                break

        F = self.get_action(F, mdp)
        for s in mdp.endstates:
            F.action[s] = 0

        self.print_F(F)
        return F

    def solve_LP(self, mdp):
        problem = pulp.LpProblem('MDP_Problem', pulp.LpMaximize)
        
        states = []
        for s in range(mdp.S):
            states.append('S'+"{:04d}".format(s))

        state_value = pulp.LpVariable.dicts('State_Values', states)
        
        problem += (pulp.lpSum([-state_value[s] for s in states]), 'State_Value_Sum')

        for i, s in enumerate(states):
            for a in range(mdp.A):
                problem += (pulp.lpSum([(mdp.discount*state_value[s_]+mdp.R[i][a][j])*mdp.T[i][a][j]-(1/mdp.S)*state_value[s] for j, s_ in enumerate(states)])<=0, 'C'+str(s)+'_'+str(a))

        problem.solve(pulp.PULP_CBC_CMD(msg=0))

        value = np.zeros((mdp.S))
        for i, v in enumerate(problem.variables()):
            value[i] = v.varValue
        
        return value

    def get_action(self, F, mdp):
        for s in range(mdp.S):
            state_action_value = np.zeros((mdp.A))
            for a in range(mdp.A):
                state_action_value[a] = np.dot(mdp.T[s][a], mdp.R[s][a]+mdp.discount*F.value)
            F.action[s] = np.argmax(state_action_value)

        return F

    def evaluate_lp(self, mdp):
        F = Policy(mdp.S)

        F.value = self.solve_LP(mdp)
        F = self.get_action(F, mdp)

        for s in mdp.endstates:
            F.action[s] = 0

        self.print_F(F)
        return F 

    def get_values(self, F, mdp):
        A = np.zeros((mdp.S, mdp.S))
        b = np.zeros((mdp.S))
        for s in range(mdp.S):
            A[s] = mdp.discount*mdp.T[s][F.action[s]]
            A[s][s] -= 1
            b[s] = -1*np.dot(mdp.T[s][F.action[s]], mdp.R[s][F.action[s]])
        v = np.linalg.solve(A, b)
        return v

    def bellman_operator(self, s, a, F, mdp):
        return np.dot(mdp.T[s][a], mdp.R[s][a] + mdp.discount*F.value)

    def evaluate_hpi(self, mdp):
        F = Policy(mdp.S)
        F.action = np.random.randint(low = 0, high = mdp.A, size = mdp.S)
        F.value = self.get_values(F, mdp)

        while True:
            IA = [None]*(mdp.S)
            IS = []
            for s in range(mdp.S):
                IA[s] = []
                for a in range(mdp.A):
                    Q = self.bellman_operator(s, a, F, mdp)
                    if Q>F.value[s] and not a==F.action[s]:
                        IA[s].append(a)
                if len(IA[s])>0:
                    IS.append(s)

            for s in IS:
                F.action[s] = random.choice(IA[s])

            F.value = self.get_values(F, mdp)
            if len(IS)==0:
                break
    
        for s in mdp.endstates:
            F.action[s] = 0

        self.print_F(F)
        return F


