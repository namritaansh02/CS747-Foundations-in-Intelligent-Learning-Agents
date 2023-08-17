import random,argparse,sys
from cv2 import EMD
parser = argparse.ArgumentParser()
import numpy as np

import time
start_time = time.time()
from algorithm import Algorithm
from mdp import MDP
from algorithm import Policy

if __name__ == "__main__":
    parser.add_argument("--mdp", type = str, required = True)
    parser.add_argument("--algorithm", type = str, default = 'vi')
    parser.add_argument("--policy", type = str, required = False)
    
    problem = None
    args = parser.parse_args()
    with open(args.mdp, encoding='utf-8') as file:
        lines = file.readlines()
    
        S = int(lines[0].split()[1]) 
        A = int(lines[1].split()[1]) 
        
        end_states = list(map(int, lines[2].split()[1:]))
        if end_states[0]==-1:
            end_states = []
        
        discount = float(lines[-1].split()[1])
        mdptype = lines[-2].split()[1]

        T = np.zeros((S, A, S))
        R = np.zeros((S, A, S))

        for line in lines[3:-2]:
            line = line.split()
            R[int(line[1])][int(line[2])][int(line[3])] = float(line[4])
            T[int(line[1])][int(line[2])][int(line[3])] = float(line[5])

        problem = MDP(S, A, T, R, discount, end_states)

    if not args.policy==None:
        with open(args.policy, encoding = 'utf-8') as file:
            lines = file.readlines()
            F = Policy(len(lines))
            for s in range(len(lines)):
                F.action[s] = int(lines[s][0])

        algo = Algorithm(args.algorithm, True)
        algo.evaluate(problem, F)
    
    else:
        algo = Algorithm(args.algorithm)
        algo.evaluate(problem)