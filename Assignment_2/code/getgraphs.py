import matplotlib.pyplot as plt
import cv2, argparse
import numpy as np
parser = argparse.ArgumentParser()

from mdp import MDP
from encoder import create_states_dict, create_mdp
from algorithm import Algorithm, Policy

def get_values(F, mdp):
    A = np.zeros((mdp.S, mdp.S))
    b = np.zeros((mdp.S))
    for s in range(mdp.S):
        A[s] = mdp.discount*mdp.T[s][F.action[s]]
        A[s][s] -= 1
        b[s] = -1*np.dot(mdp.T[s][F.action[s]], mdp.R[s][F.action[s]])
    v = np.linalg.solve(A, b)
    return v

if __name__ == '__main__':
    parser.add_argument("--graph", type = int, required = True)
    parser.add_argument("--parameters", type = str, required = True)
    parser.add_argument("--baseline", type = str, required = True)
    args = parser.parse_args()

    states = {}
    balls = None
    runs = None

    if args.graph == 1:
        balls = 15
        runs = 30
    elif args.graph == 2:
        balls = 10
        runs = 20
    elif args.graph == 3:
        balls = 15
        runs = 10
    
    actions = [0, 1, 2, 4, 6]
    action_dict = {'0':0, '1':1, '2':2, '4':3, '6':4}
    results = ['out', 0, 1, 2, 3, 4, 6]

    pA = {}
    pB = {}

    states = create_states_dict(balls, runs)

    with open(args.parameters, 'r') as file:
        lines = file.readlines()
        for a, action in enumerate(actions):
            for r, result in enumerate(results):
                pA[action, result] = float(lines[a+1].split()[r+1])
                pB[action, result] = 0.0

    F_r = Policy(len(states))
    with open(args.baseline, 'r') as f:
        lines = f.readlines()
        for s, line in enumerate(lines):
            F_r.action[s] = action_dict[line.split()[1]]
        for s in range(len(lines), len(states)):
            F_r.action[s] = np.random.randint(0, 5)
        
    y1 = []
    y2 = []
    x = []
    if args.graph == 1:
        xpoints = 50
        for q in range(xpoints+1):
            q = q*(1.0/xpoints)
            for a, action in enumerate(actions):
                pB[action, 'out'] = q 
                pB[action, 0]     = (1-q)*0.5
                pB[action, 1]     = (1-q)*0.5
            
            mdp = create_mdp(states, balls, runs, pA, pB)
            algo = Algorithm('vi')
            F = algo.evaluate(mdp)

            F_r.value = get_values(F_r, mdp)

            y2.append(F_r.value[0])
            y1.append(F.value[0])
            x.append(q)

        plt.plot(x, y1, label = 'Optimal Policy')
        plt.plot(x, y2, label = 'Random Policy')
        plt.title('Graph 1')
        plt.ylabel('Win Probability at 30 runs from 15 balls')
        plt.xlabel('Out Probability of Player B')
        plt.legend()
        plt.savefig('graphs/Graph1.png')
        plt.show()
    elif args.graph == 2:
        q = 0.25
        for a, action in enumerate(actions):
            pB[action, 'out'] = q 
            pB[action, 0]     = (1-q)*0.5
            pB[action, 1]     = (1-q)*0.5

        mdp = create_mdp(states, balls, runs, pA, pB)
        algo = Algorithm('vi')
        F = algo.evaluate(mdp)

        F_r.value = get_values(F_r, mdp)
        
        for i in range(runs):
            y2.append(F_r.value[i])
            y1.append(F.value[i])
            x.append(runs-i)
        print(x)
        plt.plot(x, y1, label = 'Optimal Policy')
        plt.plot(x, y2, label = 'Random Policy')
        plt.title('Graph 2')
        plt.ylabel('Win Probability with 10 balls remaining')
        plt.xlabel('Runs to Win')
        plt.legend()
        plt.savefig('graphs/Graph2.png')
        plt.show()
    elif args.graph == 3:
        q = 0.25
        for a, action in enumerate(actions):
            pB[action, 'out'] = q 
            pB[action, 0]     = (1-q)*0.5
            pB[action, 1]     = (1-q)*0.5

        mdp = create_mdp(states, balls, runs, pA, pB)
        algo = Algorithm('vi')
        F = algo.evaluate(mdp)

        F_r.value = get_values(F_r, mdp)
 
        for i in range(balls):
            y2.append(F_r.value[i*runs])
            y1.append(F.value[i*runs])
            x.append(balls-i)

        plt.plot(x, y1, label = 'Optimal Policy')
        plt.plot(x, y2, label = 'Random Policy')
        plt.title('Graph 3')
        plt.ylabel('Win Probability with 10 runs to score')
        plt.xlabel('Balls remaining')
        plt.legend()
        plt.savefig('graphs/Graph3.png')
        plt.show()
    
    




