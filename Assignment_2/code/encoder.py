import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np

from mdp import MDP

def create_mdpfile(mdp):
    print('numStates', mdp.S)
    print('numActions', mdp.A)
    
    endstates_str = 'end '
    for state in mdp.endstates:
        endstates_str += str(state)+' '
    
    print(endstates_str)
    
    for s in range(mdp.S):
        for a in range(mdp.A):
            for s_ in range(mdp.S):
                if mdp.T[s][a][s_]!=0:
                    print('transition', s, a, s_, mdp.R[s][a][s_], mdp.T[s][a][s_])
    
    print('mdptype episodic')
    print('discount 1')

def create_states_dict(max_balls, max_runs):
    states = {}
    
    # Actionable states for player A
    for b in range(max_balls, 0, -1):
        for r in range(max_runs, 0, -1):
            states[b, r, 0] = len(states) 

    # Actionable staes for player B
    for b in range(max_balls, 0, -1):
        for r in range(max_runs, 0, -1):
            states[b, r, 1] = len(states)  

    # Losing states under no balls remaining with both players
    for r in range(1, max_runs+1):
        states[0, r, 0] = len(states) 
        states[0, r, 1] = len(states) 

    # Losing states under player out with both players
    for b in range(max_balls+1):
        states[b, 'out', 0] = len(states) 
        states[b, 'out', 1] = len(states) 

    # Winning condition under no more runs remaining with both players
    for b in range(max_balls+1):
        for r in range(-5,1):
            states[b, r, 0] = len(states) # win when runs<=0 by player A
            states[b, r, 1] = len(states) # win when runs<=0 by player B

    return states

def create_mdp(states, max_balls, max_runs, pA, pB):
    actions = [0, 1, 2, 4, 6]
    results = ['out', 0, 1, 2, 3, 4, 6]
    players = [0, 1]

    prob = {0:pA, 1:pB}
    T = np.zeros((len(states), len(actions), len(states)))
    R = np.zeros((len(states), len(actions), len(states)))

    # Every winning transition gives reward 1
    # Every other transition give reward 0
    for s, state in enumerate(states):
        for a, action in enumerate(actions):
            for player in players:
                for balls in range(max_balls+1):
                    for runs in range(-5, 1):
                        R[s][a][states[balls, runs, player]] = 1

    # (1) Over Change
    # (2) Winning and losing conditions are endstates
    # (3) State change
    for s, state in enumerate(states):
        for a, action in enumerate(actions):
            b = state[0]
            r = state[1]
            p = state[2]
            if b>0 and not r=='out' and r>0: # detect if state is actionable
                over = (b-1)%6==0
                T[states[b, r, p]][a][states[b-1, 'out',  p            ]] = prob[p][action, 'out']
                T[states[b, r, p]][a][states[b-1,   r-0, (p+over  )%2  ]] = prob[p][action, 0]
                T[states[b, r, p]][a][states[b-1,   r-1, (p+over+1)%2  ]] = prob[p][action, 1]
                T[states[b, r, p]][a][states[b-1,   r-2, (p+over  )%2  ]] = prob[p][action, 2]
                T[states[b, r, p]][a][states[b-1,   r-3, (p+over+1)%2  ]] = prob[p][action, 3]
                T[states[b, r, p]][a][states[b-1,   r-4, (p+over  )%2  ]] = prob[p][action, 4]
                T[states[b, r, p]][a][states[b-1,   r-6, (p+over  )%2  ]] = prob[p][action, 6]
    
    endstates = []
    for player in players:
        for b in range(max_balls+1):
            endstates.append(states[b, 'out', player])
            for r in range(-5, 1):
                endstates.append(states[b, r, player])
        for r in range(1, max_runs+1):
            endstates.append(states[0, r, player])


    mdp = MDP(len(states), len(actions), T, R, 1, endstates)
    return mdp 

if __name__ == "__main__":
    parser.add_argument("--states", type = str, required = True)
    parser.add_argument("--parameters", type = str, required = True)
    parser.add_argument("--q", type = str, required = True)
    args = parser.parse_args()

    states = {}
    max_balls = None 
    max_runs = None
    
    actions = [0, 1, 2, 4, 6]
    results = ['out', 0, 1, 2, 3, 4, 6]

    pA = {}
    pB = {}

    q = float(args.q)
    
    with open(args.states, 'r') as file:
        lines = file.readlines()
        max_balls = np.max(np.array([int(line[:2]) for line in lines]))
        max_runs = np.max(np.array([int(line[2:]) for line in lines]))
        states = create_states_dict(max_balls, max_runs)

    with open(args.parameters, 'r') as file:
        lines = file.readlines()
        for a, action in enumerate(actions):
            for r, result in enumerate(results):
                pA[action, result] = float(lines[a+1].split()[r+1])
                pB[action, result] = 0.0
            pB[action, 'out'] = q 
            pB[action, 0]     = (1-q)*0.5
            pB[action, 1]     = (1-q)*0.5
        
    mdp = create_mdp(states, max_balls, max_runs, pA, pB)
    create_mdpfile(mdp)            