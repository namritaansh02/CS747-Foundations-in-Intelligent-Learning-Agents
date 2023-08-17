import random,argparse,sys
parser = argparse.ArgumentParser()
import numpy as np


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

if __name__ == "__main__":
    parser.add_argument("--value-policy", type = str, required = True)
    parser.add_argument("--states", type = str, required = True)
    
    problem = None
    args = parser.parse_args()
    states = []

    with open(args.states) as f:
        lines = f.readlines()
        for s, line in enumerate(lines):
            states.append(line[:-1])

    with open(args.value_policy, encoding='utf-8') as file:
        lines = file.readlines()
        S = len(states)
        action = [0, 1, 2, 4, 6]

        for i, line in enumerate(lines):
            if i>=S:
                break
            print(states[i], action[int(line.split()[1])], line.split()[0])