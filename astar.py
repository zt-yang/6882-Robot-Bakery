import matplotlib.pyplot as plt
import pddlgym
import random
import statistics
import numpy as np
import time
import math
import pandas as pd
from envs.utils import draw_trace, display_image
pd.set_option('display.max_columns', None)

class Node:
    """ A node class for A*/BFS """

    def __init__(self, parent=None, state=None, action=None):
        self.parent = parent  ## parent is parent of the current Node
        self.state = state  ## position is current state of the world
        self.action = action  ## action taken to go from parent to state
        self.g = 0  ## g is cost from start to current Node
        self.h = 0  ## h is heuristic based estimated cost for current Node to end Node
        self.f = 0  ## f is total cost of present node i.e. :  f = g + h

    def __eq__(self, other):
        if other == None: return False
        return self.state == other.state

def get_stats(sample):
    return round(statistics.mean(sample),2), round(statistics.variance(sample),2)

def get_distance(state, obj1, obj2):
    pos1 = None
    pos2 = None
    pos3 = None
    for obj, pos in state:
        if obj == obj1: pos1 = pos
        if obj == obj2: pos2 = pos
        if obj == 'hospital0': pos3 = pos
    if pos2 == None: pos2 = pos3
    return math.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))

def get_robot(node):
    for obj, pos in node.state:
        if obj == 'robot0': return (pos, node.g, round(node.h,1), round(node.f,1))

def check_carrying(state):
    for obj, pos in state:
        if obj == 'carrying' and pos != None:
            return True
    return False

def extract_path(current_node):
    path = []
    while current_node.parent != None:
        path.append(current_node.state)
        current_node = current_node.parent
    path.append(current_node.state)

    path.reverse()
    draw_trace(path)
    return len(path)

def plan(env, method):

    DEBUG = False
    timeout = 10
    action_cost = 1
    start_time = time.time()
    state, _ = env.reset()

    if method == "random":
        trace = [state]
        for step in range(250):
            action = random.choice(env.get_all_actions())
            next_state, reward, done, info = env.step(action)
            trace.append(next_state)
            # goal_satisfied = env.check_goal(next_state)
            if done or time.time() - start_time > timeout: break
        draw_trace(trace)
        return done, time.time() - start_time, -1, step + 1

    ## otherwise, it's a search problem
    else:
        carrying_person = False

        # Create start and end node with initized values for g, h and f
        start_node = Node(None, state, None)
        start_node.g = start_node.h = start_node.f = 0

        visited_list = []
        yet_to_visit_list = []
        yet_to_visit_list.append(start_node)

        # Adding a stop condition

        while len(yet_to_visit_list) > 0:

            # Get the current node
            current_node = yet_to_visit_list[0]
            current_index = 0
            for index, item in enumerate(yet_to_visit_list):
                if DEBUG: print('_____',get_robot(item))
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current node out off yet_to_visit list, add to visited list
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)
            if current_node.action != None:
                # state, reward, done, info = env.step(current_node.action)
                state = current_node.state
                if DEBUG: print('stepped', current_node.action, get_robot(current_node))

                if check_carrying(state) and not carrying_person:
                    if DEBUG: print('----------- before carrying victim')
                    for node in visited_list:
                        if DEBUG: print(get_robot_pos(node.state, real=True))
                    if DEBUG: print('----------- start carrying victim')
                    for node in yet_to_visit_list:
                        if DEBUG: print(get_robot_pos(node.state, real=True))
                        node.h = get_distance(node.state, "robot0", "hospital0")
                        node.f = node.g + node.h
                    carrying_person = True

            # if goal is reached, extract the path from current_node
            if env.check_goal(state) or time.time() - start_time > timeout:
                success_rate = 1
                if time.time() - start_time > timeout: success_rate = 0
                time_taken = time.time() - start_time
                nodes_expanded = len(visited_list)
                steps_in_env = extract_path(current_node)
                return success_rate, time_taken, nodes_expanded, steps_in_env

            # Generate children from all adjacent squares
            if DEBUG: print('children of', get_robot_pos(state, real=True), get_robot(current_node))
            for action in actions:
                next_state = env.get_successor_state(state, action)
                if next_state == state: continue
                child = Node(current_node, next_state, action)

                # Child is on the visited list (search entire visited list)
                if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                if method == "GBFCustom":
                    child.g = 0
                else:
                    child.g = current_node.g + action_cost

                ## Heuristic costs calculated here, this is using eucledian distance
                if method == "A*Uniform":
                    child.h = 0
                else:
                    if carrying_person:
                        child.h = get_distance(next_state, "robot0", "hospital0")
                    else:
                        child.h = get_distance(next_state, "robot0", "person0")

                child.f = child.g + child.h
                if DEBUG: print('     ', get_robot(child))

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)

methods = ["random", "A*Uniform", "A*Custom", "GBFCustom"]

table = pd.DataFrame(index=methods,
    columns=['success_rate', 'time', 'nodes_expanded', 'steps_in_env'])

env = pddlgym.make("SearchAndRescueLevel1-v0")
num_problems = 20 #len(env.problems)
for method in methods:

    print(method, '---------------')
    data = np.zeros((num_problems, 4))

    for problem in range(num_problems):

        ## initiate the environment
        env.fix_problem_index(problem)
        state, _ = env.reset()
        name = "P"+str(problem)+", "+str(method)
        display_image(env.render_from_state(state), name)
        data[problem] = plan(env, method)  ## [1,1,1,1]
        plt.savefig('paths/'+name+'.png')
        print('     Problem', problem, data[problem])

    print()
    table.loc[method] = [get_stats(data[:,0]),
                         get_stats(data[:,1]),
                         get_stats(data[:,2]),
                         get_stats(data[:,3])]

print('-----------------------------------')
print(table)