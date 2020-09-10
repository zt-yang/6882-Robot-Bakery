import matplotlib.pyplot as plt
import pddlgym
import random
import statistics
import numpy as np
import time
import math
import pandas as pd
from os.path import join
from envs.utils import draw_trace, record_trace, display_image
pd.set_option('display.max_columns', None)

from envs.robot_kitchen import RobotKitchenEnv

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

def extract_path(current_node, env=None, name=None):
    path = []
    while current_node.parent != None:
        path.append(current_node.state)
        # path.append(env.get_robot_pos(current_node.state))
        current_node = current_node.parent
    path.append(current_node.state)

    path.reverse()
    record_trace(path, env, name)
    draw_trace(path, env)
    return len(path)

def plan(env, method, output_file_name=None):

    DEBUG = False
    timeout = 10
    action_cost = 1
    start_time = time.time()
    state, _ = env.reset()
    actions = env.get_all_actions()

    if method == "random":
        trace = [env.get_robot_pos()]
        for step in range(20):
            action = random.choice(actions)
            next_state, reward, done, info = env.step(action)
            trace.append(env.get_robot_pos())
            # goal_satisfied = env.check_goal(next_state)
            if done or time.time() - start_time > timeout: break
        draw_trace(trace)
        return done, time.time() - start_time, -1, step + 1

    ## otherwise, it's a search problem
    else:

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
                # if DEBUG: print('_____',env.get_robot_pos(item.state))
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current node out off yet_to_visit list, add to visited list
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)
            if current_node.action != None:
                # state, reward, done, info = env.step(current_node.action)
                state = current_node.state
                if DEBUG: print('stepped', current_node.action, env.get_robot_pos(current_node.state))

            # if goal is reached, extract the path from current_node
            if env.check_goal(state) or time.time() - start_time > timeout:
                success_rate = 1
                if time.time() - start_time > timeout: success_rate = 0
                time_taken = time.time() - start_time
                nodes_expanded = len(visited_list)
                steps_in_env = extract_path(current_node, env=env, name=output_file_name)
                return success_rate, time_taken, nodes_expanded, steps_in_env

            # Generate children from all adjacent squares
            if DEBUG: print('children of', env.get_robot_pos(state), env.get_robot_pos(current_node.state))
            goal_objs = env._goal_objects

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
                """
                    the heuristic value is the sum of distance between chosen instance of adjacent goal objects
                    e.g., goal_config = [['D', 'm', 'D'], ['D', 'l', 'D']]
                    [BREAD3, MEAT1, BREAD1] = env._init_goal_objects()
                    node.h = env.distance(BREAD3, MEAT1, state=node.state) + env.distance(MEAT1, BREAD1, state=node.state)
                    node.f = node.g + node.h
                """
                child.h = 0
                if method != "A*Uniform":
                    for index in range(len(goal_objs) - 1):
                        obj1 = goal_objs[index]
                        obj2 = goal_objs[index + 1]
                        child.h += env.get_distance(obj1, obj2, state=child.state)

                child.f = child.g + child.h
                if DEBUG: print('     ', env.get_robot_pos(child.state))

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)

methods = [ "A*Custom", "GBFCustom"] # , "random", "A*Uniform",

table = pd.DataFrame(index=methods,
    columns=['success_rate', 'time', 'nodes_expanded', 'steps_in_env'])

env = RobotKitchenEnv()
num_problems = 2  # in total three layouts: simple 4 by 4, default 5 by 5, difficult 6 by 7
for method in methods:

    print(method, '---------------')
    data = np.zeros((num_problems, 4))

    for problem in range(num_problems):

        ## initiate the environment
        env.fix_problem_index(problem)
        state, _ = env.reset()
        name = "P"+str(problem)+", "+str(method)
        display_image(env.render_from_state(state), name)
        data[problem] = plan(env, method, output_file_name=name)  ## [1,1,1,1]
        plt.savefig(join('tests',name+'.png'))
        print('     Problem', problem, data[problem])

    print()
    table.loc[method] = [get_stats(data[:,0]),
                         get_stats(data[:,1]),
                         get_stats(data[:,2]),
                         get_stats(data[:,3])]

print('-----------------------------------')
print(table)