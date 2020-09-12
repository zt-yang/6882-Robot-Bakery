import matplotlib.pyplot as plt
import random
import statistics
import numpy as np
import time
import re
import math
import pandas as pd
import datetime
import os
from os.path import join
from envs.utils import draw_trace, record_trace, display_image
pd.set_option('display.max_columns', None)

from envs.robot_kitchen import RobotKitchenEnv, RobotKitchenEnvRelationalAction, repair_suffix

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

def extract_path(current_node, env=None, out_file=None):
    path = []
    actions = []
    while current_node.parent != None:
        path.append(current_node.state)
        actions.append(current_node.action)
        # path.append(env.get_robot_pos(current_node.state))
        current_node = current_node.parent
    path.append(current_node.state)
    path.reverse()
    actions.reverse()

    # for s, a in zip(path, actions):
    #     print(s, a)
    # print(path[-1])

    env.record_from_trace(path, out_file)
    # draw_trace(path, env)
    return len(path)

def plan(env, method, out_file=None, max_steps=40, timeout=10, heuristic_scale=10):

    DEBUG = False
    action_cost = 1
    start_time = time.time()
    state, _ = env.reset()
    actions = env.get_all_actions()

    if method == "random":
        trace = [env.get_robot_pos()]
        for step in range(num_steps):
            action = random.choice(actions)
            next_state, reward, done, info = env.step(action)
            trace.append(env.get_robot_pos())
            # goal_satisfied = env.check_goal(next_state)
            if done or time.time() - start_time > timeout: break
        draw_trace(trace)
        return done, round(time.time() - start_time, 3), -1, step + 1

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
                time_taken = round(time.time() - start_time, 3)
                nodes_expanded = len(visited_list)
                steps_in_env = extract_path(current_node, env=env, out_file=out_file)

                return success_rate, time_taken, nodes_expanded, steps_in_env

            # Generate children from all adjacent squares
            if DEBUG: print('children of', env.get_robot_pos(state), env.get_robot_pos(current_node.state))
            goal_objs = env._goal_objects

            for action in actions:
                next_state = env.get_successor_state(state, action)
                if next_state == state: continue
                child = Node(current_node, next_state, action)

                # Child is on the visited list (search entire visited list)
                if child in visited_list:
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
                        child.h += abs(env.get_distance(obj1, obj2, state=child.state)-1)
                    child.h *= heuristic_scale

                child.f = child.g + child.h
                if DEBUG: print('     ', env.get_robot_pos(child.state))

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)

def log(to_print, TXT):
    if isinstance(TXT, list):
        for txt in TXT:
            log(to_print, txt)
    else:
        TXT.write(to_print+'\n')
        # print(to_print)

def test_compare_env():
    methods = ["A*Uniform", "A*Custom", "GBFCustom"]  # ,"random", "A*Uniform", "A*Custom", "GBFCustom"
    timeout = 5
    num_problems = 1  # in total three layouts: simple 4 by 4, default 5 by 5, difficult 6 by 7
    num_layouts = 10
    GENERATE_PNG = False

    ## create a new test folder to store output
    test_folder = join('tests', datetime.datetime.now().strftime("%m%d-%H-%M-%S"))
    os.mkdir(test_folder)
    summary_TXT = open(repair_suffix(join(test_folder, 'test_statistics'), 'txt'), "w")

    ## test each layout with
    env_motion = RobotKitchenEnv()
    env_relational = RobotKitchenEnvRelationalAction()
    env_chars = {env_motion: 'M', env_relational: 'R'}

    ## create problem files by randomly shuffle the layout of predefined problems
    for problem in range(num_problems):
        env = RobotKitchenEnv()
        env.fix_problem_index(problem)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        problem_path = join(dir_path, test_folder, 'problems')
        os.mkdir(problem_path)
        problem_files = env.make_shuffled_layouts(num_layouts, join(problem_path, f'P{problem}x'))

    ## to test what scaling factors are most useful for heuristic search
    ## node.f = node.cost + node.heuristic * scale
    original_test_folder = test_folder
    for scale in [10]:  # 1, 2, 5, 10, 20

        log(f'\n\n\n========================================================================\n'
            f'Scale = {scale}\n'
            f'========================================================================\n'
            f'', [TXT, summary_TXT])

        test_folder = join(test_folder, f'scale={scale}')
        os.mkdir(test_folder)

        ## create a txt file to save statistics
        TXT = open(repair_suffix(join(test_folder, 'test_statistics'), 'txt'), "w")
        table = pd.DataFrame(index=methods,
                             columns=['success_rate', 'time', 'nodes_expanded', 'steps_in_env'])

        for env in [env_relational, env_motion]:
            log(f'Environment: {env.__class__.__name__}\n', [TXT, summary_TXT])

            for method in methods:
                log(f'---- Method: {method} --------', TXT)
                data = np.zeros((len(problem_files), 4))

                for index in range(len(problem_files)):
                    problem_file = problem_files[index]
                    problem_name = re.search(r'.*\/([^.json]*)', problem_file).group(1)

                    ## initiate the environment
                    env.fix_problem_index(int(problem_name[1]))
                    env.init_problem_from_json(problem_file)
                    state, _ = env.reset()
                    name = f"{env_chars[env]}{problem_name}_{method}"

                    if GENERATE_PNG: display_image(env.render_from_state(state), name)

                    output_file = join(test_folder, name)
                    data[index] = plan(env, method, out_file=output_file,
                                       max_steps=40, timeout=timeout, heuristic_scale=scale)
                    log(f'          Problem: {problem_name} |  Stats: {data[index]}', TXT)

                    ## output the final state as a txt, for checking the goal configuration
                    env.problem_to_json(output_file)

                    ## generate the trace of the plan
                    if GENERATE_PNG:
                        png_filename = join(test_folder, env.__class__.__name__, name + '.png')
                        os.makedirs(os.path.dirname(png_filename), exist_ok=True)
                        plt.savefig(png_filename)

                # table.loc[method] = data.T.tolist()
                table.loc[method] = [get_stats(data[:, 0]),
                                     get_stats(data[:, 1]),
                                     get_stats(data[:, 2]),
                                     get_stats(data[:, 3])]
                print()

            log('-----------------------------------', [TXT, summary_TXT])
            log(str(table) + '\n\n', [TXT, summary_TXT])

        TXT.close()
        test_folder = original_test_folder
    summary_TXT.close()

def test_RobotKitchenEnvRelationalAction():
    env = RobotKitchenEnvRelationalAction(mode='simple')

    # ## test planning
    env.init_problem_from_json(join('tests', '0912-12-02-45', 'problems', 'P0x0.json'))
    plan(env, 'A*Custom', out_file='test', timeout=3, heuristic_scale=2)

    # ## test goal checking
    # env.init_problem_from_json(join('tests', '0912-12-02-45', '1', 'RP0x0_A*Custom.json'))
    # print(env.check_goal())
    # # display_image(env.render_from_state(), '')
    # # plt.show()
    #
    # env1 = RobotKitchenEnv(mode='simple')
    # env1.set_state(env.get_state())
    # print(env.check_goal())

if __name__ == "__main__":
    test_compare_env()
    # test_RobotKitchenEnvRelationalAction()

