import numpy as np
import matplotlib.pyplot as plt
import imageio
import functools
import time
from tqdm import tqdm
from enum import Enum
import random
import copy
import math
from os import listdir
from os.path import isfile, join
import json

try:
    import jacinle
except:
    pass

class OBJ_CATS(Enum):
    TABLE = 0
    BOX1 = 1
    BOX2 = 2
    BREAD1 = 3
    BREAD2 = 4
    BREAD3 = 5
    BREAD4 = 6
    LETTUCE1 = 7
    LETTUCE2 = 8
    MEAT1 = 9
    MEAT2 = 10
    ROBOT = 11
for name in [x for x in dir(OBJ_CATS) if not x.startswith('__')]:
    globals()[name] = getattr(OBJ_CATS, name).value

try:
    from .layouts import *
    from .utils import render_from_layout, get_asset_path, animate_images, repair_suffix
except ImportError:
    from layouts import *
    from utils import render_from_layout, get_asset_path, animate_images, repair_suffix

class RobotKitchenEnv:
    """A grid world where a robot hand must take out ingredients from containers and
    assemble meals (e.g., hamburger) according to orders.

    Parameters
    ----------
    layout: np.ndarray, layout.shape = (height, width, num_objects)
    goal: goal configuration, state attributes
        goal configuration: bottom-up sequence of type of ingredients
            e.g., [['D','l','m','D'], ['D','m','l','D']]
                  i.e., it's ok to put lettuce above or below the meat, but bread is always at top & bottom
        state attributes: a dictionary of attribute-value pairs that we want to be true
            e.g., {'carrying': None}, and in the future ('contain', container, contained)
    """

    ## Types of objects
    OBJECTS = OBJ_CATS

    ## used for printing and for token image in assets folder
    ## e.g., plt.imread(get_asset_path('table.png')),
    NAMES = {
        TABLE:'table', BOX1: 'box1', BOX2: 'box2',
        BREAD1: 'bread1', BREAD2: 'bread2', BREAD3: 'bread3', BREAD4: 'bread4',
        LETTUCE1: 'lettuce1', LETTUCE2: 'lettuce2',
        MEAT1: 'meat1', MEAT2: 'meat2',
        ROBOT: 'robot'
    }

    ## for rendering
    TOKEN_IMAGES = {}

    OBJECT_CHARS = {
        TABLE: "-",
        ROBOT: "R",
        BOX1: "B",
        BOX2: "B",
        BREAD1: "D",
        BREAD2: "D",
        BREAD3: "D",
        BREAD4: "D",
        LETTUCE1: "l",
        LETTUCE2: "l",
        MEAT1: "m",
        MEAT2: "m",
    }

    CONTAINERS = ['B']

    ## Actions
    ACTIONS = UP, DOWN, LEFT, RIGHT, PICK_UP, PICK_UP_CONTAINER, PUT_DOWN = range(7)

    ## Rewards
    REWARD_SUBGOAL = 0.5
    REWARD_GOAL = 1
    MAX_REWARD = max(REWARD_GOAL, REWARD_SUBGOAL)

    def __init__(self, layout=None, goal=None, mode='default'):
        if layout is None:
            layout, goal = self._get_layout_from_mode(mode)

        self._initial_layout = layout
        self._layout = layout.copy()
        self._visible_objects = self._init_visible_objects()

        ## configurations like DEFAULT_GOAL = [['D','l','m','D'], ['D','m','l','D']]
        self._goal, self._goal_attributes = goal
        self._goal_objects = self._init_goal_objects()

        ## relation attributes such as carrying and containing
        self._attributes_in_state = ['carrying']    ## only some attributes are in the state space
        self._attributes = self.check_attributes()    ## only for initialization

        self._init_token_images()

    def reset(self):
        self._layout = self._initial_layout.copy()
        self._visible_objects = self._init_visible_objects()
        self._attributes = self.check_attributes()
        return self.get_state(), {}

    def _get_layout_from_mode(self, mode):
        if isinstance(mode, int):
            mode = {0: 'simple', 1: 'default', 2: 'difficult'}[mode]
        if mode == 'default':
            layout = DEFAULT_LAYOUT
            goal = DEFAULT_GOAL
        elif mode == 'simple':
            layout = SIMPLE_LAYOUT
            goal = SIMPLE_GOAL
        elif mode == 'difficult':
            layout = DIFFICULT_LAYOUT
            goal = DIFFICULT_GOAL
        else:
            raise Exception("Unrecognized mode.")
        return layout, goal

    def _shuffle_layout(self):
        """ shuffle the positions of objects on the table and the robot """
        shape = self._layout.shape

        def random_col():
            return random.randint(0, shape[1]-1)

        ## clear all objects except for tables
        table_pos = self._find_all_pos_by_obj(TABLE)
        self._layout = np.zeros((shape[0], shape[1], len(OBJECTS)), dtype=bool)
        for row, col in table_pos:
            self._layout[row, col, TABLE] = 1

        ## select a column to put robot arm
        rob_c = random_col()

        ## get object blocks to shuffle
        objects = self._visible_objects
        objects.remove(ROBOT)
        objects.remove(TABLE)
        for container, contained in self._attributes['containing'].items():
            for obj in contained:
                objects.remove(obj)

        blocks = np.zeros(shape[1], dtype=int)
        while len(objects) > 0:
            bin = random_col()

            ## can put objects between table and one block away from ceiling
            if blocks[bin] < shape[0] - 2:
                ## take a random object from object list and move to that position
                object = random.choice(objects)
                objects.remove(object)
                self._move_obj_to(object, (shape[0]-2-blocks[bin], bin))
                blocks[bin] += 1

        ## move robot to the top of a random column
        self._move_obj_to(ROBOT, (0, rob_c))
        self.state_to_str()
        self._initial_layout = self._layout

        return self.reset()[0]

    def make_shuffled_layouts(self, num_layouts, out_prefix):
        """ shuffle n times and return the list of problem json files for each """
        out_files = []
        for i in range(num_layouts):
            out_file = out_prefix + str(i) + '.json'
            self._shuffle_layout()
            self.problem_to_json(join('..', out_file))
            self.record_from_trace([self.get_state()], out_file.replace('.json','.gif'))
            out_files.append(out_file)
        return out_files

    def fix_problem_index(self, num):
        layout, goal = self._get_layout_from_mode(num)
        self._initial_layout = layout
        self.reset()
        self._visible_objects = self._init_visible_objects()
        self._goal, self._goal_attributes = goal  ## goal consists of the matching configuration and key attributes
        self._goal_objects = self._init_goal_objects()

    def _get_container(self, objects):
        """ return the container object among all objects in the grid """
        containers = [o for o in objects if self.OBJECT_CHARS[o] in self.CONTAINERS]
        if len(containers) != 0: return containers[0]
        return None

    def _get_non_container_obj(self, objects):
        """ given all objects in a grid,
            return the list of objects inside the container or out in th air """
        free_obj = [o for o in objects if self.OBJECT_CHARS[o] not in self.CONTAINERS]
        if len(free_obj) != 0:
            if ROBOT in free_obj: free_obj.remove(ROBOT)
            return free_obj
        return [None]

    def check_attributes(self):
        # update and return a dictionary that contains a relational representation of the state.
        self._attributes = {
            'carrying': None,
            'containing': {},
            'contained': {},
        }
        rows, cols = np.nonzero(np.sum(self._layout, axis=2) > 1)
        for i in range(len(rows)):
            objects = self._get_objs_in_pos((rows[i], cols[i]))
            contained = self._get_non_container_obj(objects)
            container = self._get_container(objects)
            if container!=None:
                if self.OBJECT_CHARS[container] == 'R':
                    self._attributes['carrying'] = contained[0]
                elif self.OBJECT_CHARS[container] == 'B':
                    self._attributes['containing'][container] = contained
                    for o in contained:
                        self._attributes['contained'][o] = container
        return self._attributes

    def get_all_actions(self):
        return [a for a in self.ACTIONS]

    def _init_visible_objects(self):
        self._visible_objects = []
        for obj_cat in self.OBJECTS:
            if len(self._find_pos_by_obj(obj_cat.value)) != 0:
                self._visible_objects.append(obj_cat.value)
        return self._visible_objects

    def _init_goal_objects(self):
        # set(x for lst in self._goal for x in lst)  ## for giving reward for achieving subgoals
        config = random.choice(self._goal)
        self._goal_objects = [TABLE]
        choosing = copy.deepcopy(self.OBJECT_CHARS)
        for item in config:
            objects = []
            for obj_cat, char in choosing.items():
                if item == char and obj_cat in self._visible_objects:
                    objects.append(obj_cat)
            chosen_obj = random.choice(objects)
            self._goal_objects.append(chosen_obj)
            # print('removed', chosen_obj, choosing[chosen_obj])
            choosing.pop(chosen_obj) ## don't choose the same object as target object
        # print(config, goal_objects)
        return self._goal_objects

    def render(self, dpi=150):
        return render_from_layout(self._layout, self._get_token_images, dpi=dpi)

    def render_from_state(self, state):
        self.set_state(state)
        return self.render()

    def record_from_trace(self, trace, out_file, dpi=300):
        """ plot the path on map from trace, which is list of robot positions """
        images = []
        for state in trace:
            self.set_state(state)
            images.append(self.render(dpi=dpi))
        animate_images(out_file, images)
        return out_file

    def problem_to_json(self, out_file):
        data = {
            'shape': self._layout.shape,
            'goal': (self._goal, self._goal_attributes),
            'state': str(self.get_state()),
        }
        out_file = repair_suffix(out_file,'JSON')
        with open(out_file, 'w') as out_file:
            json.dump(data, out_file)

    def init_problem_from_json(self, in_file):
        data = json.load(open(in_file, 'r'))
        shape = data['shape']
        self._layout = np.zeros((shape[0], shape[1], len(OBJECTS)), dtype=bool)
        self.set_state(eval(data['state']))
        self._initial_layout = self._layout
        self._goal, self._goal_attributes = data['goal']
        return self.reset()[0]

    def _init_token_images(self):
        for obj in self.OBJECTS:
            self.TOKEN_IMAGES[obj.value] = plt.imread(get_asset_path(str(self.NAMES[obj.value]) + '.png'))

    def _get_token_images(self, obs_cell):
        images = []
        for token in self.OBJECTS:
            if obs_cell[token.value]:
                images.append(self.TOKEN_IMAGES[token.value])
        return images

    def _get_objs_in_pos(self, pos):
        objs = np.nonzero(self._layout[pos[0], pos[1]])
        if len(objs) > 0: return objs[0].tolist()
        print('!! No objects found in position', pos)
        return []

    def _find_all_pos_by_obj(self, obj):
        positions = np.argwhere(self._layout[..., obj])
        if len(positions) > 0: return positions
        # print('!! No positions found for object', obj)
        return []

    def _find_pos_by_obj(self, obj):
        positions = np.argwhere(self._layout[..., obj])
        if len(positions) > 0: return positions[0]
        # print('!! No positions found for object', obj)
        return []

    def get_robot_pos(self, state=None):
        self.set_state(state)
        return self._find_pos_by_obj(ROBOT)

    def _set_robot_carrying(self, obj):
        if obj not in [ROBOT, TABLE] and self._attributes['carrying'] == None:
            self._attributes['carrying'] = obj

            ## objects in a container will no longer be in container
            self._stop_obj_contained(obj)

    def _stop_robot_carrying(self):
        """ make the robot release the object it's carrying
            return the object that is containing or supporting the released object
        """

        ## cannot releasing an object in the air, for the moment
        rob_pos = self.get_robot_pos()
        objects_below = self._get_obj_below_pos(rob_pos)
        if len(objects_below) == 0: return None

        obj = self._attributes['carrying']
        if obj != None:
            self._attributes['carrying'] = None

            ## the object is now contained if there exists a container
            objects = self._get_objs_in_pos(rob_pos)
            container = self._get_container(objects)
            if container and container != obj:
                self._set_obj_contain(container, obj)

        return obj

    def _stop_obj_contained(self, contained):
        """ a contained object is no longer contained """
        if contained in self._attributes['contained']:
            container = self._attributes['contained'][contained]
            self._attributes['contained'].pop(contained)
            self._attributes['containing'][container].remove(contained)
            if len(self._attributes['containing'][container]) == 0:
                self._attributes['containing'].pop(container)

    def _set_obj_contain(self, container, contained):
        """ a container object starts to contain another object """
        if container == contained: return
        if container not in self._attributes['containing']:
            self._attributes['containing'][container] = []
        if contained not in self._attributes['containing'][container]:
            self._attributes['containing'][container].append(contained)
            self._attributes['contained'][contained] = container

    def _move_obj_from_to(self, obj, ori, des):

        # Remove old object
        if len(ori) == 2:
            self._layout[ori[0], ori[1], obj] = 0

        # Add new object
        self._layout[des[0], des[1], obj] = 1

        ## objects carried by the robot move together with it
        if obj == ROBOT:
            ## the things that robot carries will move with it
            if self._attributes['carrying'] != None:
                object = self._attributes['carrying']
                self._move_obj_from_to(object, ori, des)

        ## objects in a container move together with it
        if obj in self._attributes['containing']:
            for carried_object in self._attributes['containing'][obj]:
                self._move_obj_from_to(carried_object, ori, des)

    def _move_obj_to(self, obj, des):
        # print('obj =', obj, 'obj_pos =', self._find_pos_by_obj(obj), 'des =', des)
        self._move_obj_from_to(obj, self._find_pos_by_obj(obj), des)

    def _get_obj_above_pos(self, pos):
        # return a list of objects above obj_r and at obj_c.
        obj_r, obj_c = pos
        if obj_r > 0:
            objects = self._get_objs_in_pos((obj_r-1, obj_c))
            if len(objects) > 0:
                objects = self._remove_robot_and_carried(objects)
                return objects
        return []

    def _get_obj_above_obj(self, obj):
        return self._get_obj_above_pos(self._find_pos_by_obj(obj))

    def _get_obj_below_pos(self, pos):
        obj_r, obj_c = pos
        if obj_r < self._layout.shape[0]-1:
            objects = self._get_objs_in_pos((obj_r+1, obj_c))
            if len(objects) > 0:
                objects = self._remove_robot_and_carried(objects)
                return objects
        return []

    def _remove_robot_and_carried(self, objects):
        if ROBOT in objects:
            objects.remove(ROBOT)
            carrying = self._attributes['carrying']
            if carrying != None:
                if carrying in objects:
                    objects.remove(carrying)
                if carrying in self._attributes['containing']:
                    for contained in self._attributes['containing'][carrying]:
                        if contained in objects:
                            objects.remove(contained)
        return objects

    def _get_obj_below_obj(self, obj):
        return self._get_obj_below_pos(self._find_pos_by_obj(obj))

    def _can_move_to_pos(self, new_pos, action):
        new_r, new_c = new_pos

        ## cannot move outside the box
        if new_r<0 or new_r>=self._layout.shape[0] or new_c<0 or new_c >= self._layout.shape[1]: return False

        ## there cannot be objects above the position
        if len(self._get_obj_above_pos(new_pos)) != 0: return False

        ## cannot move to table
        objects = self._get_objs_in_pos(new_pos)
        if TABLE in objects: return False

        ## if the robot is carrying stuff, there cannot be not contained object in new_pos
        if self._attributes['carrying'] != None:
            if self._get_container(objects) == None:
                free_objects = self._get_non_container_obj(objects)
                if None not in free_objects: return False

        ## if robot is inside a container (not carrying it), the robot can only move up
        container = self._get_container(self._get_objs_in_pos(self.get_robot_pos()))
        if container != None:
            if container != self._attributes['carrying'] and action != UP: return False

        return True

    def step(self, action, DEBUG=False, STEP=True):

        # Start out reward at 0
        reward = 0
        rob_pos = rob_r, rob_c = self._find_pos_by_obj(ROBOT)

        # Move the robot, along with the object if it has one
        if action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:

            dr, dc = {self.UP : (-1, 0), self.DOWN : (1, 0),
                      self.LEFT : (0, -1), self.RIGHT : (0, 1)}[action]
            new_pos = rob_r + dr, rob_c + dc

            if self._can_move_to_pos(new_pos, action):
                self._move_obj_from_to(ROBOT, rob_pos, new_pos)


        elif action in [self.PICK_UP, self.PICK_UP_CONTAINER]:  ## only valid if there exists object to pick up

            ## Carry the object if there is any in the new grid
            objects = self._get_objs_in_pos(rob_pos)
            objects.remove(ROBOT)
            if len(objects) > 0:

                ## choose which object to pick
                if action == self.PICK_UP_CONTAINER:
                    object = self._get_container(objects)
                elif action == self.PICK_UP:
                    ## otherwise pick up the object with smaller index
                    object = self._get_non_container_obj(objects)[0]

                if object != None:
                    self._set_robot_carrying(object)

                    ## get reward for picking up objects appeared in goal configuration
                    if object in self._goal_objects:
                        reward += self.REWARD_SUBGOAL


        elif action == self.PUT_DOWN:  ## only valid if there is an object in robot hand

            support_object = self._stop_robot_carrying()

            ## get reward for putting on objects appeared in goal configuration
            if support_object in self._goal_objects:
                reward += self.REWARD_SUBGOAL


        ## Check done: all people quenched
        done = self.check_goal()
        if done: reward += self.REWARD_GOAL

        return self.get_state(DEBUG=False), reward, done, {}

    def get_state(self, DEBUG=False):
        layout_state = tuple(('layout', tuple(sorted(map(tuple, np.argwhere(self._layout))))))
        states = [layout_state]
        for attribute in self._attributes_in_state: ## for the attributes we care about
            states.append(tuple((attribute, self._attributes[attribute])))
        state = tuple(states)
        if DEBUG: print(state)
        return state

    def _get_state_var(self, trg_var, state=None):
        if state==None: state = self.get_state()
        for var, value in state:
            if trg_var == var:
                return value
        return None

    def set_state(self, state):
        if state:
            self._layout = np.zeros_like(self._initial_layout)
            for i, j, k in self._get_state_var('layout', state):
                self._layout[i, j, k] = 1
            # self.check_attributes()
            for attribute in self._attributes_in_state:
                self._attributes[attribute] = self._get_state_var(attribute, state)
        return state

    def state_to_str(self, state=None):
        if state==None: state=self.get_state()
        layout = np.full(self._initial_layout.shape[:2], " ", dtype=object)
        for i, j, k in self._get_state_var('layout', state):
            layout[i, j] = self.OBJECT_CHARS[k]
        return '\n' + '\n'.join(''.join(row) for row in layout)

    def get_distance(self, obj1, obj2, state=None):
        self.set_state(state)
        pos1 = self._find_pos_by_obj(obj1)
        pos2 = self._find_pos_by_obj(obj2)
        return math.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))

    @functools.lru_cache(maxsize=1000)
    def compute_reward(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        _, reward, _, _ = self.step(action)
        self.set_state(original_state)
        return reward

    @functools.lru_cache(maxsize=1000)
    def compute_transition(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        next_state, _, _, _ = self.step(action)
        self.set_state(original_state)
        return next_state

    def get_successor_state(self, state, action):
        return self.compute_transition(state, action)

    @functools.lru_cache(maxsize=1000)
    def compute_done(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        _, _, done, _ = self.step(action)
        self.set_state(original_state)
        return done

    @functools.lru_cache(maxsize=1000)
    def check_goal(self, state=None):
        """ match each ingredient from bottom up """

        def subfinder(mylist, pattern):
            for start_index in range(len(mylist) - len(pattern) + 1):
                found = True
                for count_index in range(len(pattern)):
                    found = found and mylist[start_index+count_index] == pattern[count_index]
                if found: return start_index+count_index
            return None

        if state: self.set_state(state)

        ## all attributes in state must match goal attribute state
        for attribute in self._attributes_in_state:
            if self._attributes[attribute] != self._goal_attributes[attribute]:
                return False

        for col in range(self._layout.shape[1]):
            config = np.nonzero(self._layout[:,col,:])[1].tolist()
            config = [self.OBJECT_CHARS[x] for x in config]
            for goal in self._goal:
                row = subfinder(config, goal)
                if row != None:
                    # print('!! achieved goal', goal, 'at pos', (row, col))
                    return True
        return False


class RobotKitchenEnvRelationalAction(object):
    ACTION_CATS = PICK_UP, PLACE_ON = range(2)

    def __init__(self, layout=None, goal=None, mode='default'):
        super().__init__()
        self.wrapped = RobotKitchenEnv(layout, goal, mode)

        # Initialize the actions.
        self.actions = list()
        for obj in self.wrapped.OBJECT_CHARS:
            if obj != TABLE:
                self.actions.append((self.PICK_UP, obj))
            self.actions.append((self.PLACE_ON, obj))
        self.actions = tuple(self.actions)

    @property
    def _goal_objects(self):
        return self.wrapped._goal_objects

    def fix_problem_index(self, num):
        self.wrapped.fix_problem_index(num)

    def reset(self):
        return self.wrapped.reset()

    def _shuffle_layout(self):
        return self.wrapped._shuffle_layout()

    def render(self, dpi=150):
        return self.wrapped.render(dpi=dpi)

    def render_from_state(self, state=None):
        self.set_state(state)
        return self.render()

    def record_from_trace(self, trace, out_file, dpi=300):
        return self.wrapped.record_from_trace(trace, out_file, dpi)

    def problem_to_json(self, out_file):
        self.wrapped.problem_to_json(out_file)

    def init_problem_from_json(self, in_file):
        return self.wrapped.init_problem_from_json(in_file)

    def get_robot_pos(self, state=None):
        return self.wrapped.get_robot_pos(state)

    def get_distance(self, obj1, obj2, state=None):
        return 1

    def get_all_actions(self):
        return self.actions

    def step(self, action, DEBUG = False):
        robot_position = self.wrapped._find_pos_by_obj(ROBOT)

        def _check_empty(x):
            return x is None or isinstance(x, (tuple, list)) and len(x) == 0

        action_cat, action_obj = action
        if action_cat == self.PICK_UP:
            if self.wrapped._attributes['carrying'] is None and not _check_empty(self.wrapped._find_pos_by_obj(action_obj)):

                if _check_empty(self.wrapped._get_obj_above_obj(action_obj)):
                    self.wrapped._move_obj_to(ROBOT, self.wrapped._find_pos_by_obj(action_obj))
                    self.wrapped._set_robot_carrying(action_obj)  ## TODO: Test if this works

                    # self.wrapped._attributes['carrying'] = action_obj
                    # if action_obj in self.wrapped._attributes['contained']:
                    #     container = self.wrapped._attributes['contained'][action_obj]
                    #     self.wrapped._attributes['containing'][container].remove(action_obj)
                    #     self.wrapped._attributes['contained'].pop(action_obj)

        elif action_cat == self.PLACE_ON:

            object = self.wrapped._attributes['carrying']
            if object is not None:
                if action_obj == TABLE:
                    all_table_pos = self.wrapped._find_all_pos_by_obj(TABLE)
                    found = None
                    for pos in all_table_pos:
                        if len(self.wrapped._get_obj_above_pos(pos)) == 0:
                            found = tuple(pos)
                            break
                    if found is not None:
                        found = (found[0] - 1, found[1])
                        self._step_place_on(*found)
                else:
                    if not _check_empty(self.wrapped._find_pos_by_obj(action_obj)):
                        tgt_r, tgt_c = self.wrapped._find_pos_by_obj(action_obj)
                        if tgt_r >= 1:
                            tgt_r -= 1
                            self._step_place_on(tgt_r, tgt_c)
        else:
            raise ValueError('Unknown action category: {}'.format(action_cat))

        ## Check done: all people quenched
        done = self.check_goal()

        reward = 0
        if done: reward = self.REWARD_GOAL

        return self.get_state(), reward, done, {}

    def _step_place_on(self, tgt_r, tgt_c):
        object = self.wrapped._attributes['carrying']
        if object is not None:
            self.wrapped._move_obj_to(object, (tgt_r, tgt_c))
        self.wrapped._move_obj_to(ROBOT, (tgt_r, tgt_c))

        objects_below = self.wrapped._get_obj_below_pos((tgt_r, tgt_c))
        if len(objects_below) > 0:
            self.wrapped._stop_robot_carrying()  ## TODO: Test if this works
            # object = self.wrapped._attributes['carrying']
            # if object:
            #     self.wrapped._attributes['carrying'] = None
            #
            #     ## the object is now contained if there exists a container
            #     objects = self.wrapped._get_objs_in_pos((tgt_r, tgt_c))
            #     container = self.wrapped._get_container(objects)
            #     if container:
            #         if container not in self.wrapped._attributes['containing']:
            #             self.wrapped._attributes['containing'][container] = []
            #         self.wrapped._attributes['containing'][container].append(object)
            #         self.wrapped._attributes['contained'][object] = container

    def get_state(self):
        return self.wrapped.get_state()
        return tuple(sorted(map(tuple, np.argwhere(self._layout))))

    def set_state(self, state):
        return self.wrapped.set_state(state)

    def state_to_str(self, state=None):
        return self.wrapped.state_to_str(state)

    @functools.lru_cache(maxsize=1000)
    def compute_reward(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        _, reward, _, _ = self.step(action)
        self.set_state(original_state)
        return reward

    @functools.lru_cache(maxsize=1000)
    def compute_transition(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        next_state, _, _, _ = self.step(action)
        self.set_state(original_state)
        return next_state

    def get_successor_state(self, state, action):
        return self.compute_transition(state, action)

    @functools.lru_cache(maxsize=1000)
    def compute_done(self, state, action):
        original_state = self.get_state()
        self.set_state(state)
        _, _, done, _ = self.step(action)
        self.set_state(original_state)
        return done

    @functools.lru_cache(maxsize=1000)
    def check_goal(self, state=None):
        """ match each ingredient from bottom up """
        return self.wrapped.check_goal(state=state)


# For TEST ONLY.
ACTIONS = UP, DOWN, LEFT, RIGHT, PICK_UP, PICK_UP_CONTAINER, PUT_DOWN = range(7)


# Section 1: test for the motion planning setting.


def test_get_state():
    """ test state representation """
    env = RobotKitchenEnv()
    state_1 = env.get_state(DEBUG=True)

    trg_state = test_steps(DEBUG=False).get_state(DEBUG=True)

    env.set_state(trg_state)
    state_2 = env.get_state(DEBUG=True)


def test_goal_checking():
    """ directly move objects around and see if the goal table configuration is met"""

    env = RobotKitchenEnv()
    print(env._goal)
    print(env.state_to_str())
    env._move_obj_to(MEAT1, (2,3))
    env._move_obj_to(LETTUCE1, (1,3))
    env._move_obj_to(BREAD1, (0,3))

    env.check_goal()

    outfile = join('tests', "test_goal_checking")
    images = [env.render(dpi=300)]
    animate_images(outfile, images)


def test_steps(DEBUG=True):
    dpi = 300
    env = RobotKitchenEnv()

    ## some actions are invalid, check if the corresponding warning messages are printed out
    actions = [DOWN, DOWN, PICK_UP, DOWN,
               UP, RIGHT, RIGHT, RIGHT, RIGHT, DOWN, PUT_DOWN,
               UP, LEFT, LEFT, LEFT, LEFT, DOWN, PICK_UP_CONTAINER,
               RIGHT, DOWN, PUT_DOWN]

    images = []
    state, _ = env.reset()
    images.append(env.render(dpi=dpi))
    for action in tqdm(actions):
        state, reward, done, _ = env.step(action, DEBUG = DEBUG)
        images.append(env.render(dpi=dpi))

    outfile = join('tests', "test_steps")
    animate_images(outfile, images)

    return env


def test_custom_layout():
    dpi = 300

    # Create layouts of 5 by 7
    """
    +--+--+--+--+--+--+--+
    |RB|  |  |  |  |  |  |
    +--+--+--+--+--+--+--+
    |  |  |  |  |  |  |  |
    +--+--+--+--+--+--+--+
    |  |  |  |  |  |  |  |
    +--+--+--+--+--+--+--+
    |  |  |  |  |  |  |  |
    +--+--+--+--+--+--+--+
    |BL|BM|  |D4|D3|D2|D1|
    +--+--+--+--+--+--+--+
    |TB|TB|TB|TB|TB|TB|TB|
    +--+--+--+--+--+--+--+
    """
    layout = np.zeros((6, 7, len(dir(OBJ_CATS))), dtype=bool)
    for i in range(7):
        layout[5, i, TABLE] = 1
    layout[0, 0, ROBOT] = 1
    layout[4, 3, BREAD1] = 1
    layout[4, 4, BREAD2] = 1
    layout[4, 5, BREAD3] = 1
    layout[4, 6, BREAD4] = 1
    layout[4, 1, MEAT1] = 1
    layout[4, 1, MEAT2] = 1
    layout[4, 1, BOX1] = 1
    layout[4, 0, LETTUCE1] = 1
    layout[4, 0, LETTUCE2] = 1
    layout[4, 0, BOX2] = 1

    ## only one ordering of ingredients that count as a MEGA hamburger
    goal = [['D', 'm', 'l', 'm', 'D']], {'carrying': None}
    env = RobotKitchenEnv(layout=layout, goal=goal)

    ## just to visualize the layout
    # outfile = join('tests', "test_custom_layout.mp4")
    # imageio.mimsave(outfile, [env.render(dpi=300)])

    ## all actions are valid, you can add invalid ones inside to try
    actions = [RIGHT, DOWN, DOWN, DOWN, DOWN, PICK_UP_CONTAINER, ## pick up the box of meat
               UP, RIGHT, RIGHT, RIGHT, PUT_DOWN, ## put the box of meat next to the target base bread
               PICK_UP, UP, RIGHT, DOWN, PUT_DOWN, ## put down the first piece of meat
               UP, LEFT, LEFT, LEFT, LEFT, LEFT, DOWN, DOWN, PICK_UP, ## pick up a piece of lettuce
               UP, UP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, PUT_DOWN, ## put down the piece of lettuce
               LEFT, DOWN, PICK_UP, ## pick up the second piece of meat
               UP, UP, RIGHT, PUT_DOWN, ## put down the second piece of meat
               LEFT, LEFT, DOWN, DOWN, DOWN, PICK_UP, ## pick up the upper bread
               UP, UP, UP, UP, RIGHT, RIGHT, PUT_DOWN] ## put down the upper bread

    images = []
    state, _ = env.reset()
    images.append(env.render(dpi=dpi))
    for action in tqdm(actions):
        state, reward, done, _ = env.step(action, DEBUG = True)
        images.append(env.render(dpi=dpi))

    outfile = join('tests', "test_custom_layout")
    animate_images(outfile, images)


def test_simple_layout():
    dpi = 300

    env = RobotKitchenEnv()
    env.fix_problem_index(0)

    ## just to visualize the layout
    # outfile = join('tests', "test_simple_layout.mp4")
    # imageio.mimsave(outfile, [env.render(dpi=300)])

    actions = [RIGHT, RIGHT, RIGHT, DOWN, DOWN, PICK_UP, UP, PUT_DOWN]

    images = []
    state, _ = env.reset()
    images.append(env.render(dpi=dpi))
    for action in tqdm(actions):
        state, reward, done, _ = env.step(action, DEBUG=True)
        images.append(env.render(dpi=dpi))

    outfile = join('tests', "test_simple_layout")
    animate_images(outfile, images)


# Section 2: test for the relational task planning setting.

def test_steps_relational():
    dpi = 300
    env = RobotKitchenEnvRelationalAction()

    ## some actions are invalid, check if the corresponding warning messages are printed out
    actions = [(env.PICK_UP, MEAT1), (env.PLACE_ON, BREAD4)]

    images = []
    state, _ = env.reset()
    # print(env.state_to_str(state))
    # print(state)
    images.append(env.render(dpi=dpi))

    for action in tqdm(actions):
        state, reward, done, _ = env.step(action, DEBUG = True)
        # print(env.state_to_str(state))
        # print(state)
        images.append(env.render(dpi=dpi))

    outfile = join('tests', "test_steps_relational")
    animate_images(outfile, images)

def test_problem_from_json():
    json_file = join('tests', 'test_problem_from_json.json')

    env = RobotKitchenEnvRelationalAction()
    trace = [
        env.get_state(),
        env.step((env.PICK_UP, MEAT1))[0],
    ]

    env.problem_to_json(json_file)
    # env.state_to_txt(txt_name)
    trace.extend([
        env.step((env.PLACE_ON, BREAD4))[0],
        env.init_problem_from_json(json_file),
    ])

    outfile = join('tests', 'test_problem_from_json')
    env.record_from_trace(trace, outfile)

def test_checking_goals_from_json():
    mypath = "tests"
    files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and '.txt' in str(f)]
    for file_path in files:
        print(file_path)
        env = RobotKitchenEnv(txt=file_path)
        env.check_goal()

def test_shuffle_layout():
    env = RobotKitchenEnvRelationalAction(mode='simple')
    frames = [env.get_state()]
    for i in range(10):
        frames.append(env._shuffle_layout())
    env.record_from_trace(frames, join('tests', 'test_shuffle_layout'))

if __name__ == "__main__":
    """ the following test cases helps you understand the basic functions of the environment """

    ## ------------ RobotKitchenEnv()

    ## test state representation
    # test_get_state()

    ## directly move objects around, test env.check_goal()
    # test_goal_checking()

    ## given a sequence of valid and invalid actions, test env.step()
    # test_steps()

    ## given custom layout and goal configuration, test environment initialization()
    # test_custom_layout()

    ## using the simple layout defined in layout.py
    # test_simple_layout()



    ## ------------ RobotKitchenEnvRelationalAction()

    ## test for the relational task planning setting.
    # test_steps_relational()

    ## generate and reset with state txt file
    test_problem_from_json()

    ## check if a list of txt files meet the goal
    # test_checking_goals_from_json()

    ## shuffle objects on the table
    # test_shuffle_layout()