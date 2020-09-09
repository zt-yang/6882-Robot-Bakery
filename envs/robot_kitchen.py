import numpy as np
import matplotlib.pyplot as plt
import imageio

from utils import get_asset_path,render_from_layout

ACTIONS = UP, DOWN, LEFT, RIGH, PICK_UP, DROP_OFF = range(4)

class RobotKitchenEnv:
    """A grid world where a robot hand must take out ingredients from containers and
    assemble meals (e.g., hamburger) according to orders.

    Parameters
    ----------
    layout: np.ndarray, layout.shape = (height, width, num_objects)
    initial states
    """

    ## Types of objects
    OBJECTS = TABLE, BOX1, BOX2, \
              BREAD1, BREAD2, BREAD3, BREAD4, \
              LETTUCE1, LETTUCE2, MEAT1, MEAT2, \
              ROBOT \
               = range(12)

    ## for rendering
    TOKEN_IMAGES = {
        TABLE: plt.imread(get_asset_path('table.png')),
        ROBOT: plt.imread(get_asset_path('robot.png')),
        BREAD1: plt.imread(get_asset_path('bread1.png')),
        BREAD2: plt.imread(get_asset_path('bread2.png')),
        BREAD3: plt.imread(get_asset_path('bread3.png')),
        BREAD4: plt.imread(get_asset_path('bread4.png')),
        LETTUCE1: plt.imread(get_asset_path('lettuce1.png')),
        LETTUCE2: plt.imread(get_asset_path('lettuce2.png')),
        MEAT1: plt.imread(get_asset_path('meat1.png')),
        MEAT2: plt.imread(get_asset_path('meat2.png')),
        BOX1: plt.imread(get_asset_path('box1.png')),
        BOX2: plt.imread(get_asset_path('box2.png')),
    }

    OBJECT_CHARS = {
        TABLE: "X",
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

    # Create layouts
    """
    +--+--+--+--+--+
    |RB|  |  |  |  |
    +--+--+--+--+--+
    |  |  |  |  |BD|
    +--+--+--+--+--+
    |BM|  |  |  |BD|
    +--+--+--+--+--+
    |BL|  |  |BD|BD|
    +--+--+--+--+--+
    |TB|TB|TB|TB|TB|
    +--+--+--+--+--+
    """
    DEFAULT_LAYOUT = np.zeros((5, 5, len(OBJECTS)), dtype=bool)
    for i in range(5):
        DEFAULT_LAYOUT[4, i, TABLE] = 1
    DEFAULT_LAYOUT[0, 0, ROBOT] = 1
    DEFAULT_LAYOUT[1, 4, BREAD1] = 1
    DEFAULT_LAYOUT[2, 4, BREAD2] = 1
    DEFAULT_LAYOUT[3, 4, BREAD3] = 1
    DEFAULT_LAYOUT[3, 3, BREAD4] = 1
    DEFAULT_LAYOUT[2, 0, MEAT1] = 1
    DEFAULT_LAYOUT[2, 0, MEAT2] = 1
    DEFAULT_LAYOUT[2, 0, BOX1] = 1
    DEFAULT_LAYOUT[3, 0, LETTUCE1] = 1
    DEFAULT_LAYOUT[3, 0, LETTUCE2] = 1
    DEFAULT_LAYOUT[3, 0, BOX2] = 1

    ## Actions
    ACTIONS = UP, DOWN, LEFT, RIGHT, PICK_UP, DROP_OFF = range(6)

    ## Rewards
    REWARD_SUBGOAL = 0.5
    REWARD_GOAL = 1
    MAX_REWARD = max(REWARD_GOAL, REWARD_SUBGOAL)

    def __init__(self, layout=None, mode='default'):
        if layout is None:
            if mode == 'default':
                layout = self.DEFAULT_LAYOUT
            else:
                raise Exception("Unrecognized mode.")
        self._initial_layout = layout
        self._layout = layout.copy()
        self._attributes = {}

    def reset(self):
        self._layout = self._initial_layout.copy()
        return self.get_state(), {}

    def get_state(self):
        return tuple(sorted(map(tuple, np.argwhere(self._layout))))

    def get_all_actions(self):
        return [a for a in self.ACTIONS]

    def render(self, dpi=150):
        return render_from_layout(self._layout, self._get_token_images, dpi=dpi)

    def _get_token_images(self, obs_cell):
        images = []
        for token in self.OBJECTS:
            if obs_cell[token]:
                images.append(self.TOKEN_IMAGES[token])
        return images

    def step(self, action):

        # Start out reward at 0
        reward = 0
        print(action)

        # Move the robot, along with the object if it has one
        rob_r, rob_c = np.argwhere(self._layout[..., self.ROBOT])[0]
        dr, dc = {self.UP : (-1, 0), self.DOWN : (1, 0),
                  self.LEFT : (0, -1), self.RIGHT : (0, 1)}[action]
        new_r, new_c = rob_r + dr, rob_c + dc
        if 0 <= new_r < self._layout.shape[0] and 0 <= new_c < self._layout.shape[1]:
            # Remove old robot
            self._layout[rob_r, rob_c, self.ROBOT] = 0
            # Add new robot
            self._layout[new_r, new_c, self.ROBOT] = 1

            # Carry the object if there is any in the new grid
            objects = np.nonzero(self._layout[new_r, new_c])
            if len(objects) > 0:
                self._attributes['carrying'] = object
                print('start carrying',object)

            # Update local vars
            rob_r, rob_c = new_r, new_c
        assert rob_r is not None, "Missing robot in grid"

        # # Handle water pickup
        # if self._layout[rob_r, rob_c, self.ROBOT] and self._layout[rob_r, rob_c, self.WATER]:
        #     # Make robot have water
        #     self._layout[rob_r, rob_c, self.ROBOT_WITH_WATER] = 1
        #     self._layout[rob_r, rob_c, self.ROBOT] = 0
        #     # Remove water from grid
        #     self._layout[rob_r, rob_c, self.WATER] = 0
        #     # Reward for water pickup
        #     reward += self.WATER_PICKUP_REWARD
        #
        # # Handle people quenching
        # if self._layout[rob_r, rob_c, self.ROBOT_WITH_WATER] and self._layout[rob_r, rob_c, self.PERSON]:
        #     # Quench person
        #     self._layout[rob_r, rob_c, self.QUENCHED_PERSON] = 1
        #     self._layout[rob_r, rob_c, self.PERSON] = 0
        #     # Reward for quenching
        #     reward += self.QUENCH_REWARD
        #
        # Check done: all people quenched
        done = False #(len(np.argwhere(self._layout[..., self.PERSON])) == 0)

        return self.get_state(), reward, done, {}

if __name__ == "__main__":
    dpi = 300
    images = []
    env = RobotKitchenEnv()

    actions = [RIGHT, RIGHT, RIGHT, RIGHT, DOWN, LEFT]
    actions = [DOWN, DOWN, RIGHT]

    state, _ = env.reset()
    images.append(env.render(dpi=dpi))
    for action in actions:
        state, reward, done, _ = env.step(action)
        images.append(env.render(dpi=dpi))

    outfile = "test2.mp4"
    imageio.mimsave(outfile, images)