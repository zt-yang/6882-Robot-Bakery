import numpy as np

try:
    from .robot_kitchen import OBJ_CATS
except ImportError:
    from robot_kitchen import OBJ_CATS

for name in [x for x in dir(OBJ_CATS) if not x.startswith('__')]:
    globals()[name] = getattr(OBJ_CATS, name).value

OBJECTS = dir(OBJ_CATS)
DEFAULT_GOAL_ATTRIBUTES = {'carrying': None} ## in all cases, the robot shouldn't hold anything in the end

""" Default layouts of 5 by 5 -- one meat and one lettuce """

"""
+--+--+--+--+--+
|RB|  |  |  |  |
+--+--+--+--+--+
|  |  |  |  |D1|
+--+--+--+--+--+
|BM|  |  |  |D2|
+--+--+--+--+--+
|BL|  |  |D4|D3|
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

DEFAULT_GOAL = [['D','l','m','D'], ['D','m','l','D']], DEFAULT_GOAL_ATTRIBUTES



""" Simple layouts of 4 by 4 -- one meat only """

"""
+--+--+--+--+
|RB|  |  |  |
+--+--+--+--+
|  |  |  |  |
+--+--+--+--+
|BL|BM|D3|D1|
+--+--+--+--+
|TB|TB|TB|TB|
+--+--+--+--+--+--+--+
"""

SIMPLE_LAYOUT = np.zeros((4, 4, len(OBJECTS)), dtype=bool)
for i in range(4):
    SIMPLE_LAYOUT[3, i, TABLE] = 1
SIMPLE_LAYOUT[0, 0, ROBOT] = 1
SIMPLE_LAYOUT[2, 3, BREAD1] = 1
SIMPLE_LAYOUT[2, 2, BREAD3] = 1
SIMPLE_LAYOUT[2, 1, MEAT1] = 1
SIMPLE_LAYOUT[2, 0, LETTUCE1] = 1

## only one ordering of ingredients that count as a MEGA hamburger
SIMPLE_GOAL = [['D', 'm', 'D']], DEFAULT_GOAL_ATTRIBUTES



""" Difficult layouts of 6 by 7 -- two meat and one lettuce """

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
DIFFICULT_LAYOUT = np.zeros((6, 7, len(OBJECTS)), dtype=bool)
for i in range(7):
    DIFFICULT_LAYOUT[5, i, TABLE] = 1
DIFFICULT_LAYOUT[0, 0, ROBOT] = 1
DIFFICULT_LAYOUT[4, 3, BREAD1] = 1
DIFFICULT_LAYOUT[4, 4, BREAD2] = 1
DIFFICULT_LAYOUT[4, 5, BREAD3] = 1
DIFFICULT_LAYOUT[4, 6, BREAD4] = 1
DIFFICULT_LAYOUT[4, 1, MEAT1] = 1
DIFFICULT_LAYOUT[4, 1, MEAT2] = 1
DIFFICULT_LAYOUT[4, 1, BOX1] = 1
DIFFICULT_LAYOUT[4, 0, LETTUCE1] = 1
DIFFICULT_LAYOUT[4, 0, LETTUCE2] = 1
DIFFICULT_LAYOUT[4, 0, BOX2] = 1

## only one ordering of ingredients that count as a MEGA hamburger
DIFFICULT_GOAL = [['D', 'm', 'l', 'm', 'D']], DEFAULT_GOAL_ATTRIBUTES


def test_OBJ_CATS():
    print(OBJ_CATS.ROBOT)
    for name in [x for x in dir(OBJ_CATS) if not x.startswith('__')]:
        globals()[name] = getattr(OBJ_CATS, name)
    print(ROBOT)



if __name__ == "__main__":
    test_OBJ_CATS()