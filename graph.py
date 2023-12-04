"""Very simple graph for testing purposes"""

# Graph is a dictionary of nodes and their attached nodes
# Heuristic is a dictionary of nodes and their heuristic values, standing in for a real heuristic function
# The edge cost is always 1, as if a basic grid-world
# Graph design is:
# 1  6  7  .  13
# 2  .  8  .  14
# 3  4  9  10 15
#
# With this design, Greedy, Dijkstra's, and A* should all find the same path but have different node expansion orders

graph = {
        1: [2, 6],
        2: [1, 3],
        3: [2, 4],
        4: [3, 9],
        5: [],
        6: [1, 7],
        7: [6, 8],
        8: [7, 9],
        9: [4, 8, 10],
        10: [9, 15],
        11: [],
        12: [],
        13: [14],
        14: [13, 15],
        15: [10, 14],
        }

START = 2
END = 13
EDGE_COST = 1

heuristic = {
        1: 4,
        2: 5,
        3: 6,
        4: 5,
        5: 4,
        6: 3,
        7: 2,
        8: 3,
        9: 4,
        10: 3,
        11: 2,
        12: 1,
        13: 0,
        14: 1,
        15: 2,

    }