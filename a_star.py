"""Simple implementation of A*"""
from graph import graph, heuristic, START, END, EDGE_COST


def search():
    """Perform the A* search"""

    # Init the open list, closed list, parents, gcosts, and hcosts
    open_list = []
    closed_list = []
    parents = {}
    gcosts = {}
    hcosts = {}

    # Set the current node to the start node and add to the lists and dicts
    current_node = START
    parents[START] = -1
    gcosts[START] = 0
    open_list.append(START)

    # Get the next node from the open list that is not yet in the closed list
    while len(open_list) > 0:
        for _ in range(len(open_list)):
            new_node = open_list.pop(0)
            if new_node not in closed_list:
                current_node = new_node
                break

        print(f"Current node: {current_node}, checking if at goal")
        # Check if the current node is the goal
        if current_node == END:
            break

        # Expand the node and add to the list
        attached_nodes = graph[current_node]
        # For each expanded node
        for node in attached_nodes:
            if node in closed_list:
                continue
            gcost = EDGE_COST + gcosts[current_node]
            hcost = heuristic[node]
            # If it's a new node, add it to the open list
            if node not in open_list:
                open_list.append(node)
                parents[node] = current_node
                gcosts[node] = gcost
                hcosts[node] = hcost
            # If it's already in the open list, check if the new path is better
            elif gcost + hcost < gcosts[node] + hcosts[node]:
                gcosts[node] = gcost
                hcosts[node] = hcost
                parents[node] = current_node
        # Now the nodes are added, sort the list to get the next node
        open_list = sorted(open_list, key=lambda x: gcosts[x] + hcosts[x])
        # Add to the closed list since we don't need to use it any more
        closed_list.append(current_node)

    # Now find the path back.
    path = []
    while current_node != -1:
        path.append(current_node)
        current_node = parents[current_node]

    path.reverse()
    print(path)


if __name__ == "__main__":
    search()
