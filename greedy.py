"""Simple implementation of the greedy algorithm"""
from graph import graph, heuristic, START, END

def search():
    """Perform the greedy search"""
    open_list = []
    closed_list = []
    parents = {}
    hcosts = {}

    current_node = START
    parents[START] = -1
    hcosts[START] = heuristic[START]

    open_list.append(START)

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
        for node in attached_nodes:
            if node in closed_list:
                continue
            cost = heuristic[node] + hcosts[current_node]
            if node not in open_list:
                open_list.append(node)
                parents[node] = current_node
                hcosts[node] = cost
            elif cost < hcosts[node]:
                hcosts[node] = cost
                parents[node] = current_node
        # Now the nodes are added, sort the list to get the next node
        open_list = sorted(open_list, key=lambda x: hcosts[x])
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
