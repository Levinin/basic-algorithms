from graph import START, END, heuristic, graph, EDGE_COST


def search():
    open_list = []
    closed_list = []

    parents = {}
    gcosts = {}
    hcosts = {}

    current_node = START
    parents[current_node] = -1
    gcosts[current_node] = 0
    hcosts[current_node] = heuristic[current_node]

    open_list.append(current_node)

    while len(open_list) > 0:

        for _ in range(len(open_list)):
            new_node = open_list.pop(0)
            if new_node not in closed_list:
                current_node = new_node
                break

        if current_node == END:
            break

        expanded_node = graph[current_node]
        for node in expanded_node:
            if node in closed_list:
                continue

            gc = gcosts[current_node] + EDGE_COST
            hc = heuristic[node]

            if node not in open_list:
                open_list.append(node)
                parents[node] = current_node
                gcosts[node] = gc
                hcosts[node] = hc
            elif gc + hc < gcosts[node] + hcosts[node]:
                parents[node] = current_node
                gcosts[node] = gc
                hcosts[node] = hc

        closed_list.append(current_node)
        open_list = sorted(open_list, key=lambda x: gcosts[x] + hcosts[x])

    path = []
    while current_node != -1:
        path.append(current_node)
        current_node = parents[current_node]

    path.reverse()
    print(f"The path from {START} to {END} is {path}.")


if __name__ == "__main__":
    search()