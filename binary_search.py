# Binary search


def binary_search(data: list, item: int):
    li, ri, ptr, item_idx = 0, len(data) - 1, 0, -1
    while li <= ri and item_idx == -1:
        ptr = int(li + (ri - li) / 2)
        if data[ptr] == item:
            item_idx = ptr
        elif data[ptr] > item:
            ri = ptr
        else:
            li = ptr + 1

    return item_idx


if __name__ == "__main__":

    for num in range(0, 100, 5):
        print(f"Number {num} is in position {binary_search(list(range(100)), num)}")
