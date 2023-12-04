# Simple implementation of bubble sort
import random

def bubble_sort(data: list) -> list:
    for idx in range(len(data) - 1):
        for idy in range(len(data) - 1 - idx):
            if data[idy] > data[idy + 1]:
                tmp = data[idy]
                data[idy] = data[idy + 1]
                data[idy + 1] = tmp
    return data


if __name__ == "__main__":

    new_data = list(range(10000))
    random.shuffle(new_data)
    print(f"The shuffled data is: {new_data}")
    print(f"The sorted array is: {bubble_sort(new_data)}")

