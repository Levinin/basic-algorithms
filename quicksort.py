# Simple quicksort algorithm practice
#
import random

# Hoare partition scheme
def partition(data, lo, hi):
    pivot = data[((hi - lo) // 2) + lo]    # Taking the middle value

    i = lo - 1
    j = hi + 1

    while True:
        do_latch = True
        while do_latch or data[i] < pivot:
            i += 1
            do_latch = False

        do_latch = True
        while do_latch or data[j] > pivot:
            j -= 1
            do_latch = False

        if i >= j:
            return j

        data[i], data[j] = data[j], data[i]


def quicksort(data, lo, hi):
    if lo >= 0 and hi >= 0 and lo < hi:
        p = partition(data, lo, hi)
        quicksort(data, lo, p)
        quicksort(data, p + 1, hi)
    return data


if __name__ == "__main__":
    new_data = list(range(10_000))
    random.shuffle(new_data)
    quicksort(new_data,0,len(new_data) - 1)
    # print(f"The shuffled data is: {new_data}")
    # print(f"The sorted array is: {quicksort(new_data,0,len(new_data) - 1)}")


