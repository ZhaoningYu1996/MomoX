import random

def split_list(input_list, ratio=0.8):
    # Shuffle the list
    shuffled = input_list[:]
    random.shuffle(shuffled)

    # Split index
    split_idx = int(len(shuffled) * ratio)

    # Split the list
    part1 = shuffled[:split_idx]
    part2 = shuffled[split_idx:]

    return part1, part2

# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# part1, part2 = split_list(my_list, 0.7)

# print("70% of the list:", part1)
# print("30% of the list:", part2)
