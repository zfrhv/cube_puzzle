from shapes import *
from _2_verify_second_shape import solve_2d
import numpy as np

my_parts = [{
    "repeats": 4,
    "matrix": np.array([
        [1,1,1,1],
        [0,0,0,1]]),
    "size": 4
},{
    "repeats": 1,
    "matrix": np.array([
        [1,1],
        [0,1]]),
    "size": 3
},{
    "repeats": 1,
    "matrix": np.array([
        [1,1,0],
        [0,1,1]]),
    "size": 4
},{
    "repeats": 1,
    "matrix": np.array([
        [1,1,1],
        [0,1,0]]),
    "size": 4
}]

def print_2d_shape(shape):
    for row in shape:
        for cell in row:
            print("⬛" if cell == 0 else "⬜" , end="")
        print()

print_2d_shape(heart1)
print(solve_2d(heart1, my_parts, [0,0]))