from shapes import *
from _2_verify_second_shape import solve_2d
import numpy as np

my_parts = [{
    "repeats": 4,
    "matrix": np.array([
        [1,1,1],
        [0,0,1]]),
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

ayu_parts = [{
    "repeats": 2,
    "matrix": np.array([
        [1,1,1],
        [0,0,1]]),
    "size": 4
},{
    "repeats": 2,
    "matrix": np.array([
        [1,1,1],
        [0,1,1]]),
    "size": 5
},{
    "repeats": 1,
    "matrix": np.array([
        [1,1,0],
        [0,1,1]]),
    "size": 4
},{
    "repeats": 1,
    "matrix": np.array([
        [1,0,0],
        [1,1,0],
        [0,1,1]]),
    "size": 5
}]

def print_2d_shape(shape):
    for row in shape:
        for cell in row:
            print("⬛" if cell == 0 else "⬜" , end="")
        print()

print_2d_shape(heart1)
colorful_plane = np.zeros((6, 7), dtype=int)
succeded = solve_2d(heart1, ayu_parts, [0,0], colorful_plane)
print(succeded)
if succeded:
    for i, j in np.ndindex(colorful_plane.shape):
        if j==0:
            print()
        print("\033[4"+str(colorful_plane[i,j])+"m  ", end='\033[0m')