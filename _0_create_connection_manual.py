import numpy as np

from _1_create_tetris_parts import merge_cubes,convert_parts_2d,connection_cords,point_cords,at
from _2_verify_second_shape import inspect_parts,solve_2d,solve_3d
from _5_create_meshes import create_meshes
from shapes import *

# # Values from 0 to 1
# fun_meter = 1.0 # the more varaity of shapes the more fun it is (many same shapes is boring)
# _2d_difficulty_meter = 1.0 # the harder to assembly the shape 8x8 the harder it is
# _3d_difficulty_meter = 1.0 # the harder to assembly the shape 4x4x4 the harder it is

class IntegerWrapper:
    def __init__(self, value):
        self.value = value

# how often to have connections from 0 to 1 (1 = always make connections, 0 means never), more connections more longer parts
often_connect = 0.5
part_length = 4
def connect_cubes(connections, is_checked, cube_pos, part_length, part_axis):
    if part_length.value > 0:
      directions = ['up', 'down', 'front', 'back', 'right', 'left']
      np.random.shuffle(directions)
      for direction in directions:
          attempt_connect(direction, connections, is_checked, cube_pos, part_length, part_axis)

def attempt_connect(direction, connections, is_checked, cube_pos, part_length, part_axis):
    axis = 0
    step = 0
    if direction == 'up' or direction == 'down':
        axis = 2
    if direction == 'front' or direction == 'back':
        axis = 0
    if direction == 'right' or direction == 'left':
        axis = 1
    cc = connection_cords[direction](*cube_pos)
    if cc and (axis in part_axis or len(part_axis) < 2):
        neighbor_pos = point_cords[direction](*cube_pos)
        if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
            will_connect = False
            for _ in range(part_length.value):
                if np.random.rand() < often_connect:
                    will_connect = True
                    part_length.value += -1
                    break
            if will_connect:
                if axis not in part_axis:
                    part_axis.append(axis)
                connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

best_parts = None
best_sorted = None
best_reward = 0
cl = 3 # cube length: 4 -> 4x4x4 (need to edit this value in other file too)

repeats = 10
min_reward = 10
while repeats > 0 or best_reward < min_reward:
    repeats += -1
    connections = np.full((3, cl-1, cl, cl), False)

    reward = 0

    # make random connections
    is_checked = np.full((cl, cl, cl), False)

    for x in range(cl):
        for y in range(cl):
            for z in range(cl):
                if is_checked[x][y][z] == False:
                    is_checked[x][y][z] = True
                    part_axis = []
                    connect_cubes(connections, is_checked, [x,y,z], IntegerWrapper(part_length), part_axis)

    tetris_parts = merge_cubes(connections)
    tetris_parts_2d = convert_parts_2d(tetris_parts)

    sorted_parts = inspect_parts(tetris_parts_2d)
    for part in sorted_parts:
        # less repeast is more fun (total number of cubes is 64)
        # bigger part size is more fun (max size part 4x4=16)
        reward += part["size"]/part["repeats"]
        if part["size"] == 1:
            reward -= 20*part["repeats"]
        if part["size"] == 2:
            reward -= 10*part["repeats"]
        if part["size"] == 3:
            reward -= 5*part["repeats"]

    if reward > best_reward:
        best_parts = tetris_parts_2d
        best_sorted = sorted_parts
        best_reward = reward

print("best reward: " + str(best_reward))

# test for first shape (shapes are imported from shapes.py)
if solve_2d(heart1, best_sorted, [0,0]):
    print("and its solvable for the first shape!!")
else:
    print("not solvable for the first share, try to run again.")
    exit()

# test for second shape (shapes are imported from shapes.py)
for part in best_sorted: # convert best_sorted into 3d shapes
    part['matrix'] = np.expand_dims(part['matrix'], axis=2)
if solve_3d(chair, best_sorted, [0,0,0]):
    print("and its solvable for second shape too!!")
else:
    print("not solvable for the second shape, try to run again.")
    exit()

create_meshes(best_parts)
print("created final result at res.obj")

# to run it from bash until it generates succesfully:
# output="$(python _0_create_connection_manual.py)"; while echo "$output" | grep -q "not solvable"; do echo "$output"; sleep 0.2; output="$(python _0_create_connection_manual.py)"; done; echo "$output"