import numpy as np

from _1_create_tetris_parts import merge_cubes,convert_parts_2d,connection_cords,point_cords,at
from _2_verify_second_shape import inspect_parts,solve_2d
from _5_create_meshes import create_meshes

# possible shapes:
# cube
# pyramid
# stairs (Penrose stairs?)
# 2d flat shapes (anything):
#   cat
#   fish
#   bird
#   Heart
#   star
#   diamond
#   house
#   key
#   shiba?

# 4x4x4 cube (sum 64)
# 3x3x3 cube (sum 27)
# pyramid 4/5 layers (sum 35/84)
# heart 27/46/63/70
heart1 = np.array([
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])
heart3 = np.array([
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
])
# shiba (sum 59, or 61 if colored eyes instead of empty, maybe can add colors and get to 64):
shiba = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
])

def print_2d_shape(shape):
    for row in shape:
        for cell in row:
            print("⬛" if cell == 0 else "⬜" , end="")
        print()

# print_2d_shape(heart1)
# exit()

# Values from 0 to 1
fun_meter = 1.0 # the more varaity of shapes the more fun it is (many same shapes is boring)
_2d_difficulty_meter = 1.0 # the harder to assembly the shape 8x8 the harder it is
_3d_difficulty_meter = 1.0 # the harder to assembly the shape 4x4x4 the harder it is

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
cl = 3 # cube length: 4 -> 4x4x4
for _ in range(1000):
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
# if solve_2d(np.ones((8, 8)), best_sorted, [0,0]):
if solve_2d(heart1, best_sorted, [0,0]):
    print("and its solvable!!")
else:
    print("nah")
create_meshes(best_parts)