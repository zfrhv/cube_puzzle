import numpy as np

from _1_create_tetris_parts import merge_cubes,convert_parts_2d,connection_cords,at
from _2_verify_second_shape import inspect_parts
from _5_create_meshes import create_meshes

# Values from 0 to 1
fun_meter = 1.0 # the more varaity of shapes the more fun it is (many same shapes is boring)
_2d_difficulty_meter = 1.0 # the harder to assembly the shape 8x8 the harder it is
_3d_difficulty_meter = 1.0 # the harder to assembly the shape 4x4x4 the harder it is

# how often to have connections from 0 to 1 (1 = always make connections, 0 means never), more connections more longer parts
often_connect = 0.1
part_length = 12
def connect_cubes(connections, is_checked, cube_pos, part_length, part_axis):
        cc = connection_cords['up'](*cube_pos)
        if cc and (2 in part_axis or len(part_axis) < 2):
            neighbor_pos = [cube_pos[0], cube_pos[1], cube_pos[2]+1]
            if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
                will_connect = False
                for _ in range(part_length):
                    if np.random.rand() < often_connect:
                        will_connect = True
                        part_length += -1
                        break
                if will_connect:
                    if 2 not in part_axis:
                        part_axis.append(2)
                    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                    is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                    connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

        cc = connection_cords['down'](*cube_pos)
        if cc and (2 in part_axis or len(part_axis) < 2):
            neighbor_pos = [cube_pos[0], cube_pos[1], cube_pos[2]-1]
            if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
                will_connect = False
                for _ in range(part_length):
                    if np.random.rand() < often_connect:
                        will_connect = True
                        part_length += -1
                        break
                if will_connect:
                    if 2 not in part_axis:
                        part_axis.append(2)
                    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                    is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                    connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

        cc = connection_cords['front'](*cube_pos)
        if cc and (0 in part_axis or len(part_axis) < 2):
            neighbor_pos = [cube_pos[0]+1, cube_pos[1], cube_pos[2]]
            if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
                will_connect = False
                for _ in range(part_length):
                    if np.random.rand() < often_connect:
                        will_connect = True
                        part_length += -1
                        break
                if will_connect:
                    if 0 not in part_axis:
                        part_axis.append(0)
                    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                    is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                    connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

        cc = connection_cords['back'](*cube_pos)
        if cc and (0 in part_axis or len(part_axis) < 2):
            neighbor_pos = [cube_pos[0]-1, cube_pos[1], cube_pos[2]]
            if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
                will_connect = False
                for _ in range(part_length):
                    if np.random.rand() < often_connect:
                        will_connect = True
                        part_length += -1
                        break
                if will_connect:
                    if 0 not in part_axis:
                        part_axis.append(0)
                    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                    is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                    connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

        cc = connection_cords['right'](*cube_pos)
        if cc and (1 in part_axis or len(part_axis) < 2):
            neighbor_pos = [cube_pos[0], cube_pos[1]+1, cube_pos[2]]
            if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
                will_connect = False
                for _ in range(part_length):
                    if np.random.rand() < often_connect:
                        will_connect = True
                        part_length += -1
                        break
                if will_connect:
                    if 1 not in part_axis:
                        part_axis.append(1)
                    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                    is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                    connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

        cc = connection_cords['left'](*cube_pos)
        if cc and (1 in part_axis or len(part_axis) < 2):
            neighbor_pos = [cube_pos[0], cube_pos[1]-1, cube_pos[2]]
            if is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] == False:
                will_connect = False
                for _ in range(part_length):
                    if np.random.rand() < often_connect:
                        will_connect = True
                        part_length += -1
                        break
                if will_connect:
                    if 1 not in part_axis:
                        part_axis.append(1)
                    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
                    is_checked[neighbor_pos[0]][neighbor_pos[1]][neighbor_pos[2]] = True
                    connect_cubes(connections, is_checked, neighbor_pos, part_length, part_axis)

best_parts = None
best_reward = 0
for _ in range(10):
    connections = np.full((3, 3, 4, 4), False)

    reward = 0

    # make random connections
    is_checked = np.full((4, 4, 4), False)

    for x in range(4):
        for y in range(4):
            for z in range(4):
                if is_checked[x][y][z] == False:
                    is_checked[x][y][z] = True
                    part_axis = []
                    connect_cubes(connections, is_checked, [x,y,z], part_length, part_axis)
                    print(part_axis)

    tetris_parts = merge_cubes(connections)
    tetris_parts_2d = convert_parts_2d(tetris_parts)

    print(len(tetris_parts_2d), len(tetris_parts))

    sorted_parts = inspect_parts(tetris_parts_2d)
    for part in sorted_parts:
        # less repeast is more fun (total number of cubes is 64)
        # bigger part size is more fun (max size part 4x4=16)
        reward += part["size"]/part["repeats"]

    if reward > best_reward:
        best_parts = tetris_parts_2d
        best_reward = reward

    # print(sorted_parts)

create_meshes(best_parts)