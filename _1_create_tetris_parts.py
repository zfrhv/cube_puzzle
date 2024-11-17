import numpy as np

# Init connections and cubes

def at(obj, index_arr):
    for key in index_arr:
        obj = obj[key]
    return obj

# connections_x = 3d array of 3 x planes, 4 rows, 4 connection (val >= 0 yes connected otherwise not connected)
# connections_y and connections_z are the same.
connection_cords = {
    'up':        lambda x,y,z: None if z >= 3 else [2, z, x, y],
    'down':    lambda x,y,z: None if z <= 0 else [2, z-1, x, y],
    'front': lambda x,y,z: None if x >= 3 else [0, x, y, z],
    'back':    lambda x,y,z: None if x <= 0 else [0, x-1, y, z],
    'right': lambda x,y,z: None if y >= 3 else [1, y, z, x],
    'left':    lambda x,y,z: None if y <= 0 else [1, y-1, z, x],
}

# TODO accept 3d parts as well and then not only i can make square but also other shapes like a pyramid, castle, or other shape that will hold itself...


# Separate 4x4x4 into many 3d parts

def merge_next_cubes(connections, is_checked, part, part_pos, cube_pos, part_axis):
    if not is_checked[cube_pos[0]][cube_pos[1]][cube_pos[2]]:
        is_checked[cube_pos[0]][cube_pos[1]][cube_pos[2]] = True
        part[part_pos[0]][part_pos[1]][part_pos[2]] = True

        cc = connection_cords['up'](*cube_pos)
        if cc and at(connections, cc):
            if 2 not in part_axis:
                part_axis.append(2)
            neighbor_pos = [cube_pos[0], cube_pos[1], cube_pos[2]+1]
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(2)] += 1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['down'](*cube_pos)
        if cc and at(connections, cc):
            if 2 not in part_axis:
                part_axis.append(2)
            neighbor_pos = [cube_pos[0], cube_pos[1], cube_pos[2]-1]
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(2)] += -1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['front'](*cube_pos)
        if cc and at(connections, cc):
            if 0 not in part_axis:
                part_axis.append(0)
            neighbor_pos = [cube_pos[0]+1, cube_pos[1], cube_pos[2]]
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(0)] += 1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['back'](*cube_pos)
        if cc and at(connections, cc):
            if 0 not in part_axis:
                part_axis.append(0)
            neighbor_pos = [cube_pos[0]-1, cube_pos[1], cube_pos[2]]
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(0)] += -1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['right'](*cube_pos)
        if cc and at(connections, cc):
            if 1 not in part_axis:
                part_axis.append(1)
            neighbor_pos = [cube_pos[0], cube_pos[1]+1, cube_pos[2]]
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(1)] += 1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['left'](*cube_pos)
        if cc and at(connections, cc):
            if 1 not in part_axis:
                part_axis.append(1)
            neighbor_pos = [cube_pos[0], cube_pos[1]-1, cube_pos[2]]
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(1)] += -1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

    return [part, part_axis]

# connections 3x3x4x4 arrays
def merge_cubes(connections):
    is_checked = np.full((4, 4, 4), False)

    parts = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if not is_checked[x][y][z]:
                    new_part, _ = merge_next_cubes(connections, is_checked, np.full((7, 7, 7), False), [3,3,3], [x,y,z], [])

                    # clean the part
                    repeat = True
                    while repeat:
                        repeat = False
                        if np.all(new_part[0, :, :] == False):
                            new_part = np.delete(new_part, 0, axis=0)
                            repeat = True
                        if np.all(new_part[-1, :, :] == False):
                            new_part = np.delete(new_part, -1, axis=0)
                            repeat = True
                        if np.all(new_part[:, 0, :] == False):
                            new_part = np.delete(new_part, 0, axis=1)
                            repeat = True
                        if np.all(new_part[:, -1, :] == False):
                            new_part = np.delete(new_part, -1, axis=1)
                            repeat = True
                        if np.all(new_part[:, :, 0] == False):
                            new_part = np.delete(new_part, 0, axis=2)
                            repeat = True
                        if np.all(new_part[:, :, -1] == False):
                            new_part = np.delete(new_part, -1, axis=2)
                            repeat = True
                    
                    parts.append(new_part)
    return parts

def convert_parts_2d(parts):
    new_parts = []
    for part in parts:
        # Check along the x-axis
        planes_with_values = 0
        last_plane_index = 0
        for i in range(part.shape[0]):
            if np.any(part[i]):
                planes_with_values += 1
        if planes_with_values <= 1:
            new_parts.append(part[last_plane_index])
            continue

        # Check along the y-axis
        planes_with_values = 0
        last_plane_index = 0
        for i in range(part.shape[1]):
            if np.any(part[:, i]):
                planes_with_values += 1
        if planes_with_values <= 1:
            new_parts.append(part[:, last_plane_index])
            continue

        # Check along the z-axis
        planes_with_values = 0
        last_plane_index = 0
        for i in range(part.shape[2]):
            if np.any(part[:, :, i]):
                planes_with_values += 1
        if planes_with_values <= 1:
            new_parts.append(part[:, :, last_plane_index])
            continue

    return new_parts