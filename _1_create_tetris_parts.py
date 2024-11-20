import numpy as np

# Init connections and cubes

def at(obj, index_arr):
    for key in index_arr:
        obj = obj[key]
    return obj

cl = 3

point_cords = {
    'up':    lambda x,y,z: None if z >= cl-1 else [x, y, z+1],
    'down':  lambda x,y,z: None if z <= 0    else [x, y, z-1],
    'front': lambda x,y,z: None if x >= cl-1 else [x+1, y, z],
    'back':  lambda x,y,z: None if x <= 0    else [x-1, y, z],
    'right': lambda x,y,z: None if y >= cl-1 else [x, y+1, z],
    'left':  lambda x,y,z: None if y <= 0    else [x, y-1, z],
}

# connections_x = 3d array of 3 x planes, 4 rows, 4 connection (val >= 0 yes connected otherwise not connected)
# connections_y and connections_z are the same.
connection_cords = {
    'up':    lambda x,y,z: None if z >= cl-1 else [2, z, x, y],
    'down':  lambda x,y,z: None if z <= 0    else [2, z-1, x, y],
    'front': lambda x,y,z: None if x >= cl-1 else [0, x, y, z],
    'back':  lambda x,y,z: None if x <= 0    else [0, x-1, y, z],
    'right': lambda x,y,z: None if y >= cl-1 else [1, y, z, x],
    'left':  lambda x,y,z: None if y <= 0    else [1, y-1, z, x],
}

# Separate 3d cube into many 3d parts

def merge_next_cubes(connections, is_checked, part, part_pos, cube_pos, part_axis):
    if not is_checked[cube_pos[0]][cube_pos[1]][cube_pos[2]]:
        is_checked[cube_pos[0]][cube_pos[1]][cube_pos[2]] = True
        part[part_pos[0]][part_pos[1]][part_pos[2]] = True

        cc = connection_cords['up'](*cube_pos)
        if cc and at(connections, cc):
            if 2 not in part_axis:
                part_axis.append(2)
            neighbor_pos = point_cords['up'](*cube_pos)
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(2)] += 1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['down'](*cube_pos)
        if cc and at(connections, cc):
            if 2 not in part_axis:
                part_axis.append(2)
            neighbor_pos = point_cords['down'](*cube_pos)
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(2)] += -1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['front'](*cube_pos)
        if cc and at(connections, cc):
            if 0 not in part_axis:
                part_axis.append(0)
            neighbor_pos = point_cords['front'](*cube_pos)
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(0)] += 1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['back'](*cube_pos)
        if cc and at(connections, cc):
            if 0 not in part_axis:
                part_axis.append(0)
            neighbor_pos = point_cords['back'](*cube_pos)
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(0)] += -1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['right'](*cube_pos)
        if cc and at(connections, cc):
            if 1 not in part_axis:
                part_axis.append(1)
            neighbor_pos = point_cords['right'](*cube_pos)
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(1)] += 1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

        cc = connection_cords['left'](*cube_pos)
        if cc and at(connections, cc):
            if 1 not in part_axis:
                part_axis.append(1)
            neighbor_pos = point_cords['left'](*cube_pos)
            new_pos = [part_pos[0], part_pos[1], part_pos[2]]
            new_pos[part_axis.index(1)] += -1
            part, part_axis = merge_next_cubes(connections, is_checked, part, new_pos, neighbor_pos, part_axis)

    return [part, part_axis]

# connections arrays ([axis][cl-1][cl][cl])
def merge_cubes(connections):
    is_checked = np.full((cl, cl, cl), False)

    parts = []
    for x in range(cl):
        for y in range(cl):
            for z in range(cl):
                if not is_checked[x][y][z]:
                    new_part, _ = merge_next_cubes(connections, is_checked, np.full(((cl-1)*2+1, (cl-1)*2+1, (cl-1)*2+1), False), [cl-1,cl-1,cl-1], [x,y,z], [])

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