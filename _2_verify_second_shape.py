from collections import Counter
import numpy as np

# Check if its good parts for 2d (8x8)

# TODO check this. or even better then from the start when you mark some connection then disable marking others.
# its not super efficient but at least no need to rerun 1000 times

def inspect_parts(parts):
    sorted_parts = []
    for part in parts:
        found_match = False
        for sorted_part in sorted_parts:
            if compare_parts(sorted_part['matrix'], part):
                sorted_part['repeats'] += 1
                found_match = True
                break
        if not found_match:
            sorted_parts.append({
                "repeats": 1,
                "matrix": part,
                "size": np.count_nonzero(part)
            })

    return sorted_parts

def compare_parts(part1, part2):
    # flip the part
    for _ in range(2):
        # rotate the part
        for _ in range(4):
            if np.array_equal(part1, part2):
                return True
            part2 = np.rot90(part2)
        part2 = part2[::-1, :]
    return False

# TODO use more numpy
def solve_2d(plane, parts, pos, colorful_plane=None):
    next_pos_valid = False
    next_pos = pos.copy()
    while not next_pos_valid:
        next_pos[0] += 1
        if next_pos[0] >= plane.shape[0]:
            next_pos[0] = 0
            next_pos[1] += 1
        if next_pos[1] >= plane.shape[1]:
            return True
        if plane[next_pos[0]][next_pos[1]] == 1:
            next_pos_valid = True

    if plane[pos[0]][pos[1]] == 3: # if current pos already solved
        # solve next
        return solve_2d(plane, parts, next_pos, colorful_plane)
    else:
        for part in parts:
            # remove the part from parts
            part['repeats'] += -1
            if part['repeats'] == 0:
                parts.remove(part)

            part_shape = part['matrix']
            # flip the shape (2 times)
            for _ in range(2):
                # rotate the shape (4 times)
                for _ in range(4):
                    # try shape in all positions
                    for x_shift in range(part_shape.shape[0]):
                        for y_shift in range(part_shape.shape[1]):
                            shifted_pos = np.add(pos, [-x_shift, -y_shift])
                            # if piece in plane bounds
                            if shifted_pos[0] >= 0 and shifted_pos[1] >= 0 and shifted_pos[0] + part_shape.shape[0] <= plane.shape[0] and shifted_pos[1] + part_shape.shape[1] <= plane.shape[1]:
                                combo_result = 2*part_shape + plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]]
                                # 0: not allowed to place
                                # 1: allowed to place
                                # 2: putted part on not allowed place
                                # 3: putted on allowed place
                                # 5: putted on already occupied place
                                if not np.any(np.isin([2,5], combo_result)): # if no bad parts after combining
                                    plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]] += 2*part_shape
                                    succeeded = solve_2d(plane, parts, next_pos, colorful_plane)
                                    if succeeded:
                                        # put back the part and quit
                                        part['repeats'] += 1
                                        if part['repeats'] == 1:
                                            parts.append(part)
                                        # print to the solution and quit
                                        if colorful_plane is not None:
                                            last_index = np.max(colorful_plane)
                                            colorful_plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]] += (last_index+1)*part_shape
                                        return True
                                    else:
                                        plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]] -= 2*part_shape
                    part_shape = np.rot90(part_shape)
                part_shape = part_shape[::-1, :]

            # put back the part
            part['repeats'] += 1
            if part['repeats'] == 1:
                parts.append(part)
    return False



def solve_3d(field, parts, pos):
    next_pos_valid = False
    next_pos = pos.copy()
    while not next_pos_valid:
        next_pos[0] += 1
        if next_pos[0] >= field.shape[0]:
            next_pos[0] = 0
            next_pos[1] += 1
        if next_pos[1] >= field.shape[1]:
            next_pos[1] = 0
            next_pos[2] += 1
        if next_pos[2] >= field.shape[2]:
            return True
        if field[next_pos[0]][next_pos[1]][next_pos[2]] == 1:
            next_pos_valid = True

    if field[pos[0]][pos[1]][pos[2]] == 3: # if current pos already solved
        # solve next
        return solve_3d(field, parts, next_pos)
    else:
        for part in parts:
            # remove the part from parts
            part['repeats'] += -1
            if part['repeats'] == 0:
                parts.remove(part)

            part_shape = part['matrix']
            # try in all 3 axis
            for _ in range(3):
                # flip the shape (2 times)
                for _ in range(2):
                    # rotate the shape (4 times)
                    for _ in range(4):
                        # try shape in all positions
                        for x_shift in range(part_shape.shape[0]):
                            for y_shift in range(part_shape.shape[1]):
                                for z_shift in range(part_shape.shape[2]):
                                    shifted_pos = np.add(pos, [-x_shift, -y_shift, -z_shift])
                                    # if piece in field bounds
                                    if (
                                        shifted_pos[0] >= 0 and
                                        shifted_pos[1] >= 0 and
                                        shifted_pos[2] >= 0 and
                                        shifted_pos[0] + part_shape.shape[0] <= field.shape[0] and
                                        shifted_pos[1] + part_shape.shape[1] <= field.shape[1] and
                                        shifted_pos[2] + part_shape.shape[2] <= field.shape[2]
                                        ):
                                        combo_result = 2*part_shape + field[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1], shifted_pos[2] : shifted_pos[2] + part_shape.shape[2]]
                                        # 0: not allowed to place
                                        # 1: allowed to place
                                        # 2: putted part on not allowed place
                                        # 3: putted on allowed place
                                        # 5: putted on already occupied place
                                        if not np.any(np.isin([2,5], combo_result)): # if no bad parts after combining
                                            field[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1], shifted_pos[2] : shifted_pos[2] + part_shape.shape[2]] += 2*part_shape
                                            succeeded = solve_3d(field, parts, next_pos)
                                            if succeeded:
                                                # put back the part and quit
                                                part['repeats'] += 1
                                                if part['repeats'] == 1:
                                                    parts.append(part)
                                                return True
                                            else:
                                                field[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1], shifted_pos[2] : shifted_pos[2] + part_shape.shape[2]] -= 2*part_shape
                        part_shape = np.rot90(part_shape, k=1, axes=(0, 1))
                    part_shape = np.rot90(part_shape, k=2, axes=(0, 2))
                part_shape = np.transpose(part_shape, (2, 0, 1))

            # put back the part
            part['repeats'] += 1
            if part['repeats'] == 1:
                parts.append(part)
    return False