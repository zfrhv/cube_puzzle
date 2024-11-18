from collections import Counter
import numpy as np

# Check if its good parts for 2d (8x8)

# TODO check this. or even better then from the start when you mark some connection then disable marking others.
# its not super efficient but at least no need to rerun 1000 times

def inspect_parts(parts):
    counter = Counter([tuple(map(tuple, part)) for part in parts])

    return [
        {
            "repeats": count,
            "matrix": np.array(part),
            "size": np.count_nonzero(np.array(part))
        }
        for part, count in counter.items()
    ]


def solve_8x8(plane, parts, pos):
    next_pos = pos.copy()
    next_pos[0] += 1
    if next_pos[0] >= plane.shape[0]:
        next_pos[0] = 0
        next_pos[1] += 1
    if next_pos[1] >= plane.shape[1]:
        return True

    if plane[pos[0]][pos[1]] == 1: # current pos already filled
        # solve next
        return solve_8x8(plane, parts, next_pos)
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
                                combo_result = part_shape + plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]]
                                if np.max(combo_result) < 2:
                                    plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]] += part_shape
                                    succeeded = solve_8x8(plane, parts, next_pos)
                                    if succeeded:
                                        # put back the part and quit
                                        part['repeats'] += 1
                                        if part['repeats'] == 1:
                                            parts.append(part)
                                        return True
                                    else:
                                        plane[shifted_pos[0] : shifted_pos[0] + part_shape.shape[0], shifted_pos[1] : shifted_pos[1] + part_shape.shape[1]] -= part_shape
                    part_shape = np.rot90(part_shape)
                part_shape = part_shape[::-1, :]

            # put back the part
            part['repeats'] += 1
            if part['repeats'] == 1:
                parts.append(part)
    return False