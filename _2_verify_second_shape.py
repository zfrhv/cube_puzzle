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
            "part": np.array(part),
            "size": np.count_nonzero(np.array(part))
        }
        for part, count in counter.items()
    ]