# Cube 4x4x4 (connections between parts 3x3x3)
import trimesh
import numpy as np
import random

fun_meter = 0 # the more varaity of shapes the more fun it is (many same shapes is boring)
_2d_difficulty_meter = 0 # the harder to assembly the shape 8x8 the harder it is
_3d_difficulty_meter = 0 # the harder to assembly the shape 4x4x4 the harder it is

# Init connections and cubes

def at(obj, index_arr):
  for key in index_arr:
    obj = obj[key]
  return obj

def create_axis_connection():
  axis_connections = []
  for x in range(3):
    axis_connections.append([])
    for y in range(4):
      axis_connections[x].append([])
      for z in range(4):
        axis_connections[x][y].append(False)
  return axis_connections
connections = [
  create_axis_connection(), # x axis
  create_axis_connection(), # y axis
  create_axis_connection(), # z axis
]

connection_cords = {
  'up':    lambda x,y,z: None if z >= 3 else [2, z, x, y],
  'down':  lambda x,y,z: None if z <= 0 else [2, z-1, x, y],
  'front': lambda x,y,z: None if x >= 3 else [0, x, y, z],
  'back':  lambda x,y,z: None if x <= 0 else [0, x-1, y, z],
  'right': lambda x,y,z: None if y >= 3 else [1, y, z, x],
  'left':  lambda x,y,z: None if y <= 0 else [1, y-1, z, x],
}

is_checked = []
for x in range(4):
  is_checked.append([])
  for y in range(4):
    is_checked[x].append([])
    for z in range(4):
      is_checked[x][y].append(False)

# Connect the cubes

for _ in range(20):
  # cc = connection_cords['up'](0,0,0)
  # connections[cc[0]][cc[1]][cc[2]][cc[3]] = True
  cc = list(connection_cords.values())[random.randint(0, 5)](random.randint(0, 3),random.randint(0, 3),random.randint(0, 3))
  if cc:
    connections[cc[0]][cc[1]][cc[2]][cc[3]] = True

# TODO maybe use AI instead of random

# Separate 4x4x4 into many 2d parts

def merge_cubes(part, part_pos, cube_pos, part_axis):
  if not is_checked[cube_pos[0]][cube_pos[1]][cube_pos[2]]:
    is_checked[cube_pos[0]][cube_pos[1]][cube_pos[2]] = True
    part[part_pos[0]][part_pos[1]] = True

    cc = connection_cords['up'](*cube_pos)
    if cc and at(connections, cc):
      if 2 not in part_axis:
        part_axis.append(2)
        if len(part_axis) == 3:
          raise Exception("one of the parts is in 3d, its bad for 8x8. exiting...")
      neighbor_pos = [cube_pos[0], cube_pos[1], cube_pos[2]+1]
      if part_axis.index(2) == 0:
        new_pos = [part_pos[0]+1, part_pos[1]]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      elif part_axis.index(2) == 1:
        new_pos = [part_pos[0], part_pos[1]+1]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      else:
        raise Exception("error: not valid inxed :/")

    cc = connection_cords['down'](*cube_pos)
    if cc and at(connections, cc):
      if 2 not in part_axis:
        part_axis.append(2)
        if len(part_axis) == 3:
          raise Exception("one of the parts is in 3d, its bad for 8x8. exiting...")
      neighbor_pos = [cube_pos[0], cube_pos[1], cube_pos[2]-1]
      if part_axis.index(2) == 0:
        new_pos = [part_pos[0]-1, part_pos[1]]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      elif part_axis.index(2) == 1:
        new_pos = [part_pos[0], part_pos[1]-1]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      else:
        raise Exception("error: not valid inxed :/")

    cc = connection_cords['front'](*cube_pos)
    if cc and at(connections, cc):
      if 0 not in part_axis:
        part_axis.append(0)
        if len(part_axis) == 3:
          raise Exception("one of the parts is in 3d, its bad for 8x8. exiting...")
      neighbor_pos = [cube_pos[0]+1, cube_pos[1], cube_pos[2]]
      if part_axis.index(0) == 0:
        new_pos = [part_pos[0]+1, part_pos[1]]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      elif part_axis.index(0) == 1:
        new_pos = [part_pos[0], part_pos[1]+1]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      else:
        raise Exception("error: not valid inxed :/")

    cc = connection_cords['back'](*cube_pos)
    if cc and at(connections, cc):
      if 0 not in part_axis:
        part_axis.append(0)
        if len(part_axis) == 3:
          raise Exception("one of the parts is in 3d, its bad for 8x8. exiting...")
      neighbor_pos = [cube_pos[0]-1, cube_pos[1], cube_pos[2]]
      if part_axis.index(0) == 0:
        new_pos = [part_pos[0]-1, part_pos[1]]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      elif part_axis.index(0) == 1:
        new_pos = [part_pos[0], part_pos[1]-1]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      else:
        raise Exception("error: not valid inxed :/")

    cc = connection_cords['right'](*cube_pos)
    if cc and at(connections, cc):
      if 1 not in part_axis:
        part_axis.append(1)
        if len(part_axis) == 3:
          raise Exception("one of the parts is in 3d, its bad for 8x8. exiting...")
      neighbor_pos = [cube_pos[0], cube_pos[1]+1, cube_pos[2]]
      if part_axis.index(1) == 0:
        new_pos = [part_pos[0]+1, part_pos[1]]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      elif part_axis.index(1) == 1:
        new_pos = [part_pos[0], part_pos[1]+1]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      else:
        raise Exception("error: not valid inxed :/")

    cc = connection_cords['left'](*cube_pos)
    if cc and at(connections, cc):
      if 1 not in part_axis:
        part_axis.append(1)
        if len(part_axis) == 3:
          raise Exception("one of the parts is in 3d, its bad for 8x8. exiting...")
      neighbor_pos = [cube_pos[0], cube_pos[1]-1, cube_pos[2]]
      if part_axis.index(1) == 0:
        new_pos = [part_pos[0]-1, part_pos[1]]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      elif part_axis.index(1) == 1:
        new_pos = [part_pos[0], part_pos[1]-1]
        part, part_axis = merge_cubes(part, new_pos, neighbor_pos, part_axis)
      else:
        raise Exception("error: not valid inxed :/")

  return [part, part_axis]

parts = []
for x in range(4):
  for y in range(4):
    for z in range(4):
      if not is_checked[x][y][z]:
        new_part, _ = merge_cubes(np.full((7, 7), False), [3,3], [x,y,z], [])

        # clean the part
        repeat = True
        while repeat:
          repeat = False
          if all(val == False for val in new_part[0]):
            new_part = new_part[1:]
            repeat = True

        repeat = True
        while repeat:
          repeat = False
          if all(row[0] == False for row in new_part):
            new_part = new_part[:, 1:]
            repeat = True

        clean_part = np.full((4, 4), False)
        for index_x in range(min(len(new_part),4)):
          for index_y in range(min(len(new_part[index_x]),4)):
            clean_part[index_x][index_y] = new_part[index_x][index_y]
        
        parts.append(clean_part)

# Check if its good parts for 2d (8x8)

# TODO check this. or even better then from the start when you mark some connection then disable marking others.
# its not super efficient but at least no need to rerun 1000 times


# Create meshes
mesh_parts = []
for index, part in enumerate(parts):
  mesh_part = None
  for x in range(4):
    for y in range(4):
      if part[x][y]:
        if not mesh_part:
          mesh_part = trimesh.creation.box(extents=[1, 1, 1])
          mesh_part.apply_translation([x, y, 0])
        else:
          new_part = trimesh.creation.box(extents=[1, 1, 1])
          new_part.apply_translation([x, y, 0])
          mesh_part = trimesh.util.concatenate(mesh_part, new_part)
  mesh_part.apply_translation([0, 0, index*1.2])
  mesh_parts.append(mesh_part)

# Output the result

scene = trimesh.Scene(mesh_parts)
scene.export('4x4x4_cube.obj')

# TODO accept 3d parts as well and then not only i can make square but also other shapes like a cat or hat...