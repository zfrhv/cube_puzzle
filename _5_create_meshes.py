import trimesh

def create_meshes(tetris_parts):
    # Create meshes
    mesh_parts = []
    for index, part in enumerate(tetris_parts):
        mesh_part = None
        for x in range(len(part)):
            for y in range(len(part[x])):
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
    scene.export('res.obj')