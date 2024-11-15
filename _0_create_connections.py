from _1_create_tetris_parts import merge_cubes
from _2_verify_second_shape import inspect_parts
from _5_create_meshes import create_meshes

import torch
import torch.nn as nn
import torch.optim as optim

# Values from 0 to 1
fun_meter = 1.0 # the more varaity of shapes the more fun it is (many same shapes is boring)
_2d_difficulty_meter = 1.0 # the harder to assembly the shape 8x8 the harder it is
_3d_difficulty_meter = 1.0 # the harder to assembly the shape 4x4x4 the harder it is

# Connect the cubes

class PiecesGenerator(nn.Module):
    def __init__(self):
        super(PiecesGenerator, self).__init__()
        self.fc1 = nn.Linear(3, 144)
        self.fc2 = nn.Linear(144, 144)
        self.fc3 = nn.Linear(144, 144)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Sigmoid to get output between 0 and 1 for connections

# Initialize the model, optimizer, and latent space
pieces_generator = PiecesGenerator()
optimizer = optim.Adam(pieces_generator.parameters(), lr=0.001)

tetris_parts = None

last_feedback = 0


# Training loop
for iteration in range(1000):
    result_connections = pieces_generator(torch.tensor([fun_meter, _2d_difficulty_meter, _3d_difficulty_meter]))

    feedback = 1.0

    tetris_parts = merge_cubes(result_connections.view(3, 3, 4, 4) >= 0.5)

    # TODO add memory to roll back in case went bad

    # if int then not all parts successful
    if isinstance(tetris_parts, int):
        feedback = tetris_parts/64
        print(tetris_parts)
        print("0", feedback)
    else:
        sorted_parts = inspect_parts(tetris_parts)
        for part in sorted_parts:
            # less repeast is more fun (total number of cubes is 64)
            # bigger part size is more fun (max size part 4x4=16)
            feedback = feedback * (1 - part["repeats"]/64) * part["size"] / 16
        # give it the minimum
        feedback = feedback + 0.1 * (1 - feedback)
        print("1", feedback)

    # print(result_connections)

    # Calculate new random target
    new_random_target = torch.rand_like(result_connections)
    new_random_target = result_connections * feedback + new_random_target * (1 - feedback) # the worse the more significant move
    loss = nn.BCEWithLogitsLoss()(result_connections, new_random_target)

    # Backpropagation and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # optimizer.zero_grad()
    # if feedback >= last_feedback:
    #     loss.backward()
    #     optimizer.step()
    #     # reset_optimizer_state(optimizer)
    # last_feedback = feedback

if tetris_parts:
    create_meshes(tetris_parts)
    print("finished meshing all parts")
else:
    print("the parts are not good enough to mesh them")