# Create a video of a blob moving across the space.
# Small dimension, keep it 128 x 128 (this time 2**7 so that we don't have to fiddle with the stride values)
# Let's say, 30 frames of motion.

import math
import numpy as np
from numpy.random import binomial
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import os

RES = 128

image_dir = 'simvid'

if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

class Blob:
    def __init__(self, starting_position, starting_vector, radius, max_p):
        self.start_x = starting_position["x"]
        self.start_y = starting_position["y"]
        self.direction = starting_vector["direction"]
        self.dir_rad = self.direction * (math.pi / 180)
        self.speed = starting_vector["speed"]  # Speed in units of pixels per frame
        self.current_x = self.start_x
        self.current_y = self.start_y
        self.radius = radius
        self.max_p = max_p
    def __repr__(self):
        return f"Start: {self.start_x}, {self.start_y}\nDirection: {self.direction}\nSpeed: {self.speed}\nCurrently at: {self.current_x}, {self.current_y}"
    def advance(self):
        delta_y = math.sin(self.dir_rad) * self.speed
        delta_x = math.cos(self.dir_rad) * self.speed
        self.current_x = self.current_x + delta_x
        self.current_y = self.current_y + delta_y
    def distance(self):
        return math.sqrt(
            (self.current_x - self.start_x) ** 2 + (self.current_y - self.start_y) ** 2
        )


blob = Blob(
    starting_position={"x": 0, "y": 0},
    starting_vector={"direction": 45, "speed": 4},
    radius=0,
    max_p = 0.8
)
for i in range(0, 30):
    print(f"Frame {i+1}")
    blob.advance()
    blob
    print("")


def gaussian_decay(intensity, radius, grid, pos_x, pos_y):
    X = grid[0]
    Y = grid[1]
    point = np.array([pos_x, pos_y])
    dist = cdist(grid, point.reshape(1, -1))
    num = intensity * np.exp(-(dist**2) / (2 * (radius**2)))
    return num


# make data
X, Y = np.meshgrid(np.linspace(0, RES, RES), np.linspace(0, RES, RES))
grid = np.column_stack((X.ravel(), Y.ravel()))
Z_flat = gaussian_decay(0.8, 25, grid, 0, 0)
Z = Z_flat.reshape(128, 128)

# plot
# plt.imshow(Z, origin='lower')
# plt.show()

# Pick random directions, speeds, and starting positions
def sim_blob(NSIM):
    directions = np.random.uniform(-360, 360, NSIM)
    speeds     = np.random.uniform(0, 5, NSIM)
    starting_x = np.random.randint(0, RES, NSIM)
    starting_y = np.random.randint(0, RES, NSIM)
    radii      = np.random.uniform(0, 20, NSIM)
    maxp       = np.random.uniform(0.2, 1.0, NSIM)
    for i in range(0, NSIM):
        blob = Blob(
            starting_position={"x": starting_x[i], "y": starting_y[i]},
            starting_vector={"direction": directions[i], "speed": speeds[i]},
            radius = radii[i],
            max_p = maxp[i]
        )
        # Create output folder for blob images
        name_part_direction = '%.1f' % directions[i]
        name_part_speed = '%.1f' % speeds[i]
        name_part_starting_x = '%.1f' % starting_x[i]
        name_part_starting_y = '%.1f' % starting_y[i]
        name_part_radius = '%.1f' % radii[i]
        name_part_maxp = '%.1f' % maxp[i]
        out_folder = os.path.join(image_dir, f'blob_{i}_direction{name_part_direction}_speed{name_part_speed}_x{name_part_starting_x}_y{name_part_starting_y}_radius{name_part_radius}_maxp{name_part_maxp}')
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        claims = np.array([0 for x in range(0, RES**2)]).reshape(RES**2, 1)
        for frame in range(0, 30):
            Z_flat = gaussian_decay(blob.max_p, blob.radius, grid, blob.current_x, blob.current_y)
            claims += binomial(n=1, p=Z_flat)
            Z = Z_flat.reshape(RES, RES)
            # Save blob image
            img = Image.fromarray(Z*255)
            img = img.convert('L')
            img.save(os.path.join(out_folder, f'blob_frame_{frame}.png'))
            # Save claims image
            max_claims = np.max(claims)
            claims_for_image = (claims.reshape(RES, RES) / max_claims) * 255
            img = Image.fromarray(claims_for_image)
            img = img.convert('L')
            img.save(os.path.join(out_folder, f'claims_viz_{frame}.png'))
            # Save claims _data_ -- this can be kept raw
            img = Image.fromarray(claims.reshape(RES, RES))
            img = img.convert('L')
            img.save(os.path.join(out_folder, f'claims_data_{frame}.png'))
            blob.advance()


np.random.seed(123)
sim_blob(1000)
