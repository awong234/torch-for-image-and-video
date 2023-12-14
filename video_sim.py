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
plt.imshow(Z, origin='lower')
plt.show()

img = Image.fromarray(Z*255)
img = img.convert('L')
img.save('test.png')

blob = Blob(
    starting_position={"x": 0, "y": 0},
    starting_vector={"direction": 45, "speed": 5},
    radius = 15,
    max_p = 0.3
)
claims = np.array([0 for x in range(0, RES**2)]).reshape(RES**2, 1)
for i in range(0, 30):
    Z_flat = gaussian_decay(blob.max_p, blob.radius, grid, blob.current_x, blob.current_y)
    claims += binomial(n=1, p=Z_flat)
    Z = Z_flat.reshape(128, 128)
    img = Image.fromarray(Z*255)
    img = img.convert('L')
    # img.save(f'frame_{i}.png')
    blob.advance()

plt.imshow(claims.reshape(RES, RES), origin='lower')
plt.show()
