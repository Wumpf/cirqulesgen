from time import perf_counter
from PIL import Image
import numpy as np
from numpy.random import default_rng
import time

t0= time.time()
# new white image
w, h = 1024, 768
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:w, 0:h] = [255, 255, 255]

pixel_coordinates = np.stack([np.tile(np.arange(w), h), np.arange(h).repeat(w)])

# generate random circles
rng = default_rng(28346)
circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
circles_position = rng.uniform(size=(circle_count, 2)) * [w, h]

# determine area indices
pixel_coordinates_chunk_broadcast = np.broadcast_to(pixel_coordinates[None, :, :], (64,) + pixel_coordinates.shape)
perpix_area_index = np.zeros(pixel_coordinates.shape[1], dtype=np.int64)
byte_factors = 2 ** np.arange(64, dtype=np.int64)
for chunk_start in range(0, circle_count, 64):
    chunk_end = min(chunk_start + 64, circle_count)
    chunk = range(chunk_start, chunk_end)
    dist_circles_sq = np.sum((pixel_coordinates_chunk_broadcast[:len(chunk), :, :] - circles_position[chunk, :, None]) ** 2, axis=1)
    perpix_circle_hits = dist_circles_sq < circles_radius_sq[chunk, None]
    perpix_area_index ^= (perpix_circle_hits * byte_factors[:len(chunk), None]).sum(axis=0)
print("Time elapsed: ", time.time() - t0)

# determine the color for each area
unique_area_indices = np.unique(perpix_area_index)
area_colors = rng.integers(low=0, high=255, dtype=np.uint8, size=(unique_area_indices.size, 3))
area_colors[np.searchsorted(unique_area_indices, 0)] = [255, 255, 255] # we get overflow, causing our sorted area indices not to start with 0

# assign every pixel an index into the color array
pix_areas = perpix_area_index[:, None] == unique_area_indices
perpix_color_index = np.nonzero(pix_areas)[1]
data = np.reshape(area_colors[perpix_color_index], (h, w, 3))

print("Time elapsed: ", time.time() - t0)

img = Image.fromarray(data, 'RGB')
img.show()