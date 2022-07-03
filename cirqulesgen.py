from time import perf_counter
from PIL import Image
import numpy as np
from numpy.random import default_rng

# new white image
w, h = 1024, 768
# data = np.zeros((h, w, 3), dtype=np.uint8)
# data[0:w, 0:h] = [255, 255, 255]

comps_x = np.tile(np.arange(w), h)
comps_y = np.arange(h).repeat(w)

# draw random circles
rng = default_rng(28346)
circle_count = 64
radius_range = (40, 150)
circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
circles_position = rng.uniform(size=(2, circle_count)) * [[w], [h]]

dist_circles_sq = ((np.broadcast_to(comps_x[None, :], (circle_count,) + comps_x.shape) - circles_position[0, :, None]) ** 2 + 
                   (np.broadcast_to(comps_y[None, :], (circle_count,) + comps_y.shape) - circles_position[1, :, None]) ** 2)
# boolean matrix - true for each pixel (each colum is a different pixel) if it is contained in a circle (each row for a different circle)
circle_pix = dist_circles_sq < circles_radius_sq[:, None]

# determine area index for each pixel. no time for proper hashing!
perpix_area_index = np.packbits(circle_pix, axis=0)
byte_factors = 256 ** np.arange(8, dtype=np.int64)
perpix_area_index = (perpix_area_index * byte_factors[:perpix_area_index.shape[0], None]).sum(axis=0)
assert(perpix_area_index.shape == comps_x.shape)

# determine the color for each area
unique_area_indices = np.unique(perpix_area_index)
area_colors = rng.integers(low=0, high=255, dtype=np.uint8, size=(unique_area_indices.size, 3))
area_colors[np.searchsorted(unique_area_indices, 0)] = [255, 255, 255] # we get overflow, causing our sorted area indices not to start with 0

# assign every pixel an index into the color array
perpix_color_index = np.nonzero(perpix_area_index[:, None] == unique_area_indices)[1]
data = np.reshape(area_colors[perpix_color_index], (h, w, 3))

img = Image.fromarray(data, 'RGB')
img.show()