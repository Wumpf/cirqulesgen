from time import perf_counter
from PIL import Image
import numpy as np
from numpy.random import default_rng
import time

reference_image = Image.open('tiger-small.png')
w, h = reference_image.size
print("width", w, "height", h)
circle_count = 330
radius_range = (5, 60)

t0 = time.time()

pixel_coordinates = np.stack([np.tile(np.arange(w), h), np.arange(h).repeat(w)])

# generate random circles
rng = default_rng(28346)
circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
circles_position = rng.uniform(size=(circle_count, 2)) * [w, h]

# determine area indices
print("determining area indices per pixel")
pixel_coordinates_chunk_broadcast = np.broadcast_to(pixel_coordinates[None, :, :], (64,) + pixel_coordinates.shape)
perpix_area_index = np.zeros(pixel_coordinates.shape[1], dtype=np.int64)
byte_factors = 2 ** np.arange(64, dtype=np.int64)
for chunk_start in range(0, circle_count, 64):
    chunk_end = min(chunk_start + 64, circle_count)
    chunk = range(chunk_start, chunk_end)
    dist_circles_sq = np.sum((pixel_coordinates_chunk_broadcast[:len(chunk), :, :] - circles_position[chunk, :, None]) ** 2, axis=1)
    perpix_circle_hits = dist_circles_sq < circles_radius_sq[chunk, None]
    perpix_area_index ^= (perpix_circle_hits * byte_factors[:len(chunk), None]).sum(axis=0)
print("Total time elapsed: ", time.time() - t0)

print("determining affected pixels per area")
unique_area_indices = np.unique(perpix_area_index)
print(unique_area_indices.size)
pix_areas = np.broadcast_to(perpix_area_index[None, :], (unique_area_indices.size, perpix_area_index.size)) == unique_area_indices[:, None]
print("Total time elapsed: ", time.time() - t0)

print("determine the color for each area")
reference_image_data = np.array(reference_image.getdata())
area_colors = np.zeros((unique_area_indices.size, 4))
for i in range(0, unique_area_indices.size):
    area_colors[i] = np.average(reference_image_data[pix_areas[i, :]], axis=0)

# assign every pixel an index into the color array
pix_areas = perpix_area_index[:, None] == unique_area_indices
perpix_color_index = np.nonzero(pix_areas)[1]
data = np.reshape(area_colors[perpix_color_index], (h, w, 4)).astype(np.ubyte)

print("Total time elapsed: ", time.time() - t0)

img = Image.fromarray(data, 'RGBA')
img.show()
img.save("querkles.png")