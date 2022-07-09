import sys
from PIL import Image
import numpy as np
from numpy.random import default_rng

image_path = 'tiger-small.png'
circle_count = 330
radius_range = (5, 60)

def gen_random_circles(image_size, rng_seed: int):
    rng = default_rng(rng_seed)
    circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
    circles_position = rng.uniform(size=(circle_count, 2)) * image_size
    return (circles_radius_sq, circles_position)

def compute_area_indices(image_size, circles_radius_sq, circles_position):
    pixel_coordinates = np.stack([np.tile(np.arange(image_size[0]), image_size[1]), np.arange(image_size[1]).repeat(image_size[0])])
    pixel_coordinates_chunk_broadcast = np.broadcast_to(pixel_coordinates[None, :, :], (64,) + pixel_coordinates.shape)
    perpix_area_index = np.zeros(pixel_coordinates.shape[1], dtype=np.int64)
    byte_factors = 2 ** np.arange(64, dtype=np.int64)
    for chunk_start in range(0, circle_count, 64):
        chunk_end = min(chunk_start + 64, circle_count)
        chunk = range(chunk_start, chunk_end)
        dist_circles_sq = np.sum((pixel_coordinates_chunk_broadcast[:len(chunk), :, :] - circles_position[chunk, :, None]) ** 2, axis=1)
        perpix_circle_hits = dist_circles_sq < circles_radius_sq[chunk, None]
        perpix_area_index ^= (perpix_circle_hits * byte_factors[:len(chunk), None]).sum(axis=0)
    return perpix_area_index

def fill_areas(image_size, perpix_area_index, reference_image_data):
    unique_area_indices = np.unique(perpix_area_index)
    area_colors = np.zeros((unique_area_indices.size, 4))
    for i in range(0, unique_area_indices.size):
        pixels_in_area = perpix_area_index == unique_area_indices[i]
        area_colors[i] = np.average(reference_image_data[pixels_in_area], axis=0)

    # assign every pixel an index into the color array
    pix_areas = perpix_area_index[:, None] == unique_area_indices
    perpix_color_index = np.nonzero(pix_areas)[1]
    return np.reshape(area_colors[perpix_color_index], image_size + (4,)).astype(np.ubyte)

# print("Total time elapsed: ", time.time() - t0)

def save_and_show(picture):
    img = Image.fromarray(picture, 'RGBA')
    img.save("querkles.png")
    img.show()

if __name__ == '__main__':
    reference_image = Image.open('tiger-small.png')
    reference_image_data = np.array(reference_image.getdata())
    circles_radius_sq, circles_position = gen_random_circles(reference_image.size, 52346)
    perpix_area_index = compute_area_indices(reference_image.size, circles_radius_sq, circles_position)
    querkle_picture = fill_areas(reference_image.size, perpix_area_index, reference_image_data)
    #save_and_show(querkle_picture)