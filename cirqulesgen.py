import sys
from typing import Generator
from PIL import Image
import numpy as np
from numpy.random import default_rng

image_path = 'tiger-small.png'
circle_count = 200
radius_range = (5, 50)
reduce_variance_cutoff = 100
num_redistribute_passes = 10

def gen_random_circles(image_size, rng: Generator):
    circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
    circles_position = rng.uniform(size=(circle_count, 2)) * image_size
    return (circles_radius_sq, circles_position)

def redistribute_low_variance_circles(image_size, pixel_coordinates, circles_radius_sq, circles_position, reference_image_data_luminance):
    # idea: if the variance within a circle it small it covers a homogenous area, i.e. is not interesting
    # as we typically want to have circle cuts corresponding to features
    # do *not* change the circle size, otherwise we're just biasing for larger circles!
    redistribute_count = 0
    for i, (radius_sq, position) in enumerate(zip(circles_radius_sq, circles_position)):
        dist_circles_sq = np.sum((pixel_coordinates - position[:, None]) ** 2, axis=0)
        perpix_circle_hits = dist_circles_sq < radius_sq
        variance = np.var(reference_image_data_luminance[perpix_circle_hits])
        if variance < reduce_variance_cutoff:
            circles_position[i] = rng.uniform(size=2) * image_size
            redistribute_count += 1
    print("redistributed", redistribute_count, "circles")
    return redistribute_count

def compute_area_indices(pixel_coordinates, circles_radius_sq, circles_position):
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
    output = np.zeros(perpix_area_index.shape + (4,), dtype=np.ubyte)
    for i in range(0, unique_area_indices.size):
        pixels_in_area = perpix_area_index == unique_area_indices[i]
        area_color = np.average(reference_image_data[pixels_in_area], axis=0)
        output[pixels_in_area] = area_color.astype(np.ubyte)

    return np.reshape(output, image_size + (4,))

# print("Total time elapsed: ", time.time() - t0)

def save_and_show(picture):
    img = Image.fromarray(picture, 'RGBA')
    img.save("querkles.png")
    img.show()

if __name__ == '__main__':
    reference_image = Image.open('tiger-small.png')
    reference_image_data = np.array(reference_image.getdata())
    w,h = reference_image.size
    # palletize picture first?
    reference_image_data_luminance = np.matmul(reference_image_data, [0.2126, 0.7152, 0.0722, 0], dtype=np.float32)
    #img_luminance = Image.fromarray(np.reshape(reference_image_data_luminance, reference_image.size).astype(np.ubyte), 'L')
    #img_luminance.show()


    pixel_coordinates = np.stack([np.tile(np.arange(w), h), np.arange(h).repeat(w)])
    rng = default_rng(52346)

    circles_radius_sq, circles_position = gen_random_circles(reference_image.size, rng)
    for p in range(0, num_redistribute_passes):
        if redistribute_low_variance_circles(reference_image.size, pixel_coordinates, circles_radius_sq, circles_position, reference_image_data_luminance) == 0:
            break

    perpix_area_index = compute_area_indices(pixel_coordinates, circles_radius_sq, circles_position)
    querkle_picture = fill_areas(reference_image.size, perpix_area_index, reference_image_data)
    save_and_show(querkle_picture)