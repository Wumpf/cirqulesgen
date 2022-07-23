from typing import Generator
from PIL import Image
import numpy as np
import cv2
from numpy.random import default_rng


class Config:
    image_path = 'couple.png'
    circle_count = 200
    radius_range = (8, 80)
    local_variance_window_size = radius_range[1] + 1
    non_background_pdf_bias = 20
    background_color = np.array([255, 255, 255], dtype=np.ubyte)


def main():
    reference_image = load_image(Config.image_path)
    h, w, _ = reference_image.shape
    reference_image_luminance = np.matmul(reference_image, [0.2126, 0.7152, 0.0722], dtype=np.float32)
    #Image.fromarray(np.reshape(reference_image_luminance, [h,w]).astype(np.ubyte), 'L').show()
    windowed_var = compute_windowed_var(Config.local_variance_window_size, reference_image_luminance)
    circle_pdf = np.sqrt(windowed_var)
    circle_pdf[(reference_image != Config.background_color).all(axis=2)] += Config.non_background_pdf_bias
    Image.fromarray((circle_pdf).astype(np.ubyte), 'L').show()
    circle_pdf /= np.sum(circle_pdf)

    flat_pixel_coord = np.stack([np.tile(np.arange(w), h), np.arange(h).repeat(w)])
    rng = default_rng(52348)

    circles_radius_sq, circles_position = gen_random_circles(flat_pixel_coord, circle_pdf, rng, Config.radius_range, Config.circle_count)

    circle_picture = draw_circles(reference_image.shape, circles_radius_sq, circles_position)
    Image.fromarray(circle_picture, 'L').show()

    perpix_area_index = compute_area_indices(flat_pixel_coord, circles_radius_sq, circles_position)
    querkle_picture = fill_areas(perpix_area_index, np.reshape(reference_image, (h * w, 3)), Config.background_color)
    save_and_show(np.reshape(querkle_picture, reference_image.shape))


def gen_random_circles(flat_pixel_coord, circle_pdf, rng: Generator, radius_range: tuple, circle_count: int):
    circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
    circles_position = rng.choice(flat_pixel_coord, circle_count, p=np.reshape(
        circle_pdf, flat_pixel_coord.shape[1]), replace=False, axis=1)
    #circles_position = rng.uniform(size=(circle_count, 2)) * image_size
    return (circles_radius_sq, circles_position)


def compute_area_indices(flat_pixel_coord, circles_radius_sq, circles_position):
    flat_pixel_coord_chunk_broadcast = np.broadcast_to(flat_pixel_coord[:, :, None], flat_pixel_coord.shape + (64,))
    perpix_area_index = np.zeros(flat_pixel_coord.shape[1], dtype=np.int64)
    byte_factors = 2 ** np.arange(64, dtype=np.int64)
    for chunk_start in range(0, circles_radius_sq.size, 64):
        chunk_end = min(chunk_start + 64, circles_radius_sq.size)
        chunk = range(chunk_start, chunk_end)
        dist_circles_sq = np.sum((flat_pixel_coord_chunk_broadcast[:, :, :len(chunk)] - circles_position[:, None, chunk]) ** 2, axis=0)
        perpix_circle_hits = dist_circles_sq < circles_radius_sq[None, chunk]
        perpix_area_index ^= (perpix_circle_hits * byte_factors[None, :len(chunk)]).sum(axis=1)
    return perpix_area_index


def fill_areas(perpix_area_index, reference_image_data, background_color):
    unique_area_indices = np.unique(perpix_area_index)
    output = np.tile(background_color, perpix_area_index.shape + (1,))
    for i in range(0, unique_area_indices.size):
        if unique_area_indices[i] == 0:
            continue
        pixels_in_area = perpix_area_index == unique_area_indices[i]
        area_color = np.average(reference_image_data[pixels_in_area], axis=0)
        output[pixels_in_area] = area_color.astype(np.ubyte)
    return output


def draw_circles(image_shape, circles_radius_sq, circles_position):
    img = 255 * np.ones(image_shape[:2], np.ubyte)
    circles_radius_int = (np.sqrt(circles_radius_sq) + 0.5).astype(np.int32)
    for radius, pos in zip(circles_radius_int, circles_position.transpose()):
        img = cv2.circle(img, pos, radius, 0)
    return img


def compute_windowed_var(window_size: int, greyscale_image):
    # similar to https://stackoverflow.com/a/36266187
    # apparently this is *a lot* faster than both a numpy or a SciPy based solution like https://stackoverflow.com/a/33497963
    sigma = window_size / 6.0
    mean = cv2.GaussianBlur(greyscale_image, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    sqrmean = cv2.GaussianBlur(greyscale_image*greyscale_image, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    return (sqrmean - mean*mean).clip(0)


def save_and_show(picture):
    img = Image.fromarray(picture, 'RGB')
    img.save("querkles.png")
    img.show()


def load_image(path):
    reference_image = Image.open(path)
    if reference_image.mode != 'RGB':
        raise Exception("Reference image needs to be RGB")
    return np.reshape(np.array(reference_image.getdata()), (reference_image.size[1], reference_image.size[0], 3))


if __name__ == '__main__':
    main()
