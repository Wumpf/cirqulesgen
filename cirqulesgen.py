import sys
import traceback
from typing import Generator
import numpy as np
import cv2
from numpy.random import default_rng


class Config:
    image_path = 'sparrow.webp'
    circle_count = 600
    radius_range = (5, 50)
    local_variance_window_size = radius_range[1] + 1
    non_background_pdf_bias = 20
    num_grey_levels = 6  # including background


def main():
    reference_image = load_image_to_grey(Config.image_path)
    grey_levels = determine_grey_levels(Config.num_grey_levels, reference_image)
    h, w = reference_image.shape
    # Image.fromarray(np.reshape(reference_image_luminance, [h,w]).astype(np.ubyte), 'L').show()
    windowed_var = compute_windowed_var(Config.local_variance_window_size, reference_image)
    circle_pdf = np.sqrt(windowed_var)
    circle_pdf[reference_image < 230] += Config.non_background_pdf_bias
    circle_pdf /= np.sum(circle_pdf)
    cv2.imshow('circle_pdf', circle_pdf * reference_image.size * 0.25)

    flat_pixel_coord = np.stack([np.tile(np.arange(w), h), np.arange(h).repeat(w)])

    for i in range(0, 1):
        rng = default_rng(52348 + i)

        circles_radius_sq, circles_position = gen_random_circles(
            flat_pixel_coord, circle_pdf, rng, Config.radius_range, Config.circle_count)

        #circle_picture = draw_circles(reference_image.shape, circles_radius_sq, circles_position)
        #cv2.imshow('circles', circle_picture)

        perpix_area_index = compute_area_indices(flat_pixel_coord, reference_image.shape, circles_radius_sq, circles_position)
        cirqules_picture = fill_areas(grey_levels, perpix_area_index, reference_image)

        cirqules_error = ((reference_image - cirqules_picture) ** 2).mean()
        print("error", i, cirqules_error)

        cv2.imshow('cirqules ' + str(i), cirqules_picture)
        #cv2.imwrite('cirqules.png', cirqules_picture)


def determine_grey_levels(num_grey_levels: int, greyscale_image: np.ndarray) -> np.ndarray:
    image_values = greyscale_image.reshape(greyscale_image.size).astype(np.float32)
    # generate one extra label/grey-level and clamp it to 255
    # in most cases it will be that value anyways as background color is defined as white
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, grey_levels = cv2.kmeans(image_values, num_grey_levels, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    grey_levels = grey_levels.flatten().astype(np.uint8)
    grey_levels[np.argmax(grey_levels)] = 255

    quantized_picture = grey_levels[labels.flatten()].reshape(greyscale_image.shape)
    cv2.imshow('original & quantized image', cv2.hconcat((greyscale_image, quantized_picture)))

    grey_levels.sort()
    print("grey levels", grey_levels)
    return grey_levels


def gen_random_circles(flat_pixel_coord, circle_pdf, rng: Generator, radius_range: tuple, circle_count: int) -> tuple:
    circles_radius_sq = rng.triangular(left=radius_range[0], mode=radius_range[0], right=radius_range[1], size=circle_count) ** 2
    circles_position = rng.choice(flat_pixel_coord, circle_count, p=np.reshape(
        circle_pdf, flat_pixel_coord.shape[1]), replace=False, axis=1)
    # circles_position = rng.uniform(size=(circle_count, 2)) * image_size
    return (circles_radius_sq, circles_position)


def compute_area_indices(flat_pixel_coord, image_shape, circles_radius_sq, circles_position) -> np.ndarray:
    # we need to assign a unique index to every area
    # this is surprisingly hard since there's many more areas than circles
    # a error free solution would be to have a bit set per circle, but we can't afford that many bits!
    # so instead we accept some error and reuse bits for circles

    # previous, slower version using bool matricies
    # pixel_coord_y = np.broadcast_to(np.arange(0, image_shape[0])[:, None], image_shape)
    # pixel_coord_x = np.broadcast_to(np.arange(0, image_shape[1])[None, :], image_shape)
    # flat_pixel_coord_chunk_broadcast = np.broadcast_to(flat_pixel_coord[:, :, None], flat_pixel_coord.shape + (64,))
    # perpix_area_index = np.zeros(flat_pixel_coord.shape[1], dtype=np.int64)
    # byte_factors = 2 ** np.arange(64, dtype=np.int64)
    # for chunk_start in range(0, circles_radius_sq.size, 64):
    #     chunk_end = min(chunk_start + 64, circles_radius_sq.size)
    #     chunk = range(chunk_start, chunk_end)
    #     dist_circles_sq = np.sum((flat_pixel_coord_chunk_broadcast[:, :, :len(chunk)] - circles_position[:, None, chunk]) ** 2, axis=0)
    #     perpix_circle_hits = dist_circles_sq < circles_radius_sq[None, chunk]
    #     perpix_area_index ^= (perpix_circle_hits * byte_factors[None, :len(chunk)]).sum(axis=1)

    # scan line based circle drawing
    perpix_area_index = np.zeros(image_shape, dtype=np.int64)
    byte_factors = 2 ** np.arange(64, dtype=np.int64)
    circles_radius = np.sqrt(circles_radius_sq)
    for i, r in enumerate(circles_radius):
        p = circles_position[:, i]
        ymin = np.clip((p[1] - r + 0.5).astype(np.uint32), 0, image_shape[0] - 1)
        ymax = np.clip((p[1] + r + 0.5).astype(np.uint32), 0, image_shape[0] - 1) + 1
        dy = np.sqrt(np.maximum(0, -(np.arange(ymin, ymax) - p[1]) ** 2 + circles_radius_sq[i]))
        x_ranges = np.clip((np.concatenate((-dy[None, :], dy[None, :])) + p[0] + 0.5).astype(np.uint32), 0, image_shape[1])
        for iy, y in enumerate(range(ymin, ymax)):
            perpix_area_index[y, x_ranges[0, iy]:x_ranges[1, iy]] ^= byte_factors[i % byte_factors.size]

    return perpix_area_index


def fill_areas(grey_levels: np.ndarray, perpix_area_index: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    perpix_area_index = np.reshape(perpix_area_index, reference_image.size)
    reference_image_data = np.reshape(reference_image, reference_image.size)
    unique_area_indices = np.unique(perpix_area_index)
    output = 255 * np.ones(perpix_area_index.shape, np.ubyte)
    for i in range(0, unique_area_indices.size):
        if unique_area_indices[i] == 0:
            continue
        pixels_in_area = perpix_area_index == unique_area_indices[i]
        covered_pixels = reference_image_data[pixels_in_area]
        color_idx = np.argmin(np.sum((covered_pixels[None, :] - grey_levels[:, None]) ** 2, axis=1))
        output[pixels_in_area] = grey_levels[color_idx]
    return np.reshape(output, reference_image.shape)


def draw_circles(image_shape, circles_radius_sq, circles_position) -> np.ndarray:
    img = 255 * np.ones(image_shape, np.ubyte)
    circles_radius_int = (np.sqrt(circles_radius_sq) + 0.5).astype(np.int32)
    for radius, pos in zip(circles_radius_int, circles_position.transpose()):
        img = cv2.circle(img, pos, radius, 0)
    return img


def compute_windowed_var(window_size: int, greyscale_image: np.ndarray) -> np.ndarray:
    # similar to https://stackoverflow.com/a/36266187
    # apparently this is *a lot* faster than both a numpy or a SciPy based solution like https://stackoverflow.com/a/33497963
    sigma = window_size / 6.0
    greyscale_image = greyscale_image.astype(np.float32)
    mean = cv2.GaussianBlur(greyscale_image, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    sqrmean = cv2.GaussianBlur(greyscale_image*greyscale_image, (window_size, window_size), sigma, borderType=cv2.BORDER_REFLECT)
    return (sqrmean - mean*mean).clip(0)


def load_image_to_grey(path) -> np.ndarray:
    original = cv2.imread(path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('input-grey', gray)
    return gray


if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()
    finally:
        # want so see pictures in the end *even* if something crashed!
        if len(sys.argv) == 1 or sys.argv[1] != "--no-wait":
            cv2.waitKey(0)
            cv2.destroyAllWindows()
