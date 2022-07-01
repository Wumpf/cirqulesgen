from PIL import Image
import numpy as np


def is_in_circle(x, y, p_x, p_y, r):
    return 1

# new white image
w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:w, 0:h] = [255, 255, 255]

# draw a circle
r = 40
pos = (256, 256)

comps_x = np.tile(np.arange(w), h)
comps_y = np.arange(h).repeat(w)

in_circle = ((comps_x - pos[0]) ** 2 + (comps_y - pos[1]) ** 2) < r*r
data[(comps_x[in_circle], comps_y[in_circle])] = [0, 0, 0]


img = Image.fromarray(data, 'RGB')
img.show()