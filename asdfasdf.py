import random


def gen_cord(imgsize=(128, 128, 128), scale=(0.08, 1.), ratio = (3 / 4, 4 / 3)):
    sc = random.random() * (scale[1] - scale[0]) + scale[0]
    ra1 = random.random() * (ratio[1] - ratio[0]) + ratio[0]
    ra2 = random.random() * (ratio[1] - ratio[0]) + ratio[0]

    x = (sc * imgsize[0] * imgsize[0] * imgsize[0] / (ra1 * ra2)) ** (1 / 3)
    y = ra1 * x
    z = ra2 * x
    return int(x), int(y), int(z)
