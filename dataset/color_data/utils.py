from matplotlib import pyplot as plt
import numpy as np
import math


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def gen_image_color(color_list, small=False):
    angles = np.linspace(0, 2 * math.pi, 7)
    margin = 5
    circle_locx = 0.5 + 0.36 * np.cos(angles)
    circle_locy = 0.5 + 0.36 * np.sin(angles)
    fig = plt.figure(figsize=((64+2*margin)/10.0, (64+2*margin)/10.0), dpi=10)
    ax = plt.gca()
    radius = 0.12
    if small:
        radius = 0.1
    for i in range(6):
        circle = plt.Circle((circle_locx[i], circle_locy[i]), radius, color=color_list[i])
        ax.add_artist(circle)
    plt.axis('off')
    plt.tight_layout()

    # plt.subplots_adjust(left=0, right=6.4, top=6.4, bottom=0)
    # fig.savefig('plotcircles.png')
    arr = fig2data(fig)
    arr = arr[margin:64+margin, margin:64+margin, :3]

    plt.close(fig)
    return arr


def gen_image_location(color_list):
    radius = 0.08
    m = 0.16
    shift1 = np.random.uniform(m + radius, 0.5 - m - radius)
    shift2 = np.random.uniform(0.5 + m + radius, 1.0 - m - radius)
    if np.random.normal() > 0:
        shiftx = shift1
    else:
        shiftx = shift2
    shift1 = np.random.uniform(0.12, 0.3)
    shift2 = np.random.uniform(0.7, 0.88)
    if np.random.normal() > 0:
        shifty = shift1
    else:
        shifty = shift2
    margin = 5
    fig = plt.figure(figsize=((64+2*margin)/10.0, (64+2*margin)/10.0), dpi=10)
    ax = plt.gca()

    circle = plt.Circle((shiftx, shifty), radius, color=color_list[0])
    ax.add_artist(circle)
    plt.axis('off')
    plt.tight_layout()

    # plt.subplots_adjust(left=0, right=6.4, top=6.4, bottom=0)
    # fig.savefig('plotcircles.png')
    arr = fig2data(fig)
    arr = arr[margin:64+margin, margin:64+margin, :3]

    plt.close(fig)
    return arr


def gen_image_count(color_list, num_object=3, overlap=False):
    radius = 0.08
    while True:
        shifts = np.random.uniform(radius, 1.0-radius, size=(num_object, 2))
        dist1 = np.tile(np.expand_dims(shifts, axis=0), (num_object, 1, 1))
        dist2 = np.tile(np.expand_dims(shifts, axis=1), (1, num_object, 1))
        dist = np.sqrt(np.sum(np.square(dist1 - dist2), axis=2))
        np.fill_diagonal(dist, 1.0)
        if not overlap and np.min(dist) > 2.1 * radius:
            break
        if overlap and np.min(dist) > 2 * radius * 0.9:
            break

    margin = 5
    fig = plt.figure(figsize=((64+2*margin)/10.0, (64+2*margin)/10.0), dpi=10)
    ax = plt.gca()
    for i in range(num_object):
        circle = plt.Circle(shifts[i], radius, color=color_list[i])
        ax.add_artist(circle)
    plt.axis('off')
    plt.tight_layout()

    # plt.subplots_adjust(left=0, right=6.4, top=6.4, bottom=0)
    # fig.savefig('plotcircles.png')
    arr = fig2data(fig)
    arr = arr[margin:64+margin, margin:64+margin, :3]

    plt.close(fig)
    return arr


def gen_image_size(color_list, num_object=3, gap=0.04):
    if np.random.randint(0, 2) == 0:
        radius = np.random.uniform(0.08, 0.08+(0.22-gap)/2)
    else:
        radius = np.random.uniform(0.30-(0.22-gap)/2, 0.30)
    while True:
        shifts = np.random.uniform(radius, 1.0 - radius, size=(num_object, 2))
        dist1 = np.tile(np.expand_dims(shifts, axis=0), (num_object, 1, 1))
        dist2 = np.tile(np.expand_dims(shifts, axis=1), (1, num_object, 1))
        dist = np.sqrt(np.sum(np.square(dist1 - dist2), axis=2))
        np.fill_diagonal(dist, 1.0)
        if np.min(dist) > 2.1 * radius:
            break

    margin = 5
    fig = plt.figure(figsize=((64 + 2 * margin) / 10.0, (64 + 2 * margin) / 10.0), dpi=10)
    ax = plt.gca()
    for i in range(num_object):
        circle = plt.Circle(shifts[i], radius, color=color_list[i])
        ax.add_artist(circle)
    plt.axis('off')
    plt.tight_layout()

    # plt.subplots_adjust(left=0, right=6.4, top=6.4, bottom=0)
    # fig.savefig('plotcircles.png')
    arr = fig2data(fig)
    arr = arr[margin:64 + margin, margin:64 + margin, :3]

    plt.close(fig)
    return arr