from matplotlib import pyplot as plt
import numpy as np
import math
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


def gen_color_random(color_list, color_weights):
    canvas = np.zeros(shape=(64 * 64, 3), dtype=np.float32)
    num_pixels = 64 * 64
    # color_list = np.random.uniform(0, 1, size=(5, 3))
    color_weights = np.cumsum(color_weights)
    color_weights = np.insert(color_weights, 0, 0.0)
    for c in range(len(color_list)):
        canvas[int(num_pixels * color_weights[c]):int(num_pixels * color_weights[c + 1])] = color_list[c]
    canvas = np.random.permutation(canvas)
    return np.reshape(canvas, [64, 64, 3])


def gen_color_band(color_list, color_weights):
    resolution = 1000
    lutx = []
    luty = []
    lut2 = []
    for i in range(64):
        for j in range(64):
            x = (i - 32.0) / 32.0
            y = (j - 32.0) / 32.0
            if x ** 2 + y ** 2 >= 0.45 ** 2 and x ** 2 + y ** 2 <= 0.95 ** 2:
                lutx.append(i)
                luty.append(j)
                lut2.append(int((math.atan2(y, x) / 2.0 / math.pi + 0.5) * (resolution - 1)))

    color_band = np.zeros(shape=(resolution, 3), dtype=np.float32)
    color_weights = np.cumsum(color_weights)
    color_weights = np.insert(color_weights, 0, 0.0)
    for c in range(len(color_list)):
        color_band[int(resolution * color_weights[c]):int(resolution * color_weights[c + 1])] = color_list[c]

    for i in range(3):
        swap_start, swap_end = 0, 0
        while swap_end - swap_start < 10:
            swap_start = np.random.randint(0, resolution // 2 - 10)
            swap_end = np.random.randint(10, resolution // 2)
        recv_start = np.random.randint(resolution // 2, resolution - swap_end + swap_start)
        recv_end = recv_start + swap_end - swap_start
        buffer = color_band[swap_start:swap_end].copy()
        color_band[swap_start:swap_end] = color_band[recv_start:recv_end]
        color_band[recv_start:recv_end] = buffer

    canvas = np.ones(shape=(64, 64, 3), dtype=np.float32)
    canvas[lutx, luty] = color_band[lut2]
    return canvas


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


def gen_combi(params):
    count = int(params[0])

    # 0.35 + k/20: 0.4-0.8, range 1-9
    size = int(params[1])
    if size == 0:
        size = np.random.randint(1, 10) / 20.0
    else:
        size = size / 20.0 + 0.35

    # (k-5)/20: -0.2 - 0.2
    locx = int(params[2])
    if locx == 0:
        locx = (np.random.randint(1, 10) - 5) / 20.0
    else:
        locx = (locx - 5) / 20.0

    locy = int(params[3])
    if locy == 0:
        locy = (np.random.randint(1, 10) - 5) / 20.0
    else:
        locy = (locy - 5) / 20.0

    # k/10, 0.0-1.0
    color = int(params[4])
    if color == 0:
        color = np.random.randint(1, 10) / 10.0
    else:
        color = color / 10.0

    resolution = 1000
    lutx = []
    luty = []
    lut2 = []
    for i in range(64):
        for j in range(64):
            x = (i - 32.0) / 32.0 - locx
            y = (j - 32.0) / 32.0 - locy
            if x ** 2 + y ** 2 <= size ** 2:
                lutx.append(i)
                luty.append(j)
                lut2.append(int((math.atan2(y, x) / 2.0 / math.pi + 0.5) * (resolution - 1)))

    random_color = np.random.uniform(0, 0.9, size=(count, 3))
    random_color[0, 0] = np.random.uniform(0.8, 0.9)
    random_color[0, 1:] = 0.0
    random_color[1:, 0] = 0.0
    random_weights = np.random.uniform(0.01, 1, size=count - 1)
    random_weights = np.concatenate([[color], (1 - color) * random_weights / np.sum(random_weights)])

    color_band = np.zeros(shape=(resolution, 3), dtype=np.float32)
    color_weights = np.cumsum(random_weights)
    color_weights = np.insert(color_weights, 0, 0.0)
    for c in range(len(random_color)):
        color_band[int(resolution * color_weights[c]):int(resolution * color_weights[c + 1])] = random_color[c]

    for i in range(3):
        swap_start, swap_end = 0, 0
        while swap_end - swap_start < 10:
            swap_start = np.random.randint(0, resolution // 2 - 10)
            swap_end = np.random.randint(10, resolution // 2)
        recv_start = np.random.randint(resolution // 2, resolution - swap_end + swap_start)
        recv_end = recv_start + swap_end - swap_start
        buffer = color_band[swap_start:swap_end].copy()
        color_band[swap_start:swap_end] = color_band[recv_start:recv_end]
        color_band[recv_start:recv_end] = buffer

    canvas = np.ones(shape=(64, 64, 3), dtype=np.float32)
    canvas[lutx, luty] = color_band[lut2]
    return canvas
