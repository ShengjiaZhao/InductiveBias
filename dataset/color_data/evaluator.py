if __name__ == '__main__':
    from utils import *
else:
    from .utils import *

import matplotlib

color_codes = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]).astype(np.float)
ref = gen_image(color_codes, small=True)
masks = []
print(ref.dtype, np.max(ref), np.min(ref))
plt.imshow(ref)
plt.show()
for i in range(6):
    color_code = color_codes[i]
    color_mask = np.logical_and(np.logical_and(
        ref[:, :, 0] == color_code[0]*255, ref[:, :, 1] == color_code[1]*255),
                                ref[:, :, 2] == color_code[2]*255).astype(np.float)
    masks.append(np.expand_dims(color_mask, axis=-1))
    # plt.imshow(color_mask)
    # plt.show()





if __name__ == '__main__':
    arr = gen_image(['r', 'g', 'b', 'y', 'r', 'g']).astype(np.float) / 255.
    plt.imshow(arr)
    plt.show()

    colors = eval_colors(arr)
    arr = gen_image(colors)
    plt.imshow(arr)
    plt.show()
