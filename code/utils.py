import numpy as np
import matplotlib.pyplot as plt
import skimage.draw as sd
import scipy.ndimage as nd
import imageio


def create_circular_mask(imshape=(512, 512), center=(256, 256), r=5):
    y, x = np.ogrid[:imshape[0], :imshape[1]]
    d = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = d <= r
    return mask


def draw_circle_perim_aa(im, center=(256, 256), r=20, intensity=1, bg=0):
    r, c, v = sd.circle_perimeter_aa(center[0], center[1], r)
    v = v * intensity
    v = v / (intensity) * (intensity - bg) + bg
    im[r, c] = v


def draw_circle(im, center=(256, 256), r=20, intensity=1):
    rr, cc = sd.disk(center, r)
    vals = np.ones(rr.shape)
    im[rr, cc] = vals * intensity


def radon(
    im: np.ndarray,
    theta: np.ndarray = np.arange(360),
):
    p = np.zeros((im.shape[0], len(theta)), dtype=np.float64)

    for i, t in enumerate(theta):
        p[:, i] = nd.rotate(im, t, reshape=False).sum(0)
    return p


def get_phantom():
    size = 512
    r_big = 60
    r_small = 20
    r_tiny = 10
    i_big = 1
    i_small = 0.5
    half = size // 2
    off = 50
    img = np.zeros((size, size))
    cs = [(half, half + off), (half, half - off)]
    for center in cs:
        draw_circle_perim_aa(img, center, r_big, i_big)
    for center in cs:
        draw_circle(img, center, r_big, i_big)
    for center in cs:
        draw_circle_perim_aa(img, center, r_small, i_small, bg=i_big)
        draw_circle(img, center, r_small, i_small)
    for center in [(316, 256), (196, 256)]:
        draw_circle_perim_aa(img, center, r_tiny)
        draw_circle(img, center, r_tiny)
    sum = img.sum(0)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.figure()
    plt.plot(sum)
    plt.figure()
    plt.plot(img.sum(1))
    plt.show()


if __name__ == '__main__':
    im = imageio.imread('./tu.png').mean(-1)
    radon = radon(im, theta=np.array([-20, 50]))
    radon[radon < 1e-10] = 0
    index = np.arange(radon.shape[0])[:,None]
    print(index)
    radon_ex = np.hstack((index, radon))
    np.savetxt('./radon.csv', radon_ex, delimiter=',')
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.figure()
    plt.plot(im.sum(0))
    plt.figure()
    plt.plot(im.sum(1))
    plt.figure()
    plt.imshow(radon)
    plt.show()
    # get_phantom()
