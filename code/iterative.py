import astra
import numpy as np
import imageio
import skimage.transform as tr
import matplotlib.pyplot as plt


def kaczmarz(A, p, f=None):
    f = f or np.zeros(A.shape[1])
    rownorms = A.power(2).sum(1).A.squeeze()
    alpha = 0.3

    for outer in range(1):
        print(outer, flush=True)
        for n in range(A.shape[0]):
            row = (A[n]).A.squeeze()
            f = f - alpha * (f @ row - p[n]) / rownorms[n] * A[n]
            if n % (A.shape[0] // 9) == A.shape[0] // 9 - 1:
                plt.clf()
                im = np.clip(f.reshape((256, 256)), 0, 1)
                plt.imshow(im, cmap='gray')
                plt.pause(0.01)


def sirt(A, p, f=None):
    f = f or np.zeros(A.shape[1])

    C = A.sum(0).A.squeeze()
    R = A.sum(1).A.squeeze()

    for outer in range(101):
        f = f - (A.T @ ((A @ f - p) / R)) / C
        plt.clf()

        im = f.reshape((256, 256))
        plt.imshow(im, cmap='gray')
        # imageio.imwrite(f'./sirt_{outer}.png', im)
        plt.pause(0.01)

    return f


def kaczmarz_ex():
    A = np.array([
        [-0.5, 1],
        [-1.5, 1],
        [1, 0],
    ])
    p = np.array([2, -1.5, 2])
    f = np.zeros(A.shape[1])
    rownorms = (A**2).sum(1)

    iter = 50
    fs = np.empty((iter + 1, 2))
    fs[0] = f
    for n in range(iter):
        r_idx = n % 3
        f = f - (f @ A[r_idx] - p[r_idx]) / rownorms[r_idx] * A[r_idx]
        fs[n + 1] = f
    np.savetxt('./kaczmarz_.txt', fs, delimiter=',')


def sirt_ex():
    A = np.array([
        [-0.5, 1],
        [-1.5, 1],
    ])
    p = np.array([2, -1.5])
    f = np.zeros(A.shape[1])

    iter = 3
    fs = np.empty((iter + 1, 2))
    fs[0] = f
    C = A.sum(0)
    R = A.sum(1)

    for n in range(iter):
        f = f - (A.T @ ((A @ f - p) / R)) / C
        fs[n + 1] = f
        print(f)
    np.savetxt('./sirt.txt', fs, delimiter=',')


def astra_ex(im):
    vol_geom = astra.create_vol_geom(400, 400)
    for d_d in [0.3, 1, 3]:
        for n_theta in [30, 180, 400]:
            recon_id = astra.data2d.create('-vol', vol_geom, 0)
            proj_geom = astra.create_proj_geom(
                'parallel', d_d, int(420 // d_d),
                np.linspace(0, np.pi, n_theta, False))
            proj_id = astra.create_projector('linear', proj_geom, vol_geom)
            s_id, sinogram = astra.create_sino(im, proj_id, vol_geom)
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ProjectionDataId'] = s_id
            cfg['ReconstructionDataId'] = recon_id
            sart_id = astra.algorithm.create(cfg)
            astra.algorithm.run(sart_id, 10)
            r = astra.data2d.get(recon_id)
            imageio.imsave(f'fbp_detectordiff_{d_d}_ntheta_{n_theta}.png',
                           np.clip(r.reshape(im.shape), 0, 1))


def scatter_example():
    im = tr.resize(imageio.imread('./tu.png'), (256, 256)).mean(-1)
    vol_geom = astra.create_vol_geom(256, 256)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 384,
                                       np.linspace(0, np.pi, 180, False))
    proj_id = astra.create_projector('strip', proj_geom, vol_geom)
    matrix_id = astra.projector.matrix(proj_id)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    W = astra.matrix.get(matrix_id)
    sino = W @ im.flatten()
    print(sino.shape)
    sino = sino.reshape((180, -1))
    print(sino.shape)
    x = np.arange(sino.shape[1]) - sino.shape[1] // 2
    alpha = 0.00
    scatter = -alpha * x**2
    scatter -= scatter.min()
    sino += scatter
    s_id = astra.data2d.create('-sino', proj_geom, sino)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = s_id
    cfg['ReconstructionDataId'] = recon_id
    sart_id = astra.algorithm.create(cfg)
    astra.algorithm.run(sart_id, 10)
    r = astra.data2d.get(recon_id)
    imageio.imsave('scatter.png', np.clip(r.reshape(im.shape), 0, 1))


if __name__ == '__main__':
    plt.figure()
    plt.ion()
    # im = tr.resize(imageio.imread('./tu.png'), (256, 256)).mean(-1)
    # vol_geom = astra.create_vol_geom(256, 256)
    # proj_geom = astra.create_proj_geom(
    #     'parallel', 1.0, 384, np.linspace(0, np.pi, 180, False)
    # )
    # proj_id = astra.create_projector('line', proj_geom, vol_geom)
    # matrix_id = astra.projector.matrix(proj_id)
    # W = astra.matrix.get(matrix_id)
    # sino = W @ im.flatten()
    scatter_example()
