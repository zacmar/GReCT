import numpy as np
import matplotlib.pyplot as plt
import astra


def _main():
    im = np.zeros((20, 20))
    im[9:11, 9] = 1
    im[9:11, 10] = 1000
    plt.figure()
    plt.imshow(im)
    vol_geom = astra.create_vol_geom(*im.shape)
    recon_id = astra.data2d.create('-vol', vol_geom, 0)
    proj_geom = astra.create_proj_geom(
        'parallel', 1., 1003,
        np.linspace(0, np.pi, 360, False)
    )
    proj_id = astra.create_projector('line', proj_geom, vol_geom)
    s_id, sinogram = astra.create_sino(im, proj_id, vol_geom)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ProjectionDataId'] = s_id
    cfg['ReconstructionDataId'] = recon_id
    sart_id = astra.algorithm.create(cfg)
    astra.algorithm.run(sart_id, 10)
    r = astra.data2d.get(recon_id)
    plt.figure()
    plt.imshow(r, cmap='gray')
    plt.show()


if __name__ == '__main__':
    _main()
