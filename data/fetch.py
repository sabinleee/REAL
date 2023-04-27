import os 
import urllib.request
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if not os.path.exists('./data/tiny_nerf_data.npz'):
        path = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
        temp = urllib.request.urlretrieve(path, './data/tiny_nerf_data.npz')

    data = np.load('./data/tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    testimg, testpose = images[101], poses[101]
    images = images[:100,...,:3]
    poses = poses[:100]

    plt.imshow(testimg)
    plt.show()