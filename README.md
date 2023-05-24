# REAL

This repository focuses on rendering real estate room of 2D images into a 3D scene using NeRF (Neural Radiance Fields), a technique for synthesizing novel views of a scene from a set of images. It is primarily inspired by NeRF and references the following papers:

Mainly inspired by NeRF
* [Main Reference](https://www.matthewtancik.com/nerf)
  
* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
  
* [Neural Radiance Fields for Unconstrained Photo Collections](https://arxiv.org/abs/2008.02268)

<img width="600" height="200" src="./img/nerf.png"></img>

# Contents
- [REAL](#real)
- [Contents](#contents)
- [Requirements](#requirements)
- [LLFF data - Format](#llff-data---format)
    - [example for transformation](#example-for-transformation)
- [Camera pose](#camera-pose)
    - [example for camera of room](#example-for-camera-of-room)
    - [The camera parameters](#the-camera-parameters)
    - [Image list](#image-list)
    - [3D points for room](#3d-points-for-room)
- [Execution](#execution)
    - [input :](#input-)
    - [output :](#output-)
- [Training Sequence](#training-sequence)
- [Improvement](#improvement)
- [Citation](#citation)
- [License](#license)

# Requirements
For this project to render the images until now, you need to get the LLFF data format.
At the beginning, planed to convert it without any other special data format, but it is not easy to convert it.
Therefore, It is not possible if you don't have the LLFF data format which describes the camera pose and 3D points of the room and transformation of the camera parameters.

By the way, I will try to convert it to the LLFF data format automatically within a single command option(instruction) or somthing like that in the future.

The breif description about LLFF data is below.

# LLFF data - Format

LLFF can correct certain types of distortion in the input images, such as lens distortion and chromatic aberration, by estimating the intrinsic camera parameters of each image.

By estimating and correcting the intrinsic camera parameters, LLFF can enhance the quality of the images by reducing or eliminating the effects of distortion. This correction process can result in improved image sharpness, reduced color fringing, and a more accurate representation of the captured scene.

* images/ ---- .---
* sparse/ ---- .bin
* bound_poses.npy or transforms.json, {yourdata}.txt
* check [**LLFF Repository**](https://github.com/Fyusion/LLFF) first!!

### example for transformation
```
"camera_angle_x": 1.6316266081598993,
"camera_angle_y": 1.0768185778803099,
"fl_x": 903.3096914819945,
"fl_y": 904.1146220896455,
"k1": -0.0006951346026472416,
"k2": -0.0022074727073696896,
"k3": 0,
"k4": 0,
"p1": -0.00018190630274219532,
"p2": -0.00015686925639183075,
"is_fisheye": false,
"cx": 959.5738541657016,
"cy": 544.0907729519863,
"w": 1920.0,
"h": 1080.0,
"aabb_scale": 32,
"frames": [
   {
   "file_path": "./images/0017.jpg",
   ...
```

An example transformation is provided, demonstrating camera parameters like camera angle, focal length, distortion coefficients, principal point, image dimensions, and more.

# Camera pose
`./{data}/text`

### example for camera of room

These parameters define how the camera lens captures and distorts the incoming light rays. In the provided example, the focal length, principal point, and distortion coefficients are specified for a single camera.

### The camera parameters

focal length x, y /  principal point x, y / radial distortion coefficients k1~k4  / tangential distortion coefficients p1, p2.

```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 OPENCV 1920 1080 903.30969148199449 904.11462208964554 959.57385416570162 544.09077295198631 -0.0006951346026472416 -0.0022074727073696896 -0.00018190630274219532 -0.00015686925639183075
```

### Image list

These parameters represent the orientation and position of the camera in the world coordinate system. Additionally, each image may have multiple observations (POINTS2D) of 3D points in the scene, defined by their X and Y coordinates and the corresponding POINT3D_ID.

Q: quaternion which rotates a point from the world coordinate system into the camera coordinate system.

T: translation of the camera center in world coordinates.

```
IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
POINTS2D[] as (X, Y, POINT3D_ID)
Number of images: 35, mean observations per image: 1618.6571428571428
1 0.98166594374421712 0.11018315231591673 0.14857255792029417 0.046020026852707313 -0.68943682766847558 0.8318357390927269 -2.5659713605463765 1 0017.jpg
...
```

### 3D points for room
```
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# Number of points: 5740, mean track length: 9.869860627177701
944 5.1531701493470319 6.0687577564030635 5.0653561052108316 112 100 78 0.78653193433340196 32 2166 18 888 33 1394 27 1595
...
```

# Execution
You can find the sample data in the `./data` directory.
It contains some images that I took in my room and the camera pose and parameters of the room got following the structure of the LLFF.

Also you can get the whole training code in the `./train` directory which is based on [nerf](https://github.com/yenchenlin/nerf-pytorch).
Modified the code a little bit to train only with the LLFF data format and to render the images with CUDA using OpenGL.

### input : 
few images, camera pose, camera parameters with LLFF data format

<img width="300" height="160" src="./data/room/images/0001.jpg"></img>
<img width="300" height="160" src="./data/room/images/0020.jpg"></img>
<img width="300" height="160" src="./data/room/images/0027.jpg"></img>
 ...

<hr>

### output :
interactive **openGL** viewer

OpenGL is a cross-platform graphics API that specifies a standard software interface for 3D graphics processing hardware which is possible to control the camera position and viewing direction interactively.
I used OpenGL to render the images with CUDA. because it is fast and easy to use. futhermore, there's reference code for rendering with OpenGL in the LLFF repository.

Render with CUDA : [reference](https://github.com/Fyusion/LLFF#3-render-novel-views)
```
./cuda_renderer mpidir <your_posefile> <your_videofile> height crop crf
```

trained_example.mp4

https://github.com/sabin5105/REAL/assets/50198431/e3ce1e64-8531-4430-9007-4fa3ce72cab7

# Training Sequence

1. MLP
   *  Mainly predict the density of the product
   *  5-Dimension input: `x, y, z, θ, φ` -> `ρ`(density)
  
2. Volume Rendering
   *  One Ray -> One Pixel -> projection
   *  o(position of camera), d(viewing direction) -> ray (o+td)
   *  t: [0, 1] -> sample points along the ray
   *  T(t): trabsmittance
   *  c(r(t), d): weighted sum of real RGB values
   *  <img src="./img/weighted_sum.png" width="500" height="120"></img>

3. Stratified Sampling approach
   * capture the details of the scene
   * <img src="./img/stratified_sampling.png" width="300" height="50"></img>

   * <img src="./img/nerf_non_continuous.png" width="350px"></img>

4. Hierarchical Volume Sampling - Coarse network -> Fine network
   * starting with a coarse network and gradually refining the results using a fine network.
   * <img src="./img/volume_sampling.png" width="400" height="150"></img>

5. Positional Encoding
   * applied to the input to incorporate spatial information into the network.
   * <img src="./img/positional_embedding.png" width="400" height="30"></img>

   * <img src="./img/layers.png" width="400"></img>

# Improvement
* [ ] Fix positional encoding issue - Possibly the reason why the result is not good
* [ ] Add more data
* [ ] autoamtically make llff data
* [ ] train more to make it more realistic
* [ ] build configuration
* [ ] modify the whole thing to easy to use at any data with single command

# Citation
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
```
@article{mildenhall2019llff,
  title={Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines},
  author={Ben Mildenhall and Pratul P. Srinivasan and Rodrigo Ortiz-Cayon and Nima Khademi Kalantari and Ravi Ramamoorthi and Ren Ng and Abhishek Kar},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019},
}
```

# License
MIT License - see [`LICENSE`](LICENSE) for more details.