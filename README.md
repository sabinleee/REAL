# REAL 
REAL: Render Real Estate 2D images into 3D scene with NeRF

Mainly inspired by NeRF
* [Main Reference](https://www.matthewtancik.com/nerf)
  
* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
  
* [Neural Radiance Fields for Unconstrained Photo Collections](https://arxiv.org/abs/2008.02268)

<img width="600" height="200" src="./img/nerf.png"></img>

# LLFF data - Format

* sparse/ ---- .bin
* bound_poses.npy or transforms.json, {yourdata}.txt
* check [**LLFF Repository**](https://github.com/Fyusion/LLFF) first!!

* LLFF can correct certain types of distortion in the input images, such as lens distortion and chromatic aberration, by estimating the intrinsic camera parameters of each image.

example for transformation
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

# Camera pose
`./{data}/text`

example for camera of room
```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 OPENCV 1920 1080 903.30969148199449 904.11462208964554 959.57385416570162 544.09077295198631 -0.0006951346026472416 -0.0022074727073696896 -0.00018190630274219532 -0.00015686925639183075
```

# Execution
input : few images

<img width="300" height="160" src="./data/room/0001.jpg"></img>
<img width="300" height="160" src="./data/room/0020.jpg"></img>

output : trained_example.mp4

<video width="600" height="320" controls>
  <source src="./data/trained_example.mp4" type="video/mp4">
</video>

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

   * <img src="./img/stratified_sampling.png" width="300" height="50"></img>

   * <img src="./img/nerf_non_continuous.png" width="350px"></img>

4. Hierarchical Volume Sampling - Coarse network -> Fine network

   * <img src="./img/volume_sampling.png" width="400" height="150"></img>

5. Positional Encoding

   * <img src="./img/positional_embedding.png" width="400" height="30"></img>

   * <img src="./img/layers.png" width="400"></img>

# Improvement
* [ ] Add more data
* [ ] autoamtically make llff data
* [ ] train more to make it more realistic
* [ ] build configuration
* [ ] modify the whole thing to easy to use at any data with single command