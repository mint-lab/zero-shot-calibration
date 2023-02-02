# zero-shot-calibration

The origin paper is from Jae-Yeong Lee, “Zero-Shot Calibration of Fisheye Cameras,”
arXiv, 2020, https://arxiv.org/abs/2011.14607

How to use this code is described below:

```shell
usage: zero_shot_calib.py [-h] [-W WIDTH] [-H HEIGHT] [-hf HFOV] [-vf VFOV] [-d DRAW]

find proper fisheye camera focal length with spec

options:
  -h, --help            show this help message and exit
  -W WIDTH, --Width WIDTH
                        Width information of image frame
  -H HEIGHT, --Height HEIGHT
                        Height information of image frame
  -hf HFOV, --Hfov HFOV
                        Horizontal field of view angle. It should be float
  -vf VFOV, --Vfov VFOV
                        Vertical field of view angle. It should be float
  -d DRAW, --draw DRAW  draw plot or not
```

example usage:
```shell
python3 zero_shot_calib.py -W 1920 -H 1440 -hf 122.0 -vf 94.0
```

Then it will show you the focal length in the terminal.

After that, if you want to undistort the image, you can use undistort.py

```shell
usage: undistort.py [-h] [--input_image INPUT_IMAGE] [--focal FOCAL] [--output_image OUTPUT_IMAGE]

options:
  -h, --help            show this help message and exit
  --input_image INPUT_IMAGE
  --focal FOCAL
  --output_image OUTPUT_IMAGE
```

example usage:
```shell
python3 undistort.py --input_image 20210812_084000_000_0400.png --output_image undistort.png --focal 1021.92
```

then it will save the undistorted image.

Example result:

 
<figure class="half">
  <img src="20210812_084000_000_0400_undistort.png" align="left"  width = "47%" height = "47%">
  <img src="20210812_084000_000_0500_undistort.png" align="right"  width = "47%" height = "47%">
</figure>

<figure class="half">
  <img src="20210812_084000_000_0600_undistort.png" align="left"  width = "47%" height = "47%">
  <img src="20210812_084000_000_0700_undistort.png" align="right"  width = "47%" height = "47%">
</figure>
