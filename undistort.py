import argparse
import numpy as np
import cv2
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import json

def zshot_undistort_pixels(pixels: np.ndarray,
                        K: np.ndarray)-> np.ndarray:
    inv_K = np.linalg.inv(K)
    # (n, 2) -> (n, 3)
    pixels = np.array([[px[0], px[1], 1] for px in pixels], dtype=np.float32)

    # distorted pixel -> normed image plane
    d_pts = pixels @ inv_K.T
    d_pts = np.array([[pt[0] / pt[2], pt[1] / pt[2], 1] for pt in d_pts], dtype=np.float32)

    # calc distort
    r_d = np.sqrt(d_pts[:, 0] ** 2 + d_pts[:, 1] ** 2)
    r_u = np.tan(r_d)

    u_pts = (r_u / r_d)[:, np.newaxis] * d_pts
    u_pts = np.array([[pt[0], pt[1], 1] for pt in u_pts], dtype=np.float32)
    u_pixels = u_pts @ K.T

    return u_pixels[:, :2]

def zshot_undistort_image(img: np.ndarray,
                          focal: float)-> np.ndarray:
    height, width, _ = img.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2

    u_image = deepcopy(img)

    for y in range(height):
        for x in range(width):
            # calc normalized undistorted pixel, (x, y) is not distorted pixels!
            x_nu = (x - cx) / focal
            y_nu = (y - cy) / focal

            # calc radial distortion coeff
            ru = np.sqrt(x_nu ** 2 + y_nu ** 2) # tan(theta) = np.sqrt(x_nu ** 2 + y_nu ** 2)
            rd = np.arctan(ru)

            # apply it to points
            x_nd = (rd / ru) * x_nu
            y_nd = (rd / ru) * y_nu

            # # apply it back to points
            x_pd = focal * x_nd + cx
            y_pd = focal * y_nd + cy

            u_image[y, x] = img[int(y_pd), int(x_pd)]
    
    return u_image

def zshot_distort_points(pts: np.ndarray,
                         K: np.ndarray,
                         rvec: np.ndarray = np.zeros((3,), dtype=np.float32),
                         tvec: np.ndarray = np.zeros((3,), dtype=np.float32))-> np.ndarray:
    # from world to camera
    pts = R.from_rotvec(rvec).apply(pts) + tvec
    
    # from camera to normed image plane
    pts = np.array([[pt[0] / pt[2], pt[1] / pt[2], 1] for pt in pts], dtype=np.float32)

    # apply distort
    ru = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    rd = np.arctan(ru)

    # apply it to points
    pts *= (rd / ru)[:, np.newaxis]
    pts = np.array([[pt[0] / pt[2], pt[1] / pt[2], 1] for pt in pts], dtype=np.float32)
    
    # project to pixel plane
    return (pts @ K.T)[:, :2] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='20210812_084000_000_0400.png')
    parser.add_argument('--focal', type=float, default=1021.9253974248712)
    parser.add_argument('--output_image', type=str, default='output.png')
    args = parser.parse_args()

    image_path = args.input_image
    focal = args.focal
    output_image = args.output_image

    # Get distorted image information
    d_image = cv2.imread(image_path)
    Height, Width, _ = d_image.shape
    cx, cy = (Width - 1) / 2, (Height - 1) / 2

    # # set intrinsic
    # K = np.array([
    #     [focal, 0, cx],
    #     [0, focal, cy],
    #     [0, 0, 1]
    # ], dtype=np.float32)

    # with open('ETRI_44markers_images.json', 'r') as f:
    #     tmp_list = json.load(f)
    
    # for tmp in tmp_list:
    #     if tmp['file'] == image_path:
    #         d_pixel = np.array(tmp['idx_pixels'])[:, 1:]
    
    # draw circle in distorted image
    # for d_px in d_pixel:
    #     d_image = cv2.circle(d_image, (int(d_px[0]), int(d_px[1])), 10, (0, 255, 0), -1)

    # get undistorted image
    u_image = zshot_undistort_image(d_image, focal)
    # u_pixel = zshot_undistort_pixels(d_pixel, K)

    # draw pixels
    # for u_px in u_pixel:
    #     u_image = cv2.circle(u_image, (int(u_px[0]), int(u_px[1])), 10, (0, 255, 0), -1)


    # cv2.imwrite(output_image, u_image)
    # cv2.imwrite('distorted_image.png', d_image)
    cv2.imwrite(image_path[:-4] + '_undistort' + image_path[-4:], u_image)

    # cv2.waitKey(0)