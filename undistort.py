import argparse
import numpy as np
import cv2
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

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
    parser.add_argument('--input_image', type=str, default='20210812_084000_000_0000.png')
    parser.add_argument('--focal', type=float, default=1021.9253974248712)
    parser.add_argument('--output_image', type=str, default='output.png')
    args = parser.parse_args()

    image_path = args.input_image
    focal = args.focal
    output_image = args.output_image

    d_image = cv2.imread(image_path)
    height, width, _ = d_image.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2

    u_image = deepcopy(d_image)

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

            u_image[y, x] = d_image[int(y_pd), int(x_pd)]

    stacked_image = np.vstack((u_image, d_image))
    resized_image = cv2.resize(stacked_image, dsize=(0, 0), fx=0.4, fy=0.4)

    cv2.imwrite(output_image, stacked_image)