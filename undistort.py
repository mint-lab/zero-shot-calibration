import argparse
import numpy as np
import cv2
from copy import deepcopy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='20210812_084000_000_0000.png')
    parser.add_argument('--focal', type=float, default=1021.9253974248712)
    parser.add_argument('--output_image', type=str, default='output.png')
    args = parser.parse_args()

    image_path = args.image
    focal = args.focal
    output_image = args.output_image

    d_image = cv2.imread(image_path)
    height, width, _ = d_image.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2

    u_image = deepcopy(d_image)

    for y in range(height):
        for x in range(width):
            # calc normalized undistorted pixel
            x_nu = (x - cx) / focal
            y_nu = (y - cy) / focal

            # calc radial distortion coeff
            # theta = np.arctan(np.sqrt(x_nu ** 2 + y_nu ** 2))
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

    # cv2.imshow('compare image', resized_image)
    # cv2.waitKey(0)

    cv2.imwrite(output_image, stacked_image)