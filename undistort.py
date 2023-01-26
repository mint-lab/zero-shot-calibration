import numpy as np
import cv2
from copy import deepcopy

if __name__ == '__main__':
    # d_image = cv2.imread('20210812_084000_000_0400.png')
    # d_image = cv2.imread('20210812_084000_000_0500.png')
    d_image = cv2.imread('20210812_084000_000_0600.png')
    # d_image = cv2.imread('20210812_084000_000_0700.png')
    focal = 1021.9253974248712
    height, width, channel = d_image.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2

    u_image = deepcopy(d_image)

    for y in range(height):
        for x in range(width):
            # calc normalized undistorted pixel
            x_nu = (x - cx) / focal
            y_nu = (y - cy) / focal

            # calc radial distortion coeff
            # theta = np.arccos(1 / np.sqrt((x_nu ** 2 + y_nu ** 2 + 1)))
            theta = np.arctan(np.sqrt(x_nu ** 2 + y_nu ** 2))

            ru = focal * theta
            rd = focal * np.arctan(ru / focal)

            # apply it to points
            x_nd = (rd / ru) * x_nu
            y_nd = (rd / ru) * y_nu

            # # apply it back to points
            x_pd = focal * x_nd + cx
            y_pd = focal * y_nd + cy

            u_image[y, x] = d_image[int(y_pd), int(x_pd)]

    stacked_image = np.vstack((u_image, d_image))
    resized_image = cv2.resize(stacked_image, dsize=(0, 0), fx=0.4, fy=0.4)

    cv2.imshow('compare image', resized_image)

    cv2.waitKey(0)

    cv2.imwrite('cam3_undistorted.png', stacked_image)