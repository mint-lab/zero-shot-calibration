import numpy as np
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mm', '--mile_meter', type=float, required=True)
    parser.add_argument('-img', '--image', type=str, required=True)
    args = parser.parse_args()

    mm = args.mile_meter
    image = args.image

    image = cv2.imread(image)
    Height, Width = image.shape[:2]

    mm2pixels = None

    

if __name__ == "__main__":
    main()