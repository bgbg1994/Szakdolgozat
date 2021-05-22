from math import log10, sqrt
import cv2
import numpy as np
import os

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    dir_path = 'current'
    original = cv2.imread(os.path.join(dir_path, 'image.bmp'))
    dir_path = 'done'
    compressed = cv2.imread(os.path.join(dir_path, 'imagefirst_alg_res.bmp'))
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")


if __name__ == "__main__":
    main()