#!/usr/bin/env python3.6

import numpy as np
import cv2
import argparse
from scipy import signal

def encode_image(raw_img):
    # Convert the image to the YCbCr color space.
    # Instead of red, green, blue color decompositions,
    # this becomes a luma, blue difference, and red difference component.
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2YCR_CB)
    y, cb, cr = cv2.split(img)

    img_height, img_width = y.shape

    # The human eye is less sensitive to small changes in luminance
    # than it is to small changes in intensity.
    # We can use this to our advantage by downsampling the luminance
    # information without much loss to image quality. This reduces image size.
    # Perform downsampling via average pooling and strided convolution.
    downsample_factor = 2
    dim = downsample_factor
    downsample_kernel = 1.0 / (dim**2) * np.ones((dim, dim))
    y_blurred = signal.convolve2d(y, downsample_kernel, mode='valid')
    y_downsampled = y_blurred[::downsample_factor, ::downsample_factor]

    # Encode each 8x8 patch in the image with a discrete cosine transform.
    # Because human eyes are not as sensitive to high frequency color intensity
    # change information, we can try to sparsify our representation of the image
    # in the DCT domain by truncating certain frequency components without much
    # noticable loss in image quality.
    patch_dim = 8
    channel_coeffs = [[], [], []]
    for channel in [y_downsampled, cb, cr]:
        channel_height, channel_width = channel.shape
        row_range = range(0, channel_height - patch_dim, patch_dim)
        col_range = range(0, channel_width - patch_dim, patch_dim)
        for r_start, c_start in zip(row_range, col_range):
            r_end = r_start + patch_dim
            c_end = c_start + patch_dim
            patch = channel[r_start : r_end, c_start : c_end]

    return None

def decode_image(message):
    return None

def main(args):
    # Read in the desired image file in RGB
    raw_img = cv2.imread(args.image, cv2.IMREAD_COLOR)

    # Encode the image for compressed transfer
    message = encode_image(raw_img)

    # Decode image for viewing
    img_decoded = decode_image(message)

    assert img_decoded is not None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to run DFT on the input image")

    parser.add_argument("--image", type=str, required=True,
            help="The image filename to load for compression")

    args = parser.parse_args()

    main(args)

