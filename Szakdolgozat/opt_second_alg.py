import numpy as np

from scipy import sparse
from scipy.sparse import linalg

import os

import sys
import cv2

def get_colormap(original,marked,n,m,out_image_name):
    print("start")
    colormap = np.zeros(original.shape)


    for j in range(m):
        for i in range(n):

            if(original[i,j,0]==marked[i,j,0] and original[i,j,1]==marked[i,j,1] and original[i,j,2]==marked[i,j,2]):
                colormap[i,j,0]=0
                colormap[i, j, 1] = 0
                colormap[i, j, 2] = 0
            else:
                colormap[i, j, 0] =marked[i,j,0]
                colormap[i, j, 1] = marked[i,j,1]
                colormap[i, j, 2] = marked[i,j,2]

    cv2.imwrite(out_image_name, asdasd * 255)

    print("end")
    return colormap

def rgb_to_yuv(r, g, b):
    y = 0.299*r + 0.587*g + 0.114*b
    u = -0.14713*r - 0.28886*g +  0.436*b
    v = 0.615*r - 0.51498*g -0.10001*b
    return (y, u, v)

def yuv_to_rgb(y, u, v):
    r = y  +  1.13983*v
    g = y - 0.39465*u - 0.58060*v
    b = y + 2.03211*u
    return (r, g, b)

def run_colorization(out_image_name):
    dir_path = 'current'
    original = cv2.imread(os.path.join(dir_path, 'image.bmp'))
    marked = cv2.imread(os.path.join(dir_path, 'image_marked.bmp'))
    original = original.astype(float) / 255
    marked = marked.astype(float) / 255
    isColored = abs(original - marked).sum(2) > 0.01
    (Y, _, _) = rgb_to_yuv(original[:, :, 0], original[:, :, 1], original[:, :, 2])
    (_, U, V) = rgb_to_yuv(marked[:, :, 0], marked[:, :, 1], marked[:, :, 2])
    YUV = np.zeros(original.shape)
    YUV[:, :, 0] = Y
    YUV[:, :, 1] = U
    YUV[:, :, 2] = V
    n = YUV.shape[0]
    m = YUV.shape[1]
    image_size = n * m
    indices_matrix = np.arange(image_size).reshape(n, m, order='F').copy()
    wd = 1
    nr_of_px_in_wd = (2 * wd + 1) ** 2
    max_nr = image_size * nr_of_px_in_wd
    row_inds = np.zeros(max_nr, dtype=np.int64)
    col_inds = np.zeros(max_nr, dtype=np.int64)
    vals = np.zeros(max_nr)



    print("Iteration")
    length = 0
    pixel_nr = 0
    for j in range(m):
        for i in range(n):
            if (not isColored[i, j]):
                window_index = 0
                window_vals = np.zeros(nr_of_px_in_wd)
                for ii in range(max(0, i - wd), min(i + wd + 1, n)):
                    for jj in range(max(0, j - wd), min(j + wd + 1, m)):
                        if (ii != i or jj != j):
                            row_inds[length] = pixel_nr
                            col_inds[length] = indices_matrix[ii, jj]
                            window_vals[window_index] = YUV[ii, jj, 0]
                            length += 1
                            window_index += 1
                center = YUV[i, j, 0].copy()
                window_vals[window_index] = center
                variance = np.mean(
                    (window_vals[0:window_index + 1] - np.mean(window_vals[0:window_index + 1])) ** 2)  # variance as c_var
                sigma = variance * 0.6
                mgv = min((window_vals[0:window_index + 1] - center) ** 2)
                if (sigma < (-mgv / np.log(0.01))):
                    sigma = -mgv / np.log(0.01)
                if (sigma < 0.000002):
                    sigma = 0.000002
                window_vals[0:window_index] = np.exp(
                    -((window_vals[0:window_index] - center) ** 2) / sigma)
                window_vals[0:window_index] = window_vals[0:window_index] / np.sum(
                    window_vals[0:window_index])
                vals[length - window_index:length] = -window_vals[0:window_index]
            row_inds[length] = pixel_nr
            col_inds[length] = indices_matrix[i, j]
            vals[length] = 1
            length += 1
            pixel_nr += 1


    print("After Iteration Process")
    vals = vals[0:length]
    col_inds = col_inds[0:length]
    row_inds = row_inds[0:length]
    A = sparse.csr_matrix((vals, (row_inds, col_inds)), (pixel_nr, image_size))

    b = np.zeros((A.shape[0]))
    colorized = np.zeros(YUV.shape)
    colorized[:, :, 0] = YUV[:, :, 0]
    color_copy_for_nonzero = isColored.reshape(image_size,
                                               order='F').copy()
    colored_inds = np.nonzero(color_copy_for_nonzero)
    for t in [1, 2]:
        curIm = YUV[:, :, t].reshape(image_size, order='F').copy()
        b[colored_inds] = curIm[colored_inds]
        new_vals = linalg.spsolve(A,b)
        colorized[:, :, t] = new_vals.reshape(n, m, order='F')
    print("Back to RGB")
    (R, G, B) = yuv_to_rgb(colorized[:, :, 0], colorized[:, :, 1], colorized[:, :, 2])
    colorizedRGB = np.zeros(colorized.shape)
    colorizedRGB[:, :, 0] = R
    colorizedRGB[:, :, 1] = G
    colorizedRGB[:, :, 2] = B
    cv2.imwrite(out_image_name, colorizedRGB * 255)
    cv2.imshow('Second Colorization using Optimization',colorizedRGB)


