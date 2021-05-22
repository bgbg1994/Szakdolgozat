import numpy as np
import os
import cv2
import subprocess

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

    print("end")
    return colormap
def run_colorization(out_image_name,out_image_name_v):
    dir_path = 'current'
    original = cv2.imread(os.path.join(dir_path, 'image.bmp'))
    marked = cv2.imread(os.path.join(dir_path, 'image_marked.bmp'))
    original = original.astype(float) / 255
    marked = marked.astype(float) / 255
    YUV = np.zeros(original.shape)
    n = YUV.shape[0]
    m = YUV.shape[1]
    test=get_colormap(original, marked,n,m,out_image_name)
    cv2.imwrite(out_image_name_v, test * 255)
    DETACHED_PROCESS = 8
    subprocess.Popen('color.exe', creationflags=DETACHED_PROCESS, close_fds=True)
