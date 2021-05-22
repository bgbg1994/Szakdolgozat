
import numpy as np
import cv2
import scipy.sparse as sps
from scipy.sparse.linalg.dsolve import linsolve




def normalizeImage(img, dtype=np.uint8):
    out = np.copy(img)
    min = np.amin(out, axis=(0,1))
    out -= min
    max = np.amax(out, axis=(0,1))
    out /= max
    out *= 255
    return out.astype(dtype)

def show_img(image, windowname = "image", scale = 3):
    shape = image.shape
    sizeX = int(shape[0]*scale)
    sizeY = int(shape[1]*scale)
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowname, sizeY, sizeX)
    cv2.imshow(windowname,image)
    cv2.waitKey(0)

def get_exp_weights(gvals, tlen):
    mean = np.mean(gvals[0:tlen+1])
    c_var = np.mean(np.square(gvals[0:tlen+1]-mean))
    csig = c_var*0.6
    mgv = np.mean(np.square(gvals[0:tlen]-gvals[tlen]))
    if (csig<(-mgv/np.log(0.01))):
        csig = -mgv/np.log(0.01)
    if (csig < 0.000002):
        csig = 0.000002
    gvals[0:tlen] = np.exp(- np.square(gvals[0:tlen]-gvals[tlen]) / csig)
    gvals[0:tlen] = -gvals[0:tlen]/np.sum(gvals[0:tlen])
    return gvals

def get_color_solve(mark_color_yiq, mark_binary, wd = 1):
    print("Colorize")
    shape = mark_color_yiq.shape
    n = shape[0]
    m = shape[1]
    img_size = n*m
    out_img_yiq = np.zeros(shape, dtype = np.float)
    out_img_yiq[:,:,0] = mark_color_yiq[:,:,0]
    indexM = np.arange(img_size).reshape((n,m), order = 'F')
    len = 0
    consts_len = 0
    col_inds = np.zeros(img_size*(2*wd+1)**2, dtype = np.int)
    row_inds = np.zeros(img_size*(2*wd+1)**2, dtype = np.int)
    vals = np.zeros(img_size*(2*wd+1)**2, dtype = np.float)
    gvals = np.zeros((2*wd+1)**2, dtype = np.float)
    A = sps.lil_matrix((img_size, img_size), dtype = np.float)
    bU = np.zeros((A.shape[0]), dtype = np.float)
    bV = np.zeros((A.shape[0]), dtype = np.float)

    for y in range(m):
        for x in range(n):
             if (mark_binary[x,y]):
                bU[consts_len] = 1*mark_color_yiq[x,y,1]
                bV[consts_len] = 1*mark_color_yiq[x,y,2]
                vals[len] = 1
             else:
                 tlen = 0
                 for i in range(max(0, x-wd), min(x+wd, n-1)+1):
                     for j in range(max(0, y-wd), min(y+wd,m-1)+1):
                         if( i!= x or j != y):
                             row_inds[len] = consts_len
                             col_inds[len] = indexM[i,j]
                             gvals[tlen] = mark_color_yiq[i,j,0]
                             len += 1
                             tlen += 1
                 gvals[tlen] = mark_color_yiq[x,y,0]
                 gvals = get_exp_weights(gvals, tlen)

                 vals[len-tlen:len] = gvals[0:tlen]
                 vals[len] = 1

             row_inds[len] = consts_len
             col_inds[len] = indexM[x,y]
             len += 1
             consts_len += 1

    for i in range(len):
        A[row_inds[i], col_inds[i]] = vals[i]

    A = A.tocsr()
    xU = linsolve.spsolve(A, bU)
    xV = linsolve.spsolve(A, bV)
    out_img_yiq[:,:,1] = xU.reshape((n,m), order = 'F')
    out_img_yiq[:,:,2] = xV.reshape((n,m), order = 'F')

    return out_img_yiq

def YIQ2BGR(img):
    img = np.copy(img.astype(np.float))
    res = np.copy(img.astype(np.float))
    M = np.array([[1.000, -0.647, 1.703],
                  [1.000, -0.272, -1.106],
                  [1.000,  0.956,  0.621]], dtype = np.float)
    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i,j,:] = np.dot(M, img[i,j,:])
    return res

def BGR2YIQ(img):
    img = np.copy(img.astype(np.float))
    res = np.copy(img.astype(np.float))
    M = np.array([[0.114, 0.587, 0.299],
                  [-0.322, -0.274, 0.596],
                  [0.312, -0.523, 0.211]], dtype = np.float)
    shape = img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i,j,:] = np.dot(M, img[i,j,:])
    return res

def get_gray_mask(input_image, marked_image, threshold = 0.01):
    print("Get gray mask")
    res = dict()
    input_image_yiq = BGR2YIQ(input_image)
    res["gray_image"] = np.copy(input_image_yiq[:,:,0])
    mark = abs(input_image - marked_image)
    mark = np.sum(mark, axis = 2)
    mark = mark > threshold
    res["kept_mask_binary_image"] = None
    marked_image_yiq = BGR2YIQ(marked_image)

    new_marked_image_yiq = np.copy(marked_image_yiq)
    new_marked_image_yiq[:,:,0] = input_image_yiq[:,:,0]

    shape = input_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (mark[i,j]):
                new_marked_image_yiq[i,j,1] = marked_image_yiq[i,j,1]
                new_marked_image_yiq[i,j,2] = marked_image_yiq[i,j,2]
            else:
                new_marked_image_yiq[i,j,1] = 0
                new_marked_image_yiq[i,j,2] = 0


    mark_intensity = YIQ2BGR(new_marked_image_yiq)

    res["marked_intensity"] = np.copy(mark_intensity)
    res["marked_image_yiq"] = np.copy(new_marked_image_yiq)
    res["mask_image"] = np.copy(mark.astype(float))
    res["mark"] = np.copy(mark)
    return res

def colorize(color_image_yiq, mark, iteration = 1, wd = 1):

    res = dict()
    gray_image = color_image_yiq[:,:,0]
    for i in range(iteration):
        color_image_yiq = get_color_solve(color_image_yiq, mark, wd = wd)
        mark = (1-mark).astype(bool)
        color_image_yiq[:,:,0] = gray_image
    res["color_image"] = np.copy(YIQ2BGR(color_image_yiq))
    res["color_image_yiq"] = np.copy(YIQ2BGR(color_image_yiq))
    return res

def main():
    return

if __name__ == "__main__":
    main()