import numpy as np
from PIL import Image
from numpy import linalg as la


def img_to_matrix(im):
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float')
    # new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data, (height, width))
    return new_data


def matrix_to_img(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


def compress(filename, percent=0.3):
    im = Image.open(filename)
    a = img_to_matrix(im)

    u, sigma, vt = la.svd(a)
    s = np.zeros([u.shape[0], vt.shape[0]])

    for i in range(int(len(sigma)*percent)):
        s[i][i] = sigma[i]
    print(s)
    compressed_a = np.dot(np.dot(u, s), vt)

    return matrix_to_img(compressed_a)


if __name__ == "__main__":

    img = compress("a.png", percent=0.9)
    img.show()
