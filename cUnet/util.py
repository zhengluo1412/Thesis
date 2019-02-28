import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os


def load_img_cond(img_dir, cond_file, batch_size, img_shape=(512, 512, 3)):
    """
    batch training data generator
    :param img_dir: directory that saves the training images
    :param cond_file: csv file that saves the light positions
    :param batch_size:
    :param img_shape:
    :return:
    """
    img_list = sorted(os.listdir(img_dir))
    num_img = len(img_list)
    cond_ndarray = pd.read_csv(cond_file, header=None, dtype=np.float32).as_matrix()
    for ndx in range(0, num_img, batch_size):
        batch_list = img_list[ndx:min(ndx + batch_size, num_img)]
        batch_img = np.empty((len(batch_list), *(img_shape)))
        for i, img in enumerate(batch_list):
            path = img_dir + img
            batch_img[i] = mpimg.imread(path)

        batch_cond = cond_ndarray[ndx:min(ndx + batch_size, num_img)]
        yield batch_img, batch_cond

