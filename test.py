from PIL import Image
import numpy as np
import tensorflow as tf
from tf_vifvec import vifvec

def tf_inverse_image(img):
    img = (img + 0.5) * 255.
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=255)
    img = img[...,::-1]
    return img


#Number of images to calculate VIF score. If VIF is required for more than 1 image, it can be vectorized to make it faster
num_im = 1

imref1 = np.array(Image.open('bikes.bmp')).astype(float)
imref = np.reshape(imref1, [num_im, imref1.shape[0], imref1.shape[1], imref1.shape[2]])

imdist1 = np.array(Image.open('img29.bmp')).astype(float)
imdist = np.reshape(imdist1, [num_im, imdist1.shape[0], imdist1.shape[1], imdist1.shape[2]])

batches,rows,cols,ch = imref.shape

x = tf.placeholder(tf.float32, shape=[batches, rows, cols, ch], name='x')
y = tf.placeholder(tf.float32, shape=[batches, rows, cols, ch], name='y')

x_rgb = tf.stack([tf_inverse_image(x[i,:]) for i in range(batches)])
y_rgb = tf.stack([tf_inverse_image(y[i,:]) for i in range(batches)])

x_gray = tf.round(tf.image.rgb_to_grayscale(x_rgb))
y_gray = tf.round(tf.image.rgb_to_grayscale(y_rgb))


with tf.Session() as sess:

    x_gray = tf.cast(x_gray, dtype=tf.float64)
    y_gray = tf.cast(y_gray, dtype=tf.float64)

    pyr = vifvec(x_gray, y_gray)
    
    temp = sess.run(pyr, feed_dict = {x: imref, y: imdist})