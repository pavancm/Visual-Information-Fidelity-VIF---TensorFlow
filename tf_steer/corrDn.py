import numpy
import ctypes
import tensorflow as tf
from tf_steer.rconv2 import rconv2

def corrDn(image, filt, edges='reflect1', step=(1, 1), start=(0, 0), stop=None, result=None):
    """Compute correlation of matrices image with `filt, followed by downsampling.  

    These arguments should be 1D or 2D matrices, and image must be larger (in both dimensions) than
    filt.  The origin of filt is assumed to be floor(size(filt)/2)+1.
 
    edges is a string determining boundary handling:
      'circular' - Circular convolution
      'reflect1' - Reflect about the edge pixels
      'reflect2' - Reflect, doubling the edge pixels
      'repeat'   - Repeat the edge pixels
      'zero'     - Assume values of zero outside image boundary
      'extend'   - Reflect and invert (continuous values and derivs)
      'dont-compute' - Zero output when filter overhangs input boundaries

    Downsampling factors are determined by step (optional, default=(1, 1)), which should be a
    2-tuple (y, x).
 
    The window over which the convolution occurs is specfied by start (optional, default=(0,0), and
    stop (optional, default=size(image)).
 
    NOTE: this operation corresponds to multiplication of a signal vector by a matrix whose rows
    contain copies of the filt shifted by multiples of step.  See `upConv` for the operation
    corresponding to the transpose of this matrix.

    WARNING: if both the image and filter are 1d, they must be 1d in the same dimension. E.g., if
    image.shape is (1, 36), then filt.shape must be (1, 5) and NOT (5, 1). If they're both 1d and
    1d in different dimensions, then this may encounter a segfault. I've not been able to find a
    way to avoid that within this function (simply reshaping it does not work).
    """
    
    if len(filt.get_shape().as_list()) == 1:
        filt = tf.reshape(filt, [1, len(filt.get_shape().as_list())])

    if stop is None:
        stop = (image.get_shape().as_list()[1], image.get_shape().as_list()[2])
    
    #filt = tf.reverse(filt, [0,1])
    temp = rconv2(image, filt)
    
    temp = temp[:,start[0]:stop[0]:step[0],start[1]:stop[1]:step[1],:]
#    temp = tf.slice(temp, [0, start[0], start[1], 0],[temp.get_shape().as_list()[0], stop[0]-start[0], stop[1] - start[1], temp.get_shape().as_list()[3]])
#    temp = tf.image.resize_nearest_neighbor(temp, size = (int(temp.get_shape().as_list()[1]/step[0]), int(temp.get_shape().as_list()[2]/step[1])))
    return temp
