import tensorflow as tf
import math
import numpy as np

def rconv2(a,b, ctr = 0):
    
    """ RES = RCONV2(MTX1, MTX2, CTR)

    Convolution of two matrices, with boundaries handled via reflection
%   about the edge pixels.  Result will be of size of LARGER matrix.
% 
%   The origin of the smaller matrix is assumed to be its center.
%   For even dimensions, the origin is determined by the CTR (optional) 
%   argument:
%       CTR   origin
%       0     DIM/2      (default)
%       1     (DIM/2)+1  

%   Eero Simoncelli, 6/96."""

    if (a.get_shape().as_list()[1] >= b.get_shape().as_list()[0]) and (a.get_shape().as_list()[2] >= b.get_shape().as_list()[1]):
        large, small = a, b
    elif (a.get_shape().as_list()[1] <= b.get_shape().as_list()[0]) and (a.get_shape().as_list()[2] <= b.get_shape().as_list()[1]):
        large, small = b, a
    else:
        raise Exception("one arg must be larger than the other in both dimensions!")
    
    ly = large.get_shape().as_list()[1]
    lx = large.get_shape().as_list()[2]
    sy = small.get_shape().as_list()[0]
    sx = small.get_shape().as_list()[1]
    
    sy2 = math.floor((sy+ctr-1)/2)
    sx2 = math.floor((sx+ctr-1)/2)
    
    ty_top = np.arange(sy-sy2-1,0,-1)
    tx_top = np.arange(sx-sx2-1,0,-1)
    ty_bottom = np.arange(ly-2,ly-sy2-2,-1)
    tx_bottom = np.arange(lx-2,lx-sx2-2,-1)
    
    paddings = tf.constant([[0,0], [sy-sy2-1,sy2], [sx-sx2-1, sx2], [0,0]])
    clarge = tf.pad(large, paddings, mode = 'REFLECT')
    
    #clarge = tf.reshape(clarge,[clarge.get_shape().as_list()[0], clarge.get_shape().as_list()[1], clarge.get_shape().as_list()[2]])
    small = tf.reshape(small, [small.get_shape().as_list()[0], small.get_shape().as_list()[1], 1, 1])
    c = tf.nn.conv2d(clarge, small, padding="VALID", strides = [1, 1, 1, 1])
    return c