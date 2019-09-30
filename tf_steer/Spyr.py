import tensorflow as tf
from tf_steer.sp0Filters import sp0Filters
from tf_steer.sp1Filters import sp1Filters
from tf_steer.sp3Filters import sp3Filters
from tf_steer.sp5Filters import sp5Filters
from tf_steer.pyramid import pyramid
from tf_steer.maxPyrHt import maxPyrHt
from tf_steer.corrDn import corrDn
import math
import os

class Spyr(pyramid):
    filt = ''
    edges = ''
    
    #constructor
    def __init__(self, image, height='auto', filter='sp1Filters', edges='reflect1'):
        """Steerable pyramid. image parameter is required, others are optional
        
        - `image` - a 2D numpy array
  
        - `height` - an integer denoting number of pyramid levels desired.  'auto' (default) uses
        maxPyrHt from pyPyrUtils.

        - `filter` - The name of one of the steerable pyramid filters in pyPyrUtils:
        `'sp0Filters'`, `'sp1Filters'`, `'sp3Filters'`, `'sp5Filters'`.  Default is `'sp1Filters'`.
    
        - `edges` - specifies edge-handling.  Options are:
           * `'circular'` - circular convolution
           * `'reflect1'` - reflect about the edge pixels
           * `'reflect2'` - reflect, doubling the edge pixels
           * `'repeat'` - repeat the edge pixels
           * `'zero'` - assume values of zero outside image boundary
           * `'extend'` - reflect and invert
           * `'dont-compute'` - zero output when filter overhangs input boundaries.
        """
        self.pyrType = 'steerable'
        self.image = image

        if filter == 'sp0Filters':
            filters = sp0Filters()
        elif filter == 'sp1Filters':
            filters = sp1Filters()
        elif filter == 'sp3Filters':
            filters = sp3Filters()
        elif filter == 'sp5Filters':
            filters = sp5Filters()
        elif os.path.isfile(filter):
            raise Exception("Filter files not supported yet")
        else:
            raise Exception("filter parameters value %s not supported" % (filter))
        self.filt = filters

        self.edges = edges

        harmonics = filters['harmonics']
        lo0filt = filters['lo0filt']
        hi0filt = filters['hi0filt']
        lofilt = filters['lofilt']
        bfilts = filters['bfilts']
        steermtx = filters['mtx']
        
        im = self.image
        im_sz = im.get_shape().as_list()
        
        lofilt_sz = lofilt.get_shape().as_list()
        
        max_ht = maxPyrHt([im_sz[1], im_sz[2]], [lofilt_sz[0], lofilt_sz[1]])
        if height == 'auto':
            ht = max_ht
        elif height > max_ht:
            raise Exception("cannot build pyramid higher than %d levels." % (max_ht))
        else:
            ht = height

        nbands = bfilts.get_shape().as_list()[1]

        self.pyr = []
        self.pyrSize = []
        for n in range((ht*nbands)+2):
            self.pyr.append([])
            self.pyrSize.append([])

        
        pyrCtr = 0

        hi0 = corrDn(image = im, filt = hi0filt, edges = edges);

        self.pyr[pyrCtr] = hi0
        self.pyrSize[pyrCtr] = hi0.get_shape().as_list()

        pyrCtr += 1

        lo = corrDn(image = im, filt = lo0filt, edges = edges)
        for i in range(ht):
            lo_sz = [lo.get_shape().as_list()[1], lo.get_shape().as_list()[1]]
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(math.floor(math.sqrt(bfilts.get_shape().as_list()[0])))

            for b in range(bfilts.get_shape().as_list()[1]):
                filt = tf.gather(bfilts, [b], axis=1)
                #filt = tf.reshape(tf.slice(bfilts,[0,0],[bfilts.get_shape().as_list()[0],0]),[bfilts.get_shape().as_list()[0]])
                filt = tf.transpose(tf.reshape(filt, [bfiltsz,bfiltsz]))
                band = corrDn(image = lo, filt = filt, edges = edges)
                self.pyr[pyrCtr] = band
                self.pyrSize[pyrCtr] = (band.get_shape().as_list()[0], band.get_shape().as_list()[1])
                pyrCtr += 1

            lo = corrDn(image = lo, filt = lofilt, edges = edges, step = (2,2))

        self.pyr[pyrCtr] = lo
        self.pyrSize[pyrCtr] = lo.get_shape().as_list()