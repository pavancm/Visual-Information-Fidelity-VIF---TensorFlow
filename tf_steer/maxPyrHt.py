import numpy
import tensorflow as tf

def maxPyrHt(imsz, filtsz):
    ''' Compute maximum pyramid height for given image and filter sizes.
        Specifically: the number of corrDn operations that can be sequentially
        performed when subsampling by a factor of 2. '''

    if not isinstance(imsz, list) or not isinstance(filtsz, list):
        if imsz < filtsz:
            return 0
    else:
        if len(imsz) == 1:
            imsz = (imsz[0], 1)
        if len(filtsz) == 1:
            filtsz = (filtsz[0], 1)
        #if filtsz[1] == 1:  # new
        #    filtsz = (filtsz[1], filtsz[0])
        #if imsz[0] < filtsz[0] or imsz[1] < filtsz[1]:
        #    print 'flag 2'
        #    return 0

    if not isinstance(imsz, list) and not isinstance(filtsz, list):
        imsz = imsz
        filtsz = filtsz
        if imsz < filtsz:
            return 0
    elif 1 in imsz:         # 1D image
        imsz = imsz[0] * imsz[1]
        filtsz = filtsz[0] * filtsz[1]
        if imsz < filtsz:
            return 0
    #elif 1 in filtsz:   # 2D image, 1D filter
    else:   # 2D image
        #filtsz = (filtsz[0], filtsz[0])
        #print filtsz
        if ( imsz[0] < filtsz[0] or imsz[0] < filtsz[1] or
             imsz[1] < filtsz[0] or imsz[1] < filtsz[1] ):
            return 0

    if ( not isinstance(imsz, list) and not isinstance(filtsz, list) and
         imsz < filtsz ) :
        height = 0
    elif not isinstance(imsz, list) and not isinstance(filtsz, list):
        height = 1 + maxPyrHt( numpy.floor(imsz/2.0), filtsz )
    else:
        height = 1 + maxPyrHt( [numpy.floor(imsz[0]/2.0), 
                                numpy.floor(imsz[1]/2.0)], 
                               filtsz )

    return height
