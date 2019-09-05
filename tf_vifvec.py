import numpy as np
import tensorflow as tf
from tf_steer.Spyr import Spyr
import math
from tf_steer.corrDn import corrDn
import tensorflow_probability as tfp

def vifvec(imref, imdist):
    M = 3
    subbands = [4, 7, 10, 13, 16, 19, 22, 25]
    
    sigma_nsq = 0.4
    
    #Wavelet Decomposition
    pyr = Spyr(imref, 4, 'sp5Filters', 'reflect1')
    org = pyr.pyr[::-1]     #reverse list
    
    pyr = Spyr(imdist, 4, 'sp5Filters', 'reflect1')
    dist = pyr.pyr[::-1]
    
    #Calculate parameters of the distortion channel
    g_all, vv_all = vif_sub_est_M(org, dist, subbands, M)
    
    #calculate the parameters of reference
    l_arr, ssarr, cu_arr = refparams_vecgsm(org, subbands, M)
    
    num, den = [],[]
    
    for i in range(len(subbands)):
        sub = subbands[i]
        g = g_all[i]
        vv = vv_all[i]
        ss = ssarr[i]
        lam = l_arr[i]
        #cu = cuarr[i]
        
        #neigvals = len(lam)
        lev = math.ceil((sub - 1)/6)
        winsize = 2**lev + 1
        offset = (winsize - 1)/2
        offset = math.ceil(offset/M)
        
        g = g[:,offset:g.get_shape().as_list()[1]-offset,offset:g.get_shape().as_list()[2]-offset]
        vv = vv[:,offset:vv.get_shape().as_list()[1]-offset,offset:vv.get_shape().as_list()[2]-offset]
        ss = ss[:,offset:ss.get_shape().as_list()[1]-offset,offset:ss.get_shape().as_list()[2]-offset]
        
        temp1,temp2 = tf.constant(0, dtype=tf.float64),tf.constant(0, dtype=tf.float64)
        rt=[]
        for j in range(lam.get_shape().as_list()[1]):
            temp1 += tf.reduce_sum(tf.log(1 + tf.math.divide(tf.multiply(tf.multiply(g,g),ss) * lam[:,j,tf.newaxis,tf.newaxis]\
                                                             , vv + sigma_nsq)), axis=[1,2]) #distorted image information
            temp2 += tf.reduce_sum(tf.log(1 + tf.divide(ss * lam[:,j,tf.newaxis,tf.newaxis], sigma_nsq)), axis=[1,2]) #reference image information
            rt.append(tf.reduce_sum(tf.log(1 + tf.divide(ss * lam[:,j,tf.newaxis,tf.newaxis], sigma_nsq)), axis=[1,2]))
        
        num.append(temp1)
        den.append(temp2)
#    
    num = tf.convert_to_tensor(num, dtype = tf.float64)
    den = tf.convert_to_tensor(den, dtype = tf.float64)
    vif = tf.reduce_sum(num, axis=0)/tf.reduce_sum(den, axis=0)
    
    vif = tf.where(tf.is_nan(vif), tf.zeros_like(vif), vif)
    return vif
        

def vif_sub_est_M(org, dist, subbands, M):
    tol = 1e-15         #tolerance for zero variance
    g_all = []
    vv_all = []
    
    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub-1]
        yn = dist[sub-1]
        
        #size of window used in distortion channel estimation
        lev = math.ceil((sub - 1)/6)
        winsize = 2**lev + 1
        win = np.ones([winsize, winsize]).astype(float)
        win_av = win/np.sum(win)
        win_sum = np.sum(win)
        win = tf.constant(win, dtype = tf.float64)
        win_av = tf.constant(win_av, dtype = tf.float64)
        
        #force subband to be a multiple of M
        newsize = [math.floor(y.get_shape().as_list()[1]/M) * M, math.floor(y.get_shape().as_list()[2]/M) * M]
        y = y[:,:newsize[0],:newsize[1],:]
        yn = yn[:,:newsize[0],:newsize[1],:]
        
        #correlation with downsampling
        winstep = (M, M)
        winstart = (math.floor(M/2) ,math.floor(M/2))
        winstop = (y.get_shape().as_list()[1] - math.ceil(M/2) + 1, y.get_shape().as_list()[2] - math.ceil(M/2) + 1)
        
        #mean
        mean_x = corrDn(y, win_av, 'reflect1', winstep, winstart, winstop)
        mean_y = corrDn(yn, win_av, 'reflect1', winstep, winstart, winstop)
        
        #covariance
        cov_xy = corrDn(tf.multiply(y, yn), win, 'reflect1', winstep, winstart, winstop) - \
        win_sum * tf.multiply(mean_x,mean_y)
        
        #variance
        ss_x = corrDn(tf.multiply(y,y), win, 'reflect1', winstep, winstart, winstop) - win_sum * tf.multiply(mean_x,mean_x)
        ss_y = corrDn(tf.multiply(yn,yn), win, 'reflect1', winstep, winstart, winstop) - win_sum * tf.multiply(mean_y, mean_y)
        
        ss_x = tf.cast(ss_x > 0, ss_x.dtype) * ss_x
        ss_y = tf.cast(ss_y > 0, ss_y.dtype) * ss_y
        
        #Regression
        g = tf.math.divide(cov_xy,(ss_x + tol))
        
        vv = (ss_y - tf.multiply(g, cov_xy))/(win_sum)
        
        #taking care of NaN cases
        g = tf.where(ss_x < tol, tf.zeros_like(g), g)
        vv = tf.where(ss_x < tol, ss_y, vv)
        ss_x = tf.where(ss_x < tol, tf.zeros_like(ss_x), ss_x)
        
        g = tf.where(ss_y < tol, tf.zeros_like(g), g)
        vv = tf.where(ss_y < tol, tf.zeros_like(vv),vv)
        
        g = tf.cast(g > 0, g.dtype) * g
        vv = tf.where(g < 0, ss_y, vv)
        
        vv = tf.where(vv <= tol, tol * tf.ones_like(vv), vv)
#        
        g = tf.squeeze(g, axis = 3)
        vv = tf.squeeze(vv, axis = 3)
        
        g_all.append(g)
        vv_all.append(vv)
    
    return g_all, vv_all

def refparams_vecgsm(org, subbands, M):
    # This function caluclates the parameters of the reference image
    
    l_arr, ssarr, cu_arr = [],[],[]
    
    for i in range(len(subbands)):
        sub = subbands[i]
        y = org[sub-1]
        
        sizey = (math.floor(y.get_shape().as_list()[1]/M)*M, math.floor(y.get_shape().as_list()[2]/M)*M)
        y = y[:,:sizey[0],:sizey[1],:]
        
        #Collect MxM blocks, rearrange into M^2 dimensional vector
        temp = []
        for j in range(M):
            for k in range(M):
                pt = tf.squeeze(y[:,k:y.get_shape().as_list()[1]-M+k+1,j:y.get_shape().as_list()[2]-M+j+1,:], axis = 3)
                pt = tf.einsum('aij->aji',pt)
                pt = tf.reshape(pt,[pt.get_shape().as_list()[0], -1])
                temp.append(pt)
        
        temp = tf.convert_to_tensor(temp, dtype=tf.float64)
        temp = tf.einsum('ijk->jik',temp)
        mcu = tf.reduce_mean(temp, axis=2)
        mean_sub = temp - tf.reshape(tf.tile(mcu, (1, temp.get_shape().as_list()[2])), temp.get_shape().as_list())
        mean_sub_T = tf.einsum('aij->aji',mean_sub)
        cu = mean_sub @ mean_sub_T / temp.get_shape().as_list()[2]
        
##        #Calculate S field, non-overlapping blocks
        temp = []
        for j in range(M):
            for k in range(M):
                pt = tf.squeeze(y[:,k::M,j::M,:], axis = 3)
                pt = tf.einsum('aij->aji',pt)
                pt = tf.reshape(pt, [pt.get_shape().as_list()[0], -1])
                temp.append(pt)
        
        temp = tf.convert_to_tensor(temp, dtype=tf.float64)
        temp = tf.einsum('ijk->jik',temp)
        
        ss = tfp.math.pinv(cu) @ temp
        ss = tf.reduce_sum(tf.multiply(ss,temp),axis=1)/(M**2)
        ss = tf.reshape(ss, [temp.get_shape().as_list()[0], int(sizey[1]/M), int(sizey[0]/M)])
        ss = tf.einsum('aij->aji',ss)
        
        d = tf.linalg.eigvalsh(cu)
        l_arr.append(d)
        ssarr.append(ss)
        cu_arr.append(mcu)
    
    return l_arr, ssarr, cu_arr