#Thanks to https://github.com/suzusuzu
import numpy as np
def moving_window_matrix(x,window_size):
    n = x.shape[0]
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n-window_size+1, window_size), strides=(stride,stride) ).copy()

def hsvd(x, window, rank):
    m = moving_window_matrix(x, window)
    u, s, vh = np.linalg.svd(m)
    h = u[:,:rank] @ np.diag(s[:rank]) @ vh[:rank,:]
    c = h[0,:]
    c = np.append(c, h[1:,-1])
    return c, x-c
