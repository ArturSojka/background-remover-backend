cimport cython
from cython.parallel import prange
from libc.math cimport log2f, ceilf, fabsf, roundf, powf
import numpy as np

ctypedef fused number:
    Py_ssize_t
    float
    
cdef inline number clip(number x, number min_val, number max_val) noexcept nogil:
    return max(min_val, min(x, max_val))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void resize_nearest_gray(float[:,::1] src, float[:,::1] dst) noexcept:
    cdef:
        Py_ssize_t h_dst_max = dst.shape[0]
        Py_ssize_t w_dst_max = dst.shape[1]
        float scale_h = <float>src.shape[0] / <float>h_dst_max
        float scale_w = <float>src.shape[1] / <float>w_dst_max
        Py_ssize_t h_src, w_src, h_dst, w_dst

    for h_dst in prange(h_dst_max, nogil=True):
        h_src = <Py_ssize_t>(h_dst * scale_h)
        for w_dst in range(w_dst_max):
            w_src = <Py_ssize_t>(w_dst * scale_w)

            dst[h_dst, w_dst] = src[h_src, w_src]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void resize_nearest_rgb(float[:,:,::1] src, float[:,:,::1] dst) noexcept:
    cdef:
        Py_ssize_t h_dst_max = dst.shape[0]
        Py_ssize_t w_dst_max = dst.shape[1]
        Py_ssize_t depth = dst.shape[2]
        float scale_h = <float>src.shape[0] / <float>h_dst_max
        float scale_w = <float>src.shape[1] / <float>w_dst_max
        Py_ssize_t h_src, w_src, h_dst, w_dst, c

    for h_dst in prange(h_dst_max, nogil=True):
        h_src = <Py_ssize_t>(h_dst * scale_h)
        for w_dst in range(w_dst_max):
            w_src = <Py_ssize_t>(w_dst * scale_w)

            for c in range(depth):
                dst[h_dst, w_dst, c] = src[h_src, w_src, c]

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def estimate_foreground(
        float[:,:,::1] I,
        float[:,::1] alpha,
        const Py_ssize_t low_res_iter=10,
        const Py_ssize_t hi_res_iter=2,
        const Py_ssize_t low_size=32,
        const float omega=0.1,
        const float epsilon=5e-3
    ):
    cdef:
        Py_ssize_t rows = I.shape[0]
        Py_ssize_t cols = I.shape[1]
        Py_ssize_t depth = I.shape[2]
        float[:,:,::1] F = np.zeros((1,1,depth),dtype=np.float32)
        float[:,:,::1] F_tmp
        float[:,:,::1] B = np.zeros((1,1,depth),dtype=np.float32)
        float[:,:,::1] B_tmp
        float[:,::1] b = np.zeros((2, depth), dtype=np.float32)
        float[:,:,::1] I_resized
        float[:,::1] alpha_resized
        Py_ssize_t[4] h_neighbours = [1,0,0,-1]
        Py_ssize_t[4] w_neighbours = [0,-1,1,0]
        Py_ssize_t n = <Py_ssize_t>ceilf(log2f(<float>max(rows,cols)))
        Py_ssize_t layer, h, w, i, j, c, x, y, neighbour
        float a0, a1, A00, A01, A11, delta_alpha, inv_det, b00, b01, b11
        
    for layer in range(1,n+1):
        h = <Py_ssize_t>roundf(powf(<float>rows, (<float>layer)/(<float>n)))
        w = <Py_ssize_t>roundf(powf(<float>cols, (<float>layer)/(<float>n)))
        
        I_resized = np.empty((h,w,depth),dtype=np.float32)
        resize_nearest_rgb(I, I_resized)
        alpha_resized = np.empty((h,w),dtype=np.float32)
        resize_nearest_gray(alpha,alpha_resized)

        F_tmp = np.empty((h,w,depth),dtype=np.float32)
        resize_nearest_rgb(F,F_tmp)
        F = F_tmp
        B_tmp = np.empty((h,w,depth),dtype=np.float32)
        resize_nearest_rgb(B,B_tmp)
        B = B_tmp

        if h <= low_size and w <= low_size:
            n_iter = low_res_iter
        else:
            n_iter = hi_res_iter
            
        for _ in range(n_iter):
            for i in range(0,h):
                for j in range(0,w):
                    a0 = alpha_resized[i, j]
                    a1 = <float>1.0 - a0

                    A00 = a0 * a0
                    A01 = a0 * a1
                    # a10 = a01 can be omitted due to symmetry of matrix
                    A11 = a1 * a1
                    
                    for c in range(depth):
                        b[0,c] = a0*I_resized[i,j,c]
                        b[1,c] = a1*I_resized[i,j,c]
                    
                    for neighbour in range(4):
                        x = clip(i+h_neighbours[neighbour],0,h-1)
                        y = clip(j+w_neighbours[neighbour],0,w-1)
                        delta_alpha = epsilon + omega*fabsf(alpha_resized[i,j] - alpha_resized[x,y])
                        A00 += delta_alpha
                        A11 += delta_alpha
                        for c in range(depth):
                            b[0,c] += delta_alpha*F[x,y,c]
                            b[1,c] += delta_alpha*B[x,y,c]
                        
                    inv_det = <float>1.0 / (A00 * A11 - A01 * A01)

                    b00 = inv_det * A11
                    b01 = inv_det * -A01
                    b11 = inv_det * A00
                    
                    for c in range(depth):
                        F[i,j,c] = clip(b00 * b[0,c] + b01 * b[1,c], <float>0.0, <float>1.0)
                        B[i,j,c] = clip(b01 * b[0,c] + b11 * b[1,c], <float>0.0, <float>1.0)

    return np.asarray(F), np.asarray(B)