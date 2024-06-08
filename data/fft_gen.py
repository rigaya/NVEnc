import math
import cmath

def dft(f_):
    """
    愚直に和を取る
    """
    N_ = len(f_)
    F_ = [0j] * N_
    x = -1j*2*math.pi/N_
    for k in range(N_):
        for n in range(N_):
            F_[k] += f_[n] * cmath.exp(x*k*n)
    return F_

def fw(k, N):
    return cmath.exp(-1j*2*math.pi*k/N)

def ifw(k, N):
    return cmath.exp(+1j*2*math.pi*k/N)

def fftpermute(f):
    N = len(f)
    if N == 2:
        return [f[0], f[1]]
    elif N == 4:
        return [f[0], f[2], f[1], f[3]]
    elif N == 8:
        return [f[0], f[4], f[2], f[6], f[1], f[5], f[3], f[7]]
    elif N == 16:
        return [f[0], f[8], f[4], f[12], f[2], f[10], f[6], f[14], f[1], f[9], f[5], f[13], f[3], f[11], f[7], f[15]]
    elif N == 32:
        return [f[0], f[16], f[8], f[24], f[4], f[20], f[12], f[28], f[2], f[18], f[10], f[26], f[6], f[22], f[14], f[30], f[1], f[17], f[9], f[25], f[5], f[21], f[13], f[29], f[3], f[19], f[11], f[27], f[7], f[23], f[15], f[31]]
    else:
        raise ValueError("N must be 2, 4, 8, 16 or 32")
    
def ifftpermute(f):
    N = len(f)
    ifarray = fftpermute(f)
    return [ x / N for x in ifarray ]

def fft2(f):
    w0 = f[0] + fw(0, 2) * f[1]
    w1 = f[0] - fw(0, 2) * f[1]
    #print(str(f[0]) + ", " + str(f[1]) + " -> " + str(w0) + ", " + str(w1))
    return [w0, w1]

def fft4(f):
    w  = fft2([f[0], f[1]])
    w += fft2([f[2], f[3]])
    #print(w)

    for i in range(2):
        f[i]   = w[i] + fw(i, 4) * w[i+2]
        f[i+2] = w[i] - fw(i, 4) * w[i+2]
    return f

def fft8(f):
    w  = fft4([f[0], f[1], f[2], f[3]])
    w += fft4([f[4], f[5], f[6], f[7]])

    for i in range(4):
        f[i]   = w[i] + fw(i, 8) * w[i+4]
        f[i+4] = w[i] - fw(i, 8) * w[i+4]
    return f

def fft16(f):
    w = fft8([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]])
    w += fft8(f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15])
              
    for i in range(8):
        f[i]   = w[i] + fw(i, 16) * w[i+8]
        f[i+8] = w[i] - fw(i, 16) * w[i+8]
    return f

def fft32(f):
    w = fft16([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]])
    w += fft16([f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28], f[29], f[30], f[31]])
              
    for i in range(16):
        f[i]    = w[i] + fw(i, 32) * w[i+16]
        f[i+16] = w[i] - fw(i, 32) * w[i+16]
    return f

#def ifft2(f):
#    w0 = f[0] + f[1]
#    w1 = (f[0] - f[1]) * fw(0, 2)
#    return [w0, w1]
#
#def ifft4(f):
#    w = [0j] * 4
#    for i in range(2):
#        w[i]   = f[i] + f[i+2]
#        w[i+2] = (f[i] - f[i+2]) * fw(i, 4)
#
#    f  = ifft2([w[0], w[1]])
#    f += ifft2([w[2], w[3]])
#    return f
#
#def ifft8(f):
#    w = [0j] * 8
#    for i in range(4):
#        w[i]   = f[i] + f[i+4]
#        w[i+4] = (f[i] - f[i+4]) * fw(i, 8)
#
#    f  = ifft4([w[0], w[1], w[2], w[3]])
#    f += ifft4([w[4], w[5], w[6], w[7]])
#    return f


def ifft2(f):
    w0 = f[0] + ifw(0, 2) * f[1]
    w1 = f[0] - ifw(0, 2) * f[1]
    #print(str(f[0]) + ", " + str(f[1]) + " -> " + str(w0) + ", " + str(w1))
    return [w0, w1]

def ifft4(f):
    w  = ifft2([f[0], f[1]])
    w += ifft2([f[2], f[3]])
    #print(w)

    for i in range(2):
        f[i]   = w[i] + ifw(i, 4) * w[i+2]
        f[i+2] = w[i] - ifw(i, 4) * w[i+2]
    return f

def ifft8(f):
    w  = ifft4([f[0], f[1], f[2], f[3]])
    w += ifft4([f[4], f[5], f[6], f[7]])

    for i in range(4):
        f[i]   = w[i] + ifw(i, 8) * w[i+4]
        f[i+4] = w[i] - ifw(i, 8) * w[i+4]
    return f

T = 5
N = 8
freq1 = 1.0
freq2 = 2.5

f = [0j] * N

for i in range(N):
    t = i*T/N
    f[i] = math.sin(2*math.pi*freq1*t) + 0.2 * math.sin(2*math.pi*freq2*t)

print(f)

# 種々の実装でフーリエ変換を実行
#F_dft = dft(f)
#F_fft2 = fft2(fftpermute(f[0:2]))
#F_fft4 = fft4(fftpermute(f[0:4]))
#F_fft = fft8(fftpermute(f))

#F_ifft = [ x/N for x in ifft8(fftpermute(F_fft)) ]

#print(F_fft2)
#print(F_fft4)
#print(F_dft)
#print(F_fft)
#print(F_ifft)