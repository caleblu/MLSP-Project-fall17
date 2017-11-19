from dtw import dtw
from numpy.linalg import norm
import numpy as np

def dist_fn(x, y):
    return norm(x - y, ord = 1)

def align(src, tgt):
    jnt_vct = []
    for s,t in zip(src, tgt):
        jnt_vct.append(align_one(s, t))
    return np.r_[tuple(jnt_vct)]

def align_one(src, tgt):
    """
    :param src: spectral features of source
    :param tgt: spectral features of target
    :return: joint vector after align the source and target
    """
    _, _, _, twf = dtw(src, tgt, dist_fn)
    assert(len(twf[0]) == len(twf[1]))
    jnt_vector = np.c_[src[twf[0]], tgt[twf[1]]]
    return jnt_vector

if __name__=="__main__":
    from mfcc import audiofile_to_input_vector
    src = audiofile_to_input_vector("vcc2016_training/SF1/100001.wav", 12, 0)
    tgt = audiofile_to_input_vector("vcc2016_training/TM1/100001.wav", 12, 0)
    jnt_vector = align(src, tgt)
    print jnt_vector.shape
    print jnt_vector