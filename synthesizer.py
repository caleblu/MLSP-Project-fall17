__author__ = 'gardenia'

import pyworld as pw
import pysptk

def synthesize(f0, ap, fs, mcep):
    spc = pysptk.mc2sp(mcep, 0.42, 1024)
    return pw.synthesize(f0, spc, ap, fs, frame_period= 5.0)