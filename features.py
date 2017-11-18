import pyworld as pw
import soundfile as sf
import os
import sys
import numpy as np
import h5py
import pysptk
from parameterizer import spc2npow


## F0
'''
x, fs = sf.read('sound.wav') # read audio file
_f0, t = pw.dio(x, fs)    # raw pitch extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
wav = pw.synthesize(f0, sp, ap, fs)


sf.write('analysis/source1_analysis.wav', wav, fs)


nonzero_indices = np.nonzero(f0)
f0s = np.log(f0[nonzero_indices])
f0stats = np.array([np.mean(f0s), np.std(f0s)])


with h5py.File('stats/f0stats', 'w') as hf:
    hf.create_dataset("stats/f0stats",  data = f0stats)

print("f0stats save into " + "stats/f0stats")
'''

## MCEP Source

x, fs = sf.read('source.wav') # read audio file
f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)

spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)

# spc, dim, alpha
org_mcep = pysptk.sp2mc(spc, 24, 0.42)
org_pow = spc2npow(spc)


## MCEP Target

x, fs = sf.read('target.wav') # read audio file
f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)

spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)

tar_mcep = pysptk.sp2mc(spc, 24, 0.42)
tar_npow = spc2npow(spc) # normalized power



#assert len(org_mcep) == len(tar_mcep)
#assert len(org_pow) == len(tar_npow)
    
