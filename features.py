import pyworld as pw
import soundfile as sf
import pysptk
from parameterizer import spc2npow
import alignment


## MCEP Source

x, fs = sf.read('source.wav') # read audio file
f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)

spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)

# spc, dim, alpha
org_mcep = pysptk.sp2mc(spc, 24, 0.42)


## MCEP Target

x, fs = sf.read('target.wav') # read audio file
f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)

spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)

tar_mcep = pysptk.sp2mc(spc, 24, 0.42)

