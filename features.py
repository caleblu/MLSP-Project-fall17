import pyworld as pw
import soundfile as sf
import pysptk

def extract_features(path):
    x, fs = sf.read(path) # read audio file
    f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)
    spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)
    # spc, dim, alpha
    mcep = pysptk.sp2mc(spc, 24, 0.42)
    ap = pw.d4c(x, f0, time_axis, fs)
    return f0, mcep, ap, fs


