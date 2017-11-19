import pyworld as pw
import soundfile as sf
import pysptk
from parameterizer import spc2npow
import alignment
import synthesizer

## MCEP Source

x, fs = sf.read('source.wav') # read audio file

f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)

spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)

aperiodicity = pw.d4c(x, f0, time_axis, fs)     # extract aperiodicity

# spc, dim, alpha
org_mcep = pysptk.sp2mc(spc, 24, 0.42)

synthesized_source_wav = synthesizer.synthesize(f0, spc, aperiodicity, fs)


sf.write('synth/synthesized_source.wav', synthesized_source_wav, fs)

## MCEP Target

x, fs = sf.read('target.wav') # read audio file

f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)

spc = pw.cheaptrick(x, f0, time_axis, fs, fft_size=1024)

aperiodicity = pw.d4c(x, f0, time_axis, fs)     # extract aperiodicity

tar_mcep = pysptk.sp2mc(spc, 24, 0.42)

synthesized_target_wav = synthesizer.synthesize(f0, spc, aperiodicity, fs)

 
sf.write('synth/synthesized_target.wav', synthesized_target_wav, fs)