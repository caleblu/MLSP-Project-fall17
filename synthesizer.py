import pyworld as pw

def synthesize(f0, spc, aperiodicity, fs):
	return pw.synthesize(f0, spc, aperiodicity, fs, frame_period= 5.0)