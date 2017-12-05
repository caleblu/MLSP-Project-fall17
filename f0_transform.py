import pyworld as pw
import numpy as np
import soundfile as sf
import pickle

def get_f0(path):
    x, fs = sf.read(path) # read audio file
    f0, time_axis = pw.harvest(x, fs, f0_floor=40.0, f0_ceil=500.0, frame_period=5.0)
    return f0

def load_f0(datapath, speaker_path, train_start, train_end):
    f0s = []
    for i in range(train_start, train_end+1):
        audio_path = datapath + speaker_path + str(i) + '.wav'
        f0 = get_f0(audio_path)
        f0s.append(f0[f0>0])
    return np.array(f0s)

def train_ratio(datapath, source, target, train_start, train_end):
    src_f0 = load_f0(datapath, source, train_start, train_end)
    tgt_f0 = load_f0(datapath, target, train_start, train_end)
    return np.mean(map(np.mean,tgt_f0)) / np.mean(map(np.mean,src_f0))

def main():
    datapath = "vcc2016_training/"
    source = "SF1/"
    target = "TM1/"
    train_start = 100001
    train_end = 100010
    ratio = train_ratio(datapath, source, target, train_start, train_end)
    with open("features/%s_%s_1.f0" % (source[:-1], target[:-1]), "wb") as f:
        pickle.dump(ratio, f, pickle.HIGHEST_PROTOCOL)

def load_ratio(path):
    with open(path, "rb") as f:
        ratio = pickle.load(f)
    return ratio
if __name__=="__main__":
    main()