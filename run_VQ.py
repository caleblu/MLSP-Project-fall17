__author__ = 'gardenia'
from features import extract_features
from alignment import align
from gmm import train_GMM, predict_GMM
from synthesizer import synthesize
import numpy as np
import soundfile as sf
import os

path = "vcc2016_training/"
# training examples
train_start = 100001
train_end = 100030

# test examples
test_start = 100151
test_end = 100151


def load_train(speaker_path):
    mceps = []
    for i in range(train_start, train_end+1):
        audio_path = path + speaker_path + str(i) + '.wav'
        f0, mcep, ap, fs = extract_features(audio_path)
        mceps.append(mcep)
    return np.array(mceps)

def convert(audio_path, output_path, GMM):
    f0, mcep, ap, fs = extract_features(audio_path)
    mcep_transfrom =  predict_GMM(mcep, GMM)
    wav = synthesize(f0, ap, fs, mcep_transfrom)
    sf.write(output_path, wav, fs)

def convert_test(speaker_path, GMM_model):
    for i in range(test_start, test_end+1):
        audio_path = path + speaker_path + str(i) + '.wav'
        if not os.path.exists('result/' + speaker_path):
            os.makedirs('result/' + speaker_path)
        output_path = 'result/' + speaker_path + str(i) + '_convert.wav'
        convert(audio_path, output_path, GMM_model)

def main():
    print "loading mcep features"
    src = load_train("SF1/")
    tgt = load_train("TM1/")
    print "calculating dtw"
    jnt_vct = align(src, tgt)
    print "gmm training"
    GMM_model = train_GMM(jnt_vct)
    print "converting"
    convert_test("SF1/", GMM_model)

if __name__=="__main__":
    main()