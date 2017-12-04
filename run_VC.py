__author__ = 'gardenia'
from features import extract_features
from alignment import align
from gmm import train_GMM, predict_GMM_VQ, predict_GMM_FULL
from synthesizer import synthesize
import numpy as np
import soundfile as sf
import os
from sklearn.externals import joblib
import time

path = "vcc2016_training/"
# training examples
train_start = 100001
train_end = 100150

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

def convert(audio_path, output_path, GMM, method='FULL'):
    f0, mcep, ap, fs = extract_features(audio_path)
    if method=='FULL':
        mcep_transform =  predict_GMM_FULL(mcep, GMM)
    elif method=='VQ':
        mcep_transform =  predict_GMM_VQ(mcep, GMM)
    else:
        # use full conversion as default
        mcep_transform =  predict_GMM_FULL(mcep, GMM)
    wav = synthesize(f0, ap, fs, mcep_transform)
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
    print "save gmm model"
    joblib.dump(GMM_model, 'model/gmm%s.pkl' % int(time.time()))
    # GMM_model = joblib.load(path_to_pkl)
    print "converting"
    convert_test("SF1/", GMM_model)

if __name__=="__main__":
    main()