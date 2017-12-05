__author__ = 'gardenia'
from features import extract_features
from alignment import align
from gmm import train_GMM, predict_GMM_VQ, predict_GMM_FULL
from f0_transform import train_ratio
from synthesizer import synthesize
import numpy as np
import soundfile as sf
import os
from sklearn.externals import joblib
import time
import pickle


def load_train(datapath, speaker_path, train_start, train_end):
    mceps = []
    for i in range(train_start, train_end+1):
        audio_path = datapath + speaker_path + str(i) + '.wav'
        f0, mcep, ap, fs = extract_features(audio_path)
        mceps.append(mcep)
    return np.array(mceps)

def convert(audio_path, output_path, GMM, method='FULL', f0_ratio=1):
    f0, mcep, ap, fs = extract_features(audio_path)
    if method=='FULL':
        mcep_transform =  predict_GMM_FULL(mcep, GMM)
    elif method=='VQ':
        mcep_transform =  predict_GMM_VQ(mcep, GMM)
    else:
        # use full conversion as default
        mcep_transform =  predict_GMM_FULL(mcep, GMM)
    wav = synthesize(f0 * f0_ratio, ap, fs, mcep_transform)
    sf.write(output_path, wav, fs)

def convert_test(datapath, speaker_path, jnt_path, test_start, test_end, GMM_model, method='FULL', f0_ratio=1):
    for i in range(test_start, test_end+1):
        audio_path = datapath + speaker_path + str(i) + '.wav'
        if not os.path.exists('result/' + jnt_path):
            os.makedirs('result/' + jnt_path)
        output_path = 'result/' + jnt_path + str(i) + '_convert_%.2f.wav' % f0_ratio
        convert(audio_path, output_path, GMM_model, method, f0_ratio)

def dump(d, path):
    with open(path, "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)



def main():
    datapath = "vcc2016_training/"
    # training examples
    train_start = 100001
    train_end = 100150

    # test examples
    test_start = 100151
    test_end = 100161
    source_path = "SF1/"
    target_path = "TM1/"
    jnt_name = source_path[:-1] + "_" + target_path[:-1] + '_' + str(train_end - train_start + 1)
    jnt_path = 'features/' + jnt_name + ".jnt"
    gmm_path = 'model/gmm_' + jnt_name + ".pkl"
    f0_path = 'features/' + jnt_name + ".f0"

    # get GMM_model from saved files, otherwise train the model and save it.
    if os.path.isfile(gmm_path):
        print "loading gmm from" + gmm_path
        GMM_model = joblib.load(gmm_path)
    else:
        if os.path.isfile(jnt_path):
            print "loading joint vector from" + jnt_path
            jnt_vct = load(jnt_path)
        else:
            print "loading mcep features"
            src = load_train(datapath, "SF1/", train_start, train_end)
            tgt = load_train(datapath, "TM1/", train_start, train_end)
            print "calculating dtw"
            jnt_vct = align(src, tgt)
            print "saving joint vector to " + jnt_path
            dump(jnt_vct, jnt_path)
        print "gmm training"
        GMM_model = train_GMM(jnt_vct)
        print "save gmm model"
        joblib.dump(GMM_model, gmm_path)

    if os.path.isfile(f0_path):
        print "loading f0 ratio from" + f0_path
        f0_ratio = load(f0_path)
    else:
        print "calculating f0 ratio"
        f0_ratio = train_ratio(datapath, source_path, target_path, train_start, train_end)
        print "saving f0 ratio"
        dump(f0_ratio, f0_path)

    print "converting test audios"
    convert_test(datapath, source_path, jnt_name + '/',test_start, test_end, GMM_model, 'FULL' ,f0_ratio)

if __name__=="__main__":
    main()