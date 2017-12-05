__author__ = 'gardenia'
from features import extract_features
from alignment import align_one
import numpy as np

def melCD(source_path, target_path, align=False):
    f0, src_mcep, ap, fs = extract_features(source_path)
    f0, tgt_mcep, ap, fs = extract_features(target_path)
    if align:
        jnt = align_one(src_mcep, tgt_mcep)
        mid = len(jnt) / 2
        src_mcep = jnt[:mid]
        tgt_mcep = jnt[mid:]
    lenth = min(len(src_mcep), len(tgt_mcep))
    print lenth
    melcd = 10.0 / np.log(10) * np.sqrt(2 * np.linalg.norm(src_mcep[:lenth] - tgt_mcep[:lenth],'fro'))
    return melcd

def evaluate_test(datapath, speaker_path, jnt_path, test_start, test_end, method='FULL', f0_ratio=1):
    melcds = []
    for i in range(test_start, test_end+1):
        audio_path = datapath + speaker_path + str(i) + '.wav'
        output_path = 'result/' + jnt_path + str(i) + '_convert_%.2f_%s.wav' % (f0_ratio, method)
        melcds.append(melCD(audio_path, output_path))
    return melcds

def evaluate_origin(datapath, src_path, tgt_path, test_start, test_end):
    melcds = []
    for i in range(test_start, test_end+1):
        audio_path = datapath + src_path + str(i) + '.wav'
        output_path = datapath + tgt_path + str(i) + '.wav'
        melcds.append(melCD(audio_path, output_path, align=True))
    return melcds

full = evaluate_test("vcc2016_training/", "SF1/", "SF1_TM1_150/", 100151, 100161, 'FULL', 0.56)
vq = evaluate_test("vcc2016_training/", "SF1/", "SF1_TM1_150/", 100151, 100161, 'VQ', 0.56)
origin = evaluate_origin("vcc2016_training/", "SF1/","TM1/", 100151, 100161)

for v in zip(origin, full, vq):
    print v
