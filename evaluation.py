from features import extract_features
from alignment import align_one
import numpy as np
import pandas as pd

def melCD(source_path, target_path, align=False):
    f0, src_mcep, ap, fs = extract_features(source_path)
    f0, tgt_mcep, ap, fs = extract_features(target_path)
    if align:
        jnt = align_one(src_mcep, tgt_mcep)
        mid = 25
        src_mcep = jnt[:,:mid]
        tgt_mcep = jnt[:,mid:]
    lenth = min(len(src_mcep), len(tgt_mcep))
    print lenth
    melcd = 10.0 / np.log(10) * np.sqrt(2 * np.linalg.norm(src_mcep[:lenth] - tgt_mcep[:lenth],'fro'))
    return melcd

def evaluate_test(datapath, speaker_path, jnt_path, test_start, test_end, method='FULL', f0_ratio=1):
    melcds = []
    for i in range(test_start, test_end+1):
        audio_path = datapath + speaker_path + str(i) + '.wav'
        output_path = 'result/' + jnt_path + str(i) + '_convert_%.2f_%s.wav' % (f0_ratio, method)
        melcds.append(melCD(audio_path, output_path, align=True))
    return melcds

def evaluate_origin(datapath, src_path, tgt_path, test_start, test_end):
    melcds = []
    for i in range(test_start, test_end+1):
        audio_path = datapath + src_path + str(i) + '.wav'
        output_path = datapath + tgt_path + str(i) + '.wav'
        melcds.append(melCD(audio_path, output_path, align=True))
    return melcds

def mcep_alignment(datapath, src_path, tgt_path, jnt_path, test_start,f0_ratio=1):
    for i in range(test_start, test_start+1):
        source_path = datapath + src_path + str(i) + '.wav'
        target_path = datapath + tgt_path + str(i) + '.wav'
        full_path = 'result/' + jnt_path + str(i) + '_convert_%.2f_%s.wav' % (f0_ratio, 'FULL')
        vq_path = 'result/' + jnt_path + str(i) + '_convert_%.2f_%s.wav' % (f0_ratio, 'VQ')
        f0, src_mcep, ap, fs = extract_features(source_path)
        f0, tgt_mcep, ap, fs = extract_features(target_path)
        f0, full_mcep, ap, fs = extract_features(full_path)
        f0, vq_mcep, ap, fs = extract_features(vq_path)
        jnt = align_one(src_mcep, tgt_mcep)
        mid = 25
        src = jnt[:,:mid]
        tgt = jnt[:,mid:]
        full = align_one(full_mcep, tgt_mcep)[:,:mid]
        vq = align_one(vq_mcep, tgt_mcep)[:,:mid]
        pd.DataFrame(src).to_csv("result/%d_srouce_mcep_align.csv" % test_start, header=None, index=None)
        pd.DataFrame(tgt).to_csv("result/%d_target_mcep_align.csv" % test_start, header=None, index=None)
        pd.DataFrame(full).to_csv("result/%d_full_mcep_align.csv" % test_start, header=None, index=None)
        pd.DataFrame(vq).to_csv("result/%d_vq_mcep_align.csv" % test_start, header=None, index=None)
        pd.DataFrame(np.mean(src,0)).to_csv("result/%d_srouce_mcep_align_mean.csv" % test_start, header=None, index=None)
        pd.DataFrame(np.mean(tgt,0)).to_csv("result/%d_target_mcep_align_mean.csv" % test_start, header=None, index=None)
        pd.DataFrame(np.mean(full,0)).to_csv("result/%d_full_mcep_align_mean.csv" % test_start, header=None, index=None)
        pd.DataFrame(np.mean(vq,0)).to_csv("result/%d_vq_mcep_align_mean.csv" % test_start, header=None, index=None)
    return src_mcep, tgt_mcep, full_mcep, vq_mcep

if __name__=="__main__":
    mcep_alignment("vcc2016_training/", "SF1/","TM1/", "SF1_TM1_150/", 100151, 0.56)
    # full = evaluate_test("vcc2016_training/", "TM1/", "SF1_TM1_150/", 100151, 100161, 'FULL', 0.56)
    # vq = evaluate_test("vcc2016_training/", "TM1/", "SF1_TM1_150/", 100151, 100161, 'VQ', 0.56)
    # origin = evaluate_origin("vcc2016_training/", "SF1/","TM1/", 100151, 100161)
    #
    # for v in zip(origin, full, vq):
    #     print v
