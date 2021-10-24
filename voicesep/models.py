import numpy as np
from nussl import separation, evaluation
from dataviz import visualize_and_embed
import pandas as pd

def _report_sdr(alg_name, scores):
    SDR = {}
    SIR = {}
    SAR = {}
    print(alg_name)
    print(''.join(['-' for i in range(len(alg_name))]))
    for key in scores:
        if key not in ['combination', 'permutation']:
            SDR[key] = np.mean(scores[key]['SI-SDR'])
            SIR[key] = np.mean(scores[key]['SI-SIR'])
            SAR[key] = np.mean(scores[key]['SI-SAR'])
            print(f'{key} SI-SDR: {SDR[key]:.2f} dB')
            print(f'{key} SI-SIR: {SIR[key]:.2f} dB')
            print(f'{key} SI-SAR: {SAR[key]:.2f} dB')
            print()
    print()

def evaluate_model(alg, truth):
    alg_estimates = alg()

    if isinstance(alg, separation.primitive.HPSS):
        alg_estimates = alg_estimates[::-1]

    visualize_and_embed(alg_estimates)
    print(truth)
    bss = evaluation.BSSEvalScale(
        truth, alg_estimates,
        source_labels=['accompaniment', 'voice'])
    scores = bss.evaluate()
    _report_sdr(str(alg).split(' on')[0], scores)
    return scores

def evaluate_dict_models(alg_dict, truth):
    scores = []
    for name, alg in alg_dict.items():
        score = evaluate_model(alg, truth)
        scores.append(score.update({"Alg":name}))
    return pd.DataFrame.from_records(scores).set_index("Alg")