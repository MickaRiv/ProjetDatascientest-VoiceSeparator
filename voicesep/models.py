import numpy as np
from nussl import separation, evaluation
from .dataviz import visualize_and_embed
import pandas as pd

def report_sdr(alg_name, scores):
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

def evaluate_model(alg, truth, visualize=True, report=True):
    alg_estimates = alg()

    if (isinstance(alg, separation.primitive.HPSS) or
        isinstance(alg, separation.primitive.TimbreClustering)):
        alg_estimates = alg_estimates[::-1]
    
    alg_estimates = {"Accompaniment":alg_estimates[0],
                     "Voice":alg_estimates[1]}
    if visualize:
        visualize_and_embed(alg_estimates)
    bss = evaluation.BSSEvalScale(
        truth, alg_estimates,
        source_labels=['accompaniment', 'voice'])
    scores = bss.evaluate()
    if report:
        report_sdr(str(alg).split(' on')[0], scores)
    return scores

def evaluate_dict_models(alg_dict, truth, **kwargs):
    scores = []
    for name, alg in alg_dict.items():
        score = evaluate_model(alg, truth, **kwargs)
        score.pop("combination")
        score.pop("permutation")
        score = {(subject,metric):np.mean(val)
                 for subject,metrics in score.items()
                 for metric,val in metrics.items()}
        score.update({"Alg":name})
        scores.append(score)
    scores_df = pd.DataFrame.from_records(scores).set_index("Alg")
    scores_df.columns = pd.MultiIndex.from_tuples(scores_df.columns)
    return scores_df