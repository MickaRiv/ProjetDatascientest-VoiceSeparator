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

    i1, i2 = 0, 1
    if (isinstance(alg, separation.primitive.HPSS)):
        i1, i2 = i2, i1
    
    alg_estimates = {"Accompaniment":alg_estimates[i1],
                     "Voice":alg_estimates[i2]}
    bss = evaluation.BSSEvalScale(
        truth, [alg_estimates[key] for key in ["Accompaniment","Voice"]],
        source_labels=['accompaniment', 'voice'])
    scores = bss.evaluate()
    if isinstance(alg, separation.primitive.TimbreClustering):
        alt_bss = evaluation.BSSEvalScale(
                truth, [alg_estimates[key] for key in ["Voice","Accompaniment"]],
                source_labels=['accompaniment', 'voice'])
        alt_scores = alt_bss.evaluate()
        a = np.mean([np.mean(alt_scores[key]['SI-SDR']) for key in ['accompaniment', 'voice']])
        b = np.mean([np.mean(scores[key]['SI-SDR']) for key in ['accompaniment', 'voice']])
        print(a,b)
        if a > b:
            scores = alt_scores
    if visualize:
        visualize_and_embed(alg_estimates)
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
