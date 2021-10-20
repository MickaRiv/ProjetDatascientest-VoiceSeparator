# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:27:51 2021

@author: magla
"""

#from same website
import numpy as np
def _report_sdr(approach, scores):
    SDR = {}
    SIR = {}
    SAR = {}
    print(approach)
    print(''.join(['-' for i in range(len(approach))]))
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
    
# from same website
from nussl import separation,evaluation
from dataviz import visualize_and_embed

def run_viz_and_evaluate(alg,sources_list):
    alg_estimates = alg()

    if isinstance(alg, separation.primitive.HPSS):
        alg_estimates = alg_estimates[::-1]

    visualize_and_embed(alg_estimates)
    print(sources_list)
    bss = evaluation.BSSEvalScale(
        sources_list, alg_estimates,
        source_labels=['acc', 'vox'])
    scores = bss.evaluate()
    _report_sdr(str(alg).split(' on')[0], scores)