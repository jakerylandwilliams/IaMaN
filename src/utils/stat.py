from collections import Counter
import numpy as np
import torch

# purpose: aggregate a sequence of vectors according to weights and track the mapping of lower-level components
# arguments:
# - vecs: list of 1-d arrays (vectors)
# - atns: list of floats (intensities) defining the aggregation of the vectors
# - seps: list of booleans (separator-statuses) for aggregation
# - nrms: list of floats (weights) derived from atns for the aggregation of the vectors
# prereqs: sequences of vectors and intensities, from which positive weights are measured (nrms)
# output: a no-longer vector sequence for decoding at the next tag level 
def agg_vecs(vecs, atns, seps, nrms = []):
    if not len(nrms): nrms = [1 for _ in atns]
    tvecs, tnrms, tatns = [], [], []
    aggvecs, aggnrms, aggatns = [], [], []
    for vec, nrm, atn, sep in zip(vecs, nrms, atns, seps):
        aggvecs.append(vec); aggnrms.append(nrm); aggatns.append(atn)
        if sep:
            aggnrm = sum(aggnrms)
            aggatn = sum(aggatns)
            if type(aggvecs[0]) == torch.Tensor:
                aggvec = (aggnrm/aggatn)*torch.stack(aggvecs).T.matmul(torch.tensor(aggatns, dtype = torch.double)) # , device = torch.cuda.current_device()
            else:
                aggvec = (aggnrm/aggatn)*np.array(aggvecs).T.dot(aggatns)
            tvecs.append(aggvec); tnrms.append(aggnrm); tatns.append(aggatn)
            aggvecs, aggnrms, aggatns = [], [], []
    if aggvecs:
        aggnrm = sum(aggnrms)
        aggatn = sum(aggatns)
        if type(aggvecs[0]) == torch.Tensor:
            aggvec = (aggnrm/aggatn)*torch.stack(aggvecs).T.matmul(torch.tensor(aggatns, dtype = torch.double)) # , device = torch.cuda.current_device()
        else:
            aggvec = (aggnrm/aggatn)*np.array(aggvecs).T.dot(aggatns)
        tvecs.append(aggvec); tnrms.append(aggnrm); tatns.append(aggatn)
    ixs = []; aixs = []
    for ix, sep in enumerate(seps):
        ixs.append(ix)
        if sep:
            aixs.append(ixs)
            ixs = []
    return tvecs, tnrms, tatns, aixs        

# purpose: blend two Counters of 1-summing probabilities by using a transfer probability and mean type
# arguments:
# P1: Counter of probabilities (summing to 1)
# P2: Counter of probabilities (summing to 1), assumed to have the same key set as P1
# transfer_p: float, randing over 0–1 weighting the end distribution's reliance on either of P1 (0) or P2 (1)
# mean: str, either: 'arithmetic' (default), 'geometric', or 'harmonic', indicating how the probabilities will be averaged
# prereqs: two Counters of probabilities, each with totality and ranging over the same key sets, as well as a transfer probability.
# output: Counter of blended probabilities (summing to 1)
def blend_predictions(P1, P2, transfer_p, mean = 'arithmetic'):
    if mean == 'geometric':
        blended = {t: np.exp(np.log(P1[t])*(1 - transfer_p) + np.log(P2[t])*transfer_p) for t in P1}
    elif mean == 'harmonic':
        blended = {t: (P1[t]*P2[t])/((1 - transfer_p)*P2[t] + transfer_p*P1[t]) for t in P1}
    else:
        blended = {t: P1[t]*(1 - transfer_p) + P2[t]*transfer_p for t in P1}
    blended_sum = sum(blended.values())
    return Counter({t: blended[t]/blended_sum for t in blended})

