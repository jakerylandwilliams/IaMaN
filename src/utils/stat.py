from collections import Counter
import numpy as np
import torch

# produces a vector sequence for viterbi decoding at the appropriate tag (sep-indicated) level
# for example: record the locations of whatevers within tokens and aggregate the token vectors
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

def blend_predictions(P1, P2, transfer_p, mean = 'arithmetic'):
    if mean == 'geometric':
        blended = {t: np.exp(np.log(P1[t])*(1 - transfer_p) + np.log(P2[t])*transfer_p) for t in P1}
    elif mean == 'harmonic':
        blended = {t: (P1[t]*P2[t])/((1 - transfer_p)*P2[t] + transfer_p*P1[t]) for t in P1}
    else:
        blended = {t: P1[t]*(1 - transfer_p) + P2[t]*transfer_p for t in P1}
    blended_sum = sum(blended.values())
    return Counter({t: blended[t]/blended_sum for t in blended})

