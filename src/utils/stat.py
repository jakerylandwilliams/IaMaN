from collections import Counter, defaultdict
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

def noise(p, beta, f = []):
    if len(f) != len(p):
        f = np.ones(len(p))
    return p*beta + (lambda x: (1 - x)/(1-x).sum())(f/f.sum())*(1 - beta)

def evaluate_segmentation(pred_segs, true_segs):
    pred_spans = list(np.cumsum([len(t) for t in pred_segs]))
    pred_spans = set([(sh-len(gt), sh) for sh, gt in zip(pred_spans, pred_segs)])
    true_spans = list(np.cumsum([len(t) for t in true_segs]))
    true_spans_nsp = {(sh-len(gt), sh): 1 for sh, gt in zip(true_spans, true_segs)}
    true_spans = {(sh-len(gt), sh): 1 for sh, gt in zip(true_spans, true_segs)}
    confusion = {'sp': {"TP": 0, "FP": 0, "FN": 0, "P": 0, "R": 0, "F": 0},
                 'ns': {"TP": 0, "FP": 0, "FN": 0, "P": 0, "R": 0, "F": 0}}
    for pred_span, pred_seg in zip(pred_spans, pred_segs):
        if pred_span not in true_spans:
            confusion['sp']['FP'] += 1
            if pred_seg != ' ':
                confusion['ns']['FP'] += 1
        else:
            confusion['sp']['TP'] += 1
            if pred_seg != ' ':
                confusion['ns']['TP'] += 1
    for true_span, true_seg in zip(true_spans, true_segs):
        if true_span not in pred_spans:
            confusion['sp']['FN'] += 1
            if true_seg != ' ':
                confusion['ns']['FN'] += 1
    for ky in confusion:
        confusion[ky]['P'] = confusion[ky]['TP']/(confusion[ky]['TP'] + confusion[ky]['FP']) if (confusion[ky]['TP'] + confusion[ky]['FP']) else 0
        confusion[ky]['R'] = confusion[ky]['TP']/(confusion[ky]['TP'] + confusion[ky]['FN']) if (confusion[ky]['TP'] + confusion[ky]['FN']) else 0
        confusion[ky]['F'] = 2*confusion[ky]['P']*confusion[ky]['R']/(confusion[ky]['P']+confusion[ky]['R']) if (confusion[ky]['P']+confusion[ky]['R']) else 0
    return confusion

def merge_confusion(x, y):
    confusion = {sp_ky: {ky: x[sp_ky][ky] + y[sp_ky][ky] for ky in ["TP", "FP", "FN"]}
                 for sp_ky in ['sp', 'ns']}    
    for ky in confusion:
        confusion[ky]['P'] = confusion[ky]['TP']/(confusion[ky]['TP'] + confusion[ky]['FP']) if (confusion[ky]['TP'] + confusion[ky]['FP']) else 0
        confusion[ky]['R'] = confusion[ky]['TP']/(confusion[ky]['TP'] + confusion[ky]['FN']) if (confusion[ky]['TP'] + confusion[ky]['FN']) else 0
        confusion[ky]['F'] = 2*confusion[ky]['P']*confusion[ky]['R']/(confusion[ky]['P']+confusion[ky]['R']) if (confusion[ky]['P']+confusion[ky]['R']) else 0
    return confusion

def evaluate_tagging(pred_segs, true_segs, pred_tags, true_tags):
    pred_spans = list(np.cumsum([len(t) for t in pred_segs]))
    pred_spans = {(sh-len(gt), sh): (gl, gt)
                  for sh, gt, gl in zip(pred_spans, pred_segs, pred_tags)}
    true_spans = list(np.cumsum([len(t) for t in true_segs]))
    true_spans = {(sh-len(gt), sh): (gl, gt)
                  for sh, gt, gl in zip(true_spans, true_segs, true_tags)}
    accuracy = {'sp': defaultdict(list), 'ns': defaultdict(list), 'tsp': [], 'tns': []}
    for true_span in true_spans:
        if true_span in pred_spans:
            result = true_spans[true_span] == pred_spans[true_span]
        else:
            result = False
        accuracy['sp'][true_spans[true_span][0]].append(result)
        accuracy['tsp'].append(result)
        if true_spans[true_span][1] != ' ':
            accuracy['ns'][true_spans[true_span][0]].append(result)
            accuracy['tns'].append(result)
    return accuracy

def merge_accuracy(x,y):
    return {'tsp': x['tsp'] + y['tsp'], 'tns': x['tns'] + y['tns'],
            'sp': defaultdict(list, {ky: x['sp'][ky] + y['sp'][ky]
                                     for ky in set(list(x['sp'].keys()) + list(y['sp'].keys()))}),
            'ns': defaultdict(list, {ky: x['ns'][ky] + y['ns'][ky]
                                     for ky in set(list(x['ns'].keys()) + list(y['ns'].keys()))})}

def evaluate_document(document, layers, covering = []):
    false_cover = False
    if not covering: 
        covering = list([t._form for s in document._sentences for t in s._tokens])
        false_cover = True

    eot_confusion = None
    # build token sequences
    pred_toks = [t._form for s in document._sentences for t in s._tokens]
    gold_toks = [ct for cs in covering for ct in cs]
    if not false_cover:
        # evaluate token segmentation performance
        eot_confusion = evaluate_segmentation(pred_toks, gold_toks)
    
    pos_accuracy = None
    # build pos tag sequences
    if 'pos' in layers:
        pred_stream = [t._pos for s in document._sentences for t in s._tokens]
        gold_stream = [POS for ls in layers['pos'] for POS in ls]
        # evaluate pos tagging performance
        pos_accuracy = evaluate_tagging(pred_toks, gold_toks, pred_stream, gold_stream)
    
    eos_confusion = None
    # build sentence sequences
    pred_sents = [''.join([t._form for t in s._tokens]) for s in document._sentences]
    gold_sents = [''.join([ct for ct in cs]) for cs in covering]
    if not false_cover:
        # evaluate sentence segmentation performance
        eos_confusion = evaluate_segmentation(pred_sents, gold_sents)
    
    sty_accuracy = None
    # build sentence-tag sequences
    if 'sty' in layers:
        pred_stys = [s._sty for s in document._sentences if s._sty is not None]
        gold_stys = [ls[0] for ls in layers['sty']]
        # evaluate sentence-tag performance
        sty_accuracy = evaluate_tagging(pred_sents, gold_sents, pred_stys, gold_stys)
    
    arc_accuracy = None
    # build tagged-arc sequences
    if ('sup' in layers) and ('dep' in layers):
        pred_arcs = [(str(t._sup), t._dep) for s_i, s in enumerate(document._sentences) for ix, t in enumerate(s._tokens)]
        gold_arcs = [SUPDEP for s_i, ls in enumerate(layers['sup']) for SUPDEP in zip(ls, layers['dep'][s_i])]
        # evaluate arcs tagging performance
        arc_accuracy = evaluate_tagging(pred_toks, gold_toks, pred_arcs, gold_arcs)
    
    return eot_confusion, pos_accuracy, eos_confusion, sty_accuracy, arc_accuracy