from collections import Counter, defaultdict
import numpy as np
from functools import reduce
import multiprocessing
from .fnlp import get_context
import json, torch

def build_layers(d, covering, ltypes, layers):
    temp_ltypes = []
    temp_layers = []
    if (covering and ltypes) and len(ltypes) == len(layers):
        for layer, ltype in zip(layers, ltypes):
            if ltype in ['lem', 'pos', 'ent', 'dep', 'sup', 'sty']:
                if all([len(s_c) == len(s_l) for s_c, s_l in zip(covering, layer)]):
                    temp_ltypes.append(ltype); temp_layers.append([])
                    for s_i, layer_sent in enumerate(layer):
                        layer_sent = np.array(layer_sent)
                        layer_norms = np.cumsum([len(t) for t in covering[s_i]])
                        whatever_norms = np.cumsum([len(t) for t in d[s_i]])
                        temp_layers[-1].append([layer_sent[whatever_norm <= layer_norms][0] for
                                                whatever_norm in whatever_norms])
    ltypes = list(temp_ltypes); del(temp_ltypes)
    layers = list(temp_layers); del(temp_layers)
    return ltypes, layers

def build_document(d, ltypes, layers):
    meta = {'Tds': [Counter()], 'total_sentences': len(d), 'M': 0,
            'alphas': defaultdict(list), 'Tls': [defaultdict(lambda : defaultdict(Counter))],
            'L': defaultdict(lambda : defaultdict(set)), 'f0': Counter()}
    delta = 0; s_j = 0
    nvs, ess, eds = [], [], []
    for s_j, s in enumerate(d):
        nvs.append([]); ess.append([]); eds.append([])
        for i, t in enumerate(list(s)):
            meta['f0'][t] += 1
            eds[-1].append(False)
            if i == len(s) - 1:
                ess[-1].append(True)
            else:
                ess[-1].append(False)
            delta += 1; meta['M'] += 1
            if t not in meta['Tds'][0]:
                # record the novelty and vocabulary expansion events
                nvs[-1].append(True)
                # record the word introduction rate
                alpha = 1/delta
                for Mt in range(meta['M']-delta,meta['M']+1):
                    meta['alphas'][Mt].append(alpha)
                delta = 0
            else:
                nvs[-1].append(False)
            meta['Tds'][0][t] += 1
            for li, ltype in enumerate(ltypes):
                meta['Tls'][0][ltype][layers[li][s_j][i]][t] += 1 
                meta['L'][ltype][t].add(layers[li][s_j][i])
    if eds:
        eds[s_j][-1] = True
    meta['Tls'] = [dict(meta['Tls'][0])]
    meta['L'] = dict(meta['L'])
    return nvs, ess, eds, meta

def build_eots(d, covering):
    eots = []
    for c_s, s in zip(covering, d):
        locs = np.cumsum(list(map(len, s)))
        c_locs = np.cumsum(list(map(len, c_s)))
        eots.append([loc in c_locs for loc in locs])
    return eots

def build_bots(d, covering):
    bots = []
    for c_s, s in zip(covering, d):
        locs = np.cumsum(list(map(len, s))) - len(s[0])
        c_locs = np.cumsum(list(map(len, c_s))) - len(c_s[0])
        bots.append([loc in c_locs for loc in locs])
    return bots

def build_iats(d, covering):
    iats = []
    for c_s, s in zip(covering, d):
        locs = np.array([0] + list(np.cumsum(list(map(len, s)))))
        spns = set(list(zip(locs[:-1], locs[1:])))
        c_locs = np.array([0] + list(np.cumsum(list(map(len, c_s)))))
        c_spns = set(list(zip(c_locs[:-1], c_locs[1:])))
        iats.append([spn in c_spns for spn in sorted(spns)])
    return iats

def process_document(x): 
    doc, covering, ltypes, layers = x
    # tokenize the document (note: currently the tokenizer is non-serializable, and should be a broadcast variable, too!)
#     d = [self.tokenizer.tokenize(s) for s in doc]
#     d = [bc['tokenizer'].tokenize(s) for s in doc]
    d = json.loads(doc)
    ## build the higher-level training layers (gold linguistic tags)
    ## assures layers conform to sent-tok covering; tags projects to the whatever level
    ltypes, layers = build_layers(d, covering, ltypes, layers)
    nvs, ess, eds, meta = build_document(d, ltypes, layers)
    # if there was a covering, fit the model to its segmentation (end of token prediction)
    # as well as any other cover layers (POS, etc.) onto the end of token-positioned whatevers
    if covering:
        its = build_iats(d, covering); bts = build_bots(d, covering); ets = build_eots(d, covering)
        layers = layers + [nvs, its, bts, ets, ess, eds]
        ltypes = ltypes + ['nov', 'iat', 'bot', 'eot', 'eos', 'eod']
        ## replacing # meta['max_sent'] = max([max([sum(s) for s in ets]), self._max_sent])
        meta['max_sent'] = max([sum(s) for s in ets]) 
        ##
    else:
        layers = layers + [nvs, ess, eds]
        ltypes = ltypes + ['nov', 'eos', 'eod']
        meta['max_sent'] = 0
    meta['lorder'] = list(ltypes)
    
    return d, layers, ltypes, meta

def count(x, bc = {}):
    if not type(bc) == dict: # this allows the function to access broadcast variables through spark
        bc = dict(bc.value)
    doc, covering, ltypes, layers = x
    # tokenize the document (note: currently the tokenizer is non-serializable, and should be a broadcast variable, too!)
#     d = [self.tokenizer.tokenize(s) for s in doc]
#     d = [bc['tokenizer'].tokenize(s) for s in doc]
    d = json.loads(doc)
    ##
    ltypes, layers = build_layers(d, covering, ltypes, layers)
    nvs, ess, eds, meta = build_document(d, ltypes, layers)
    if covering:
        its = build_iats(d, covering); bts = build_bots(d, covering); ets = build_eots(d, covering)
        layers = layers + [nvs, its, bts, ets, ess, eds]
        ltypes = ltypes + ['nov', 'iat', 'bot', 'eot', 'eos', 'eod']
        meta['max_sent'] = max([sum(s) for s in ets])
    else:
        layers = layers + [nvs, ess, eds]
        ltypes = ltypes + ['nov', 'eos', 'eod']
        meta['max_sent'] = 0
    meta['lorder'] = list(ltypes)
    ##
    ## replacing: doc_Fs = [self.count(d, old_ife = old_ife)]
    yield from Counter([((t,'form'), c) 
                        for s in [[t_s for ds in d for t_s in ds]] for i, t in enumerate(s)
                        for c in get_context(i,  list(s), m = bc['m']) 
                        if ((not bc['old_ife']) or (bc['old_ife'] and (t in bc['old_ife'] and c[0] in bc['old_ife'])))]).items()
    yield from Counter([((t,'form'), tuple([str(c[1]), c[1], 'attn'])) # c
                        for s in [[t_s for ds in d for t_s in ds]] for i, t in enumerate(s)
                        for c in get_context(i,  list(s), m = bc['m']) 
                        if ((not bc['old_ife']) or (bc['old_ife'] and (t in bc['old_ife'] and c[0] in bc['old_ife'])))]).items()
    ##
    ## replacing: for ltype, layer_Fs in self.count_layers(d, layers = layers, ltypes = ltypes, old_ife = old_ife): doc_Fs.append(layer_Fs)
    for layer, ltype in zip(layers, ltypes):
        if ltype == 'form': continue
        yield from Counter([((l,ltype), c) 
                            for s, s_l in [list(zip(*[(t_s, t_l) for ds, ds_l in zip(d, layer) for t_s, t_l in zip(ds, ds_l)]))]
                            for i, (t, l) in enumerate(list(zip(s,s_l)))
                            for c in get_context(i,  list(s), m = bc['m']) 
                            if ((not bc['old_ife']) or (bc['old_ife'] and (t in bc['old_ife'] and c[0] in bc['old_ife'])))]).items()
        ## accrue data for positional distributions
        yield from Counter([((l,ltype), tuple([str(c[1]), c[1], 'attn'])) # c
                            for s, s_l in [list(zip(*[(t_s, t_l) for ds, ds_l in zip(d, layer) for t_s, t_l in zip(ds, ds_l)]))]
                            for i, (t, l) in enumerate(list(zip(s,s_l)))
                            for c in get_context(i,  list(s_l), m = bc['m']) # note s_l, not l!
                            if ((not bc['old_ife']) or (bc['old_ife'] and (t in bc['old_ife'] and c[0] in bc['old_ife'])))]).items()
        ##
    ##

def aggregate_processed(a, b): # a/b = (Fs, meta)
    alphas = a['alphas']
    for Mt in b['alphas']:
        for alpha in b['alphas'][Mt]:
            alphas[Mt].append(alpha)
    lorder = a['lorder']
    for ltype in b['lorder']:
        if ltype not in lorder: 
            lorder.append(ltype)
    L = a['L']
    for ltype in b['L']:
        for t in b['L'][ltype]:
            L[ltype][t].union(b['L'][ltype][t])
    return({'M': a['M'] + b['M'],
            'total_sentences': a['total_sentences'] + b['total_sentences'],
            'alphas': alphas,
            'Tds': a['Tds'] + b['Tds'],
            'Tls': a['Tls'] + b['Tls'], 'L': L,
            'max_sent': max([a['max_sent'], b['max_sent']]),
            'lorder': lorder,
            'f0': a['f0'] + b['f0']})

def to_gpu(x):
    if torch.cuda.is_available():
        return x.to('cuda')
    return x.to('cpu')