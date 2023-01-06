from collections import Counter, defaultdict
import numpy as np
from functools import reduce
import multiprocessing
from .fnlp import get_context
import json, torch

# purpose: aligns lists of tags to a sequence of whatevers, according to a covering (gold tokenization)
# arguments:
# - d: list (document) of lists (sentences) of strings (whatevers) representing a (rolled) document 
# - covering: list (document) of lists (sentences) of strings (tokens) representing a 'gold standard' tokenization for the document
# - ltypes: list (layer types) of strings (layer type names) representing the types of layers to be aligned
# - layers: list (layer types) of lists (document layers) of lists (sentence layers) of strings (tags) aligned to d's order
# prereqs:  d and covering must join to the same string, that is: assert "".join(d) == "".join(covering)
# output: updated versions of ltypes and layers, aligned the the sequence structure of the document's (d's) underlying tokenization
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

# purpose: process a document (a sequence of whatevers) for its generative statistics and align them for sequence learning
# arguments:
# - d: list (document) of lists (sentences) of strings (whatevers) representing a (rolled) document 
# - ltypes: list (layer types) of strings (layer type names) representing the types of layers to be aligned
# - layers: list (layer types) of lists (document layers) of lists (sentence layers) of strings (tags) aligned to d's order
# prereqs: ltypes and layers should first be processed by build_layers for alignment to d
# output: 
# - nvs: list (document) of lists (sentences) of booleans (novelties) representing the introduction of whatevers to the document
# - ess: list (document) of lists (sentences) of booleans (end of sentence signatures) representing the whatevers that ended sentences
# - eds: list (document) of lists (sentences) of booleans (end of document signatures) representing the whatevers that ended documents
def build_document(d, ltypes, layers):
    s_j = 0; Td = Counter()
    nvs, ess, eds = [], [], []
    for s_j, s in enumerate(d):
        nvs.append([]); ess.append([]); eds.append([])
        for i, t in enumerate(list(s)):
            eds[-1].append(False)
            if i == len(s) - 1:
                ess[-1].append(True)
            else:
                ess[-1].append(False)
            if t not in Td:
                # record the novelty and vocabulary expansion events
                nvs[-1].append(True)
            else:
                nvs[-1].append(False)
            Td[t] += 1
    if eds:
        eds[s_j][-1] = True
    return nvs, ess, eds

# purpose: process a covering (gold standard tokenization) for a boolean layer representing whatevers that end tokens
# arguments:
# - d: see build_layers
# - covering: see build_layers
# prereqs: d and covering must join to the same string, that is: assert "".join(d) == "".join(covering)
# output: 
# - eots: list (document) of lists (sentences) of booleans (end of token signatures) representing the whatevers that ended tokens
def build_eots(d, covering):
    eots = []
    for c_s, s in zip(covering, d):
        locs = np.cumsum(list(map(len, s)))
        c_locs = np.cumsum(list(map(len, c_s)))
        eots.append([loc in c_locs for loc in locs])
    return eots

# purpose: process a covering (gold standard tokenization) for a boolean layer representing whatevers that begin tokens
# arguments:
# - d: see build_layers
# - covering: see build_layers
# prereqs: d and covering must join to the same string, that is: assert "".join(d) == "".join(covering)
# output:
# - bots: list (document) of lists (sentences) of booleans (beginning of token signatures) representing the whatevers that began tokens
def build_bots(d, covering):
    bots = []
    for c_s, s in zip(covering, d):
        locs = np.cumsum(list(map(len, s))) - len(s[0])
        c_locs = np.cumsum(list(map(len, c_s))) - len(c_s[0])
        bots.append([loc in c_locs for loc in locs])
    return bots

# purpose: process a covering (gold standard tokenization) for a boolean layer representing whatevers that are tokens
# arguments:
# - d: see build_layers
# - covering: see build_layers
# prereqs: d and covering must join to the same string, that is: assert "".join(d) == "".join(covering)
# output:
# - iats: list (document) of lists (sentences) of booleans (atomic token signatures) representing the whatevers that are themselves tokens
def build_iats(d, covering):
    iats = []
    for c_s, s in zip(covering, d):
        locs = np.array([0] + list(np.cumsum(list(map(len, s)))))
        spns = set(list(zip(locs[:-1], locs[1:])))
        c_locs = np.array([0] + list(np.cumsum(list(map(len, c_s)))))
        c_spns = set(list(zip(c_locs[:-1], c_locs[1:])))
        iats.append([spn in c_spns for spn in sorted(spns)])
    return iats

# purpose: build all tag layers for a document and collect meta data
# arguments:
# - x: tuple of various data, serialized for spark distribution:
# -- doc: json serialized tokenized document (see build_layers)
# -- covering: list (see build_layers)
# -- ltypes: list (see build_layers)
# -- layers: list (see build_layers)
# - bc: dict (meta values and function) of variable control utilities for operating process_document as a distributed (map) process
# prereqs: must be built for spark compatability
# output: tuple of processed data
# - d: list document (see build_layers), deserialized from doc
# - layers: list (document) of lists (layers) of lists (sentences) of strings (tags) aligned to d
def process_document(x, bc = {}): 
    if not type(bc) == dict: # access broadcast variables through spark
        bc = dict(bc.value)

    doc, covering, ltypes, layers = json.loads(x) # d
    # once bc['tokenize'] serializes: replace d with doc
    d = [bc['tokenize'](s) for s in doc]
    ## build the higher-level training layers (gold linguistic tags)
    ## assures layers conform to sent-tok covering; tags projects to the whatever level
    ltypes, layers = build_layers(d, covering, ltypes, layers)
    nvs, ess, eds = build_document(d, ltypes, layers) # , meta
    # if there was a covering, fit the model to its segmentation (end of token prediction)
    # as well as any other cover layers (POS, etc.) onto the end of token-positioned whatevers
    if covering:
        its = build_iats(d, covering); bts = build_bots(d, covering); ets = build_eots(d, covering)
        layers = layers + [nvs, its, bts, ets, ess, eds]
        ltypes = ltypes + ['nov', 'iat', 'bot', 'eot', 'eos', 'eod']
    else:
        layers = layers + [nvs, ess, eds]
        ltypes = ltypes + ['nov', 'eos', 'eod']
    yield d, layers, ltypes

# purpose: count the co-occurrences of whatevers over any target sequence of tags (including whatevers, too) for a document
# arguments:
# - x: tuple of various data, serialized for spark distribution (see process_document)
# - bc: dict (meta values and function) of variable control utilities for operating process_document as a distributed (map) process (see process_document)
# prereqs: must be built for spark compatability
# output:
# - yielded counted co-occurrences between the whatever background distribution, layered over all target tags, and structured as Counter objects for downstream (possibly spark-based) reduce aggregation
def count(x, bc = {}):
    if not type(bc) == dict: # access broadcast variables through spark
        bc = dict(bc.value)
    docs, layers, ltypes = json.loads(x)
    dcons = [[get_context(i, list(unroll(d)), m = bc['m'])
              for s in [[t_s for ds in d for t_s in ds]] for i, t in enumerate(s)] for d in docs]
    # count whatevers and their positional abundance
    yield from Counter([((t,'form'), tuple(c))
                        for d, dcon in zip(docs, dcons)
                        for s in [[t_s for ds in d for t_s in ds]] for i, t in enumerate(s) for c in dcon[i] if t in bc['targets']]).items()
    yield from Counter([((t,'form'), tuple([str(c[1]), c[1], 'attn']))
                        for d, dcon in zip(docs, dcons)
                        for s in [[t_s for ds in d for t_s in ds]] for i, t in enumerate(s) for c in dcon[i] if t in bc['targets']]).items()
    # count additional layers and their positional abundance
    yield from Counter([((l,dltype), tuple(c))
                        for d, dcon, dltypes, dlayers in zip(docs, dcons, ltypes, layers) for dltype, dlayer in zip(dltypes, dlayers)
                        for s, s_l in [list(zip(*[(t_s, t_l) for ds, ds_l in zip(d, dlayer) for t_s, t_l in zip(ds, ds_l)]))]
                        for i, (t, l) in enumerate(list(zip(s,s_l))) for c in dcon[i] if t in bc['targets']]).items()
    ## accrue data for positional distributions
    yield from Counter([((l,dltype), tuple([str(c[1]), c[1], 'attn']))
                        for d, dcon, dltypes, dlayers in zip(docs, dcons, ltypes, layers) for dltype, dlayer in zip(dltypes, dlayers)
                        for s, s_l in [list(zip(*[(t_s, t_l) for ds, ds_l in zip(d, dlayer) for t_s, t_l in zip(ds, ds_l)]))]
                        for i, (t, l) in enumerate(list(zip(s,s_l))) for c in dcon[i] if t in bc['targets']]).items()

# purpose: move tensor to gpu (if available)
# arguments:
# - x: array of arbitrary dimension
# prereqs: gpu should be available, otherwise array will be maintained on cpu
# output: a same-dimension array, ideally on gpu
def to_gpu(x):
    if torch.cuda.is_available():
        return x.to('cuda')
    return x.to('cpu')

# purpose: make sure tensor is on cpu (detached from any gpu)
# arguments: 
# - x: array of arbitrary dimension
# prereqs: not much (numpy and torch)
# output: numpy array of arbitrary dimension
def detach(x):
    if type(x) == torch.Tensor:
        x = x.clone().detach().cpu()
    return np.array(x)

# purpose: flattens a list of lists of things into list of things
# arguments: 
# - d: a list of lists of things
# prereqs: nothing
# output: a flattened list of things from d
def unroll(d):
    return [t for s in d for t in s]

def stick_spaces(stream):
    tokens = []
    for wi, w in enumerate(stream):
        if not tokens:
            tokens.append(w)
        elif w == ' ':
            if (tokens[-1][-1] != ' ') and (wi != len(stream)-1):
                tokens.append(w)
            else:
                tokens[-1] = tokens[-1] + w
        else:
            if tokens[-1][-1] == ' ':
                tokens[-1] = tokens[-1] + w
            else:
                tokens.append(w)
    return(tuple(tokens))