import json, re
import numpy as np
from tqdm import tqdm
from abc import ABC
from abc import abstractmethod
from collections import Counter, defaultdict
from ..utils.fnlp import get_context
from ..utils.hr_bpe.src.bpe import HRBPE

class Whatever:
    def __init__(self, form, ix = None, sep = None, nov = None, oov = None, vec = None):
        self._form = str(form) 
        self._ix = int(ix) if ix is not None else None # note: all ix are scoped as referential 
        self._sep = bool(sep) if sep is not None else None  #  to the objects' starting character indices
        self._nov = bool(nov) if nov is not None else None
        self._oov = bool(oov) if oov is not None else None
        self._vec = np.array(vec) if vec is not None else None
    def __str__(self):
        return self._form
    def __repr__(self):
        return self._form
    def __eq__(self, other):
        return self._ix == other._ix
    def __gt__(self, other):
        return self._ix > other._ix
    def __lt__(self, other):
        return self._ix < other._ix
    def __ne__(self, other):
        return self._ix != other._ix
    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)
    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)
    def __len__(self):
        return len(self._form)
    def __contains__(self, other):
        return self.__ge__(other) and ((self._ix - other._ix) <= (other.__len__() - self.__len__()))
        
class Token(Whatever):
    def __init__(self, whatevers, ix = None, sep = None, nov = None, oov = None, vec = None,
                 lem = None, sen = None, pos = None, ent = None, 
                 dep = None, sup = None, infs = None):
        super(Token, self).__init__("".join([w._form for w in whatevers]), 
                                    ix = ix, sep = sep, nov = nov, oov = oov, vec = vec)
        self._whatevers = list(whatevers)
        self._lem = str(lem) if lem is not None else None
        self._sen = str(sen) if sen is not None else None
        self._pos = str(pos) if pos is not None else None
        self._ent = str(ent) if ent is not None else None
        self._dep = str(dep) if dep is not None else None
        self._sup = int(sup) if sup is not None else None
        self._infs = [int(inf) for inf in infs] if infs is not None else None
        
class Sentence(Token):
    def __init__(self, tokens, ix = None, sep = None, nov = None, oov = None, vec = None,
                 lem = None, sen = None, pos = None, ent = None, 
                 dep = None, sup = None, infs = None, tops = None):
        super(Sentence, self).__init__([w for t in tokens for w in t._whatevers], 
                                        ix = ix, sep = sep, nov = nov, oov = oov, vec = vec,
                                        lem = lem, pos = pos, dep = dep, 
                                        sup = sup, infs = infs)
        self._tokens = list(tokens)
        self._tops = [str(top) for top in tops] if tops is not None else None
        
class Document(Sentence):
    def __init__(self, sentences, ix = None, sep = None, nov = None, oov = None, vec = None,
                 lem = None, sen = None, pos = None, ent = None, 
                 dep = None, sup = None, infs = None, tops = None):
        super(Document, self).__init__([t for s in sentences for t in s._tokens], 
                                        ix = ix, sep = sep, nov = nov, oov = oov, vec = vec,
                                        lem = lem, pos = pos, dep = dep, 
                                        sup = sup, infs = infs)
        self._sentences = list(sentences)

class LM(ABC):
    def __init__(self, form2ind=None, covering_vocab = set()):
        self._covering_vocab = covering_vocab
        self._covered = {}
        self._covering = {}
        if self._covering_vocab:
            if form2ind:
                form2ind = {t: i for i, t in enumerate(set(list(form2ind.keys())+list(self._covering_vocab)))}
            else:
                form2ind = {t: i for i, t in enumerate(self._covering_vocab)}
        if form2ind is None:
            self._form2ind = {}
        else:
            self._form2ind = form2ind
        self._ind2form = {v: k for k, v in self._form2ind.items()}

    def __len__(self):
        return len(self._form2ind)

    def add_form(self, tok):
        if tok not in self._form2ind:
            self._form2ind[tok] = len(self._form2ind)
            self._ind2form[self._form2ind[tok]] = tok

    def del_form(self, tok):
        if tok in self._form2ind:
            idx = self._form2ind[tok]

            del self._ind2form[idx]
            del self._form2ind[tok]

            # shifting down each type that's a larger index
            # than the one just removed
            i = idx + 1
            while i in self._ind2form:
                t = self._ind2tok[i]
                self._form2ind[t] = i - 1
                self._ind2form[i - 1] = t

                del self._ind2form[i]

    def save(self, path, data=None):
        if data is None:
            data = {}
        # note: this is not yet complete and full models won't save
        #
        json.dump(data, open(path, 'w+'))

    def load(self, path):
        data = json.load(open(path))

        self._form2ind = data['form2ind']
        self._ind2form = {v: k for k, v in self._form2ind.items()}
        # note: this is not yet complete and full models won't load
        #
        return data

    def init(self, m = 10, zeta = 0.01, positional = True, seed=None):
        self._seed = seed
        if self._seed:
            np.random.seed(seed=self._seed)
        self._zeta = zeta
        self._positional = positional
        self._m = m
        self._padding = ['<pad>'] * self._m; self._padding[0] = '<sos>'
        self._X = Counter([((t,'form'),c) for s in [self._padding]
                           for i, t in enumerate(s)
                           for c in get_context(i, s, m = self._m, positional = self._positional)
                           if len(s) - 1])
        self._T, self._C, self._Tds = Counter(), Counter(), defaultdict(Counter)
        self._TCs = defaultdict(set); self._CTs = defaultdict(set); 
        self._D = defaultdict(set)
        self._Cp = Counter(); self._C_tot = 0
        for t, c in tqdm(self._X):
            self._X[(t,c)] = self._X[(t,c)]*self._zeta; self._T[t] += self._X[(t,c)]; self._C[c] += self._X[(t,c)]
            self._TCs[t].add(c); self._CTs[c].add(t)
            self._Cp[t] += self._X[(t,c)]; self._C_tot += self._X[(t,c)]

    # coverings and layers are tokenizations and tag sets, ltypes keys which
    def fit(self, docs, docs_name, covering = [], layers = [], ltypes = [], 
            method = 'hr-bpe', init_method = 'char', param_method = 'est_theta', 
            reg_model = 'mixing', early_stop = True, num_batches = 100, batch_size = 10_000,
            action_protect = ["\n","[*\(\{\[\)\}\]\.\?\!\,\;][ ]*\w", "\w[ ]*[*\(\{\[\)\}\]\.\?\!\,\;]"]):
        
        print("Training tokenizer...")
        actions_per_batch = int(batch_size/1)
        self._tokenizer_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{reg_model}_{param_method}_{self._seed}_{docs_name}'
        if not all([len("".join(s_c)) == len(s) for d_c, d in zip(covering, docs) for s_c, s in zip(d_c, d)]):
            covering = []
        else:
            self._covering_vocab = set()
            docs = [["".join(s) for s in d] for d in docs]
            
        self.tokenizer = HRBPE(param_method = param_method, reg_model = reg_model, early_stop = early_stop,
                               tok2ind = {}, covering_vocab = self._covering_vocab)
        self.tokenizer.init([s for d in docs for s in d], seed = self._seed, method = init_method, 
                            covering = [s for d in covering for s in d], action_protect = action_protect)
        self.tokenizer.fit(num_batches, batch_size, actions_per_batch=actions_per_batch, seed=self._seed)
        # tokenize the documents
        print('Tokenizing documents...')
        docs = [[self.tokenizer.tokenize(s) for s in d] for d in tqdm(docs)]
        
        ## build the higher-level training layers (gold linguistic tags)
        ## assure all layers are conforming to their own necessary alignments, and apply their tags at the whatever level
        ltypes, layers = self.build_layers(docs, covering, ltypes, layers)
                
        ##################################
        print('Fitting language model...')
        self.tune(docs)
        print('Indexing documents...')
        self._alphas = defaultdict(list); self._total_sentences = 0; self._total_tokens = 0; 
        novs, oovs, eoss, eods = self.index_documents(docs)
        assert(all([len(s_l) == len(s) for d_l, d in zip(novs, docs) for s_l, s in zip(d_l, d)]))
        assert(all([len(s_l) == len(s) for d_l, d in zip(oovs, docs) for s_l, s in zip(d_l, d)]))
        self.tune(docs, layer = novs, ltype = 'nov')
        self.tune(docs, layer = oovs, ltype = 'oov')
        self.tune(docs, layer = eoss, ltype = 'eos')
        self.tune(docs, layer = eods, ltype = 'eod')
        
        # if there was a covering, fit the model to its segmentation (end of token prediction)
        # as well as any other cover layers (POS, etc.) onto the end of token-positioned whatevers
        if covering:
            eots = self.build_eots(docs, covering)
            print('Fitting token boundaries...')
            self.tune(docs, layer = eots, ltype = 'eot')
            if ltypes:
                print('Fitting additional tag layers...')
                self.tune_layers(docs, layers, ltypes)

        # compute the marginals    
        self.compute_marginals()
        
        # report model statistics
        print('Done.')
        print('Model params, tokens, contexts, and % capacity used:', 
              len(self._X), len(self._T), len(self._C), round(100*len(self._X)/(len(self._T)*len(self._C)), 3))
        
    def build_eots(self, docs, covering):
        eots = []
        for c_d, d in zip(covering, docs):
            eots.append([])
            for c_s, s in zip(c_d, d):
                c_locs = np.cumsum(list(map(len, c_s)))
                eots[-1].append([loc in c_locs for loc in np.cumsum(list(map(len, s)))])
        return eots
        
    def build_layers(self, docs, covering, ltypes, layers):
        ## build the higher-level training layers (gold linguistic tags)
        ## assure all layers are conforming to their own necessary alignments, and apply their tags at the whatever level
        temp_ltypes = []
        temp_layers = []
        if (covering and ltypes) and len(ltypes) == len(layers):
            for layer, ltype in zip(layers, ltypes):
                if ltype in ['lem', 'pos', 'bio', 'dep', 'sup']:
                    if all([len(s_c) == len(s_l) for d_c, d_l in zip(covering, layer) for s_c, s_l in zip(d_c, d_l)]):
                        temp_ltypes.append(ltype)
                        print(f'Processing {ltype}-tags for whatever-layer prediction...')
                        temp_layers.append([])
                        for d_i, layer_doc in tqdm(list(enumerate(layer))):
                            temp_layers[-1].append([])
                            for s_i, layer_sent in enumerate(layer_doc):
                                layer_sent = np.array(layer_sent)
                                layer_norms = np.cumsum([len(t) for t in covering[d_i][s_i]])
                                whatever_norms = np.cumsum([len(t) for t in docs[d_i][s_i]])
                                temp_layers[-1][-1].append([layer_sent[whatever_norm <= layer_norms][0] for
                                                            whatever_norm in whatever_norms])
                    else:
                        print([(len(s_c), len(s_l)) for d_c, d_l in zip(covering, layer) for s_c, s_l in zip(d_c, d_l)])
            ltypes = list(temp_ltypes); del(temp_ltypes)
            layers = list(temp_layers); del(temp_layers)
        return ltypes, layers
    
    def index_documents(self, docs, modify_model = True):
        novs = []; oovs = []; eoss = []; eods = [] ## collect other whatever-level layers
        for d_i, d in tqdm(list(enumerate(docs))):
            delta = 0; M = 0
            novs.append([]); oovs.append([]); eoss.append([]); eods.append([])
            for s_j, s in enumerate(d):
                if modify_model:
                    self._total_sentences += 1
                novs[-1].append([]); oovs[-1].append([]); eoss[-1].append([]); eods[-1].append([])
                for i, t in enumerate(list(s)): 
                    eods[-1][-1].append(False)
                    if i == len(s) - 1:
                        eoss[-1][-1].append(True)
                    else:
                        eoss[-1][-1].append(False)
                    delta += 1; M += 1
                    if t not in self._Tds[d_i]:
                        # record the novelty and vocabulary expansion events
                        novs[-1][-1].append(True)
                        if t not in self._D:
                            oovs[-1][-1].append(True)
                        else:
                            oovs[-1][-1].append(False)
                        # record the word introduction rate
                        alpha = 1/delta
                        if modify_model:
                            for Mt in range(M-delta,M+1):
                                self._alphas[Mt].append(alpha)
                        delta = 0
                    else:
                        novs[-1][-1].append(False)
                        oovs[-1][-1].append(False)    
                    if modify_model:
                        # increment the token's document frequency and
                        # record the tokens presence in the document
                        self._Tds[d_i][t] += 1
                        self._D[t].add(d_i)
            if modify_model:
                self._total_tokens += M
            eods[d_i][s_j][-1] = True
        return novs, oovs, eoss, eods # = index_documents(self, docs, modify_model = True)
        
    def compute_marginals(self):
        print('Computing marginal statistics...')
        self._ltypes = defaultdict(set)
        for t, c in tqdm(self._X):
            self._T[t] += self._X[(t,c)]; self._C[c] += self._X[(t,c)]; self._C_tot += self._X[(t,c)]
            self._TCs[t].add(c); self._CTs[c].add(t)
            # check to make sure all target forms are in the LM's vocabulary
            if t[0] or (type(t[0]) == bool):
                self.add_form(t[0])
                if t[1] and t[0] != '<pad>' and t[0] != '<sos>':
                    self._ltypes[t[1]].add(t[0])
        self._Cp = {t: sum([self._C[c] for c in self._TCs[t]] + [0]) for t in self._T}
        self._Tp = {c: sum([self._T[t] for t in self._CTs[c]] + [0])/self._C_tot for c in self._C}
        self._beta = {t: self._Cp[t]/self._C_tot for t in self._T}
        self._Cn = {t: len(self._C) - len(self._TCs[t]) for t in self._T}
        self._Tn = {c: len(self._T) - len(self._CTs[c]) for c in self._C}
        self._missing_mass = sum([(1 - self._beta[t])/self._Cn[t] for t in self._Cn])
        
    def tune(self, docs, layer = None, ltype = 'form'):
        if layer is not None:
            if not all([len(s_l) == len(s) for d_l, d in zip(layer, docs) for s_l, s in zip(d_l, d)]):
                layer = None
                ltype = 'form'
        else:
            layer = None
            ltype = 'form'
        
        if layer is None:
            layer = list(docs)
            ltype = 'form'
            
        print(f'Absorbing {ltype}-layer co-occurrences...')
        self._X += Counter([((l,ltype), c) for d, d_l in tqdm(list(zip(docs, layer))) for s, s_l in zip(d, d_l)
                            for i, (t, l) in enumerate(
                                list(zip(self._padding, ['']*self._m)) + 
                                list(zip(s,s_l))
                            )
                            for c in get_context(i, self._padding + list(s), m = self._m, positional = self._positional)])
        
    def tune_layers(self, docs, layers, ltypes):
        for layer, ltype in zip(layers, ltypes):
            self.tune(docs, layer = layer, ltype = ltype)

    # likely, none of these are necessary, unless they pertain to token-level vocab elements
    def encode(self, text):
        return self.tokens_to_indices(self.tokenize(text))    

    def decode(self, indices):
        return ''.join(self.indices_to_tokens(indices))

    def tokens_to_indices(self, toks):
        return [self._form2ind[t] for t in toks]

    def indices_to_tokens(self, indices):
        return [self._ind2form[i] for i in indices]

    # likely going to be issues in the below with '<new>' tokens
    def NLL(self, t, s, i): 
        Csum = sum([self._C.get(c,0.) 
                    for c in get_context(i, s, m = self._m, positional = self._positional)])
        if not Csum: Csum = self._missing_mass
        return sum([-np.log10(self._T[t]/Csum),
                    -np.log10([(self._beta[t]*(self._X[(t,c)]/self._T[t]) if (t,c) in self._X else 
                                (1 - self._beta[t])/self._Cn[t])
                               for c in get_context(i, s, m = self._m, positional = self._positional)
                              ]).sum()])

    
    def predict(self, s, i, Td = Counter(), damping = 0.01, seed = None):
        if seed is not None:
            np.random.seed(seed)
            
        if i < 0:
            i = len(s) + i
        if i >= len(s):
            i = len(s) - 1
        
        LM_Ps = {(t, 'form'): -self.NLL((t, 'form'), s, i) for t in self._ltypes['form'] if t}
        MaxPs_list = [LM_Ps[t] for t in LM_Ps if not np.isinf(LM_Ps[t])]
        MaxPs = np.max(MaxPs_list) if MaxPs_list else 0
        if MaxPs:
            LM_Ps = {t: (10**(LM_Ps[t] - MaxPs) if not np.isinf(LM_Ps[t]) else LM_Ps[t]) for t in LM_Ps}
            Psum = sum([xyz for xyz in LM_Ps.values() if not np.isinf(xyz)] + [0])
            LM_Ps = Counter({t: (LM_Ps[t]/Psum if not np.isinf(LM_Ps[t]) else 0) for t in LM_Ps})
        else:
            LM_Ps = Counter({t:0 for t in self._ltypes['form'] if t}); LM_Ps[''] = 1.#########################

        if Td:
            # damping tells how strongly to attend to document specificty, with document-occurrence statistics for noise
            # the smaller the damping factor is, the more strongly the noise will re-enforce similar-document vocabulary
            ##
            Dsums = {d_i: sum(self._Tds[d_i].values()) + len(self._D)*damping for d_i in self._Tds} # this should be a marginalized pre-compute
            DPs = {d_i: -sum([Td[t]*np.log10((self._Tds[d_i].get(t, 0) + damping)/Dsums[d_i]) for t in Td]) 
                   for d_i in self._Tds}
            
            MaxPs_list = [DPs[d_i] for d_i in DPs if not np.isinf(DPs[d_i])]
            MaxPs = np.max(MaxPs_list) if MaxPs_list else 0
            if MaxPs:
                DPs = {d_i: (10**-(DPs[d_i] - MaxPs) if not np.isinf(DPs[d_i]) else DPs[d_i]) for d_i in DPs}
                Psum = sum([xyz for xyz in DPs.values() if not np.isinf(xyz)] + [0])
                DPs = Counter({d_i: (DPs[d_i]/Psum if not np.isinf(DPs[d_i]) else 0) for d_i in DPs})
            else:
                DPs = Counter({d_i:1/self._Tds for d_i in self._Tds})
            ##
            # resonance provides nuance in modeling prob random doc contains token (specificity)
            # term frequency makes this object (e.g., weighted as immediately below ---> leave commented) un-useful, 
            # i.e., noise distributions should be uniform-ish
            # noise_Ps = Counter({t: sum([DPs[d_i]*(self._Tds[d_i].get(t[0], 0) + damping)/Dsums[d_i] for d_i in self._Tds]) for t in LM_Ps}) #### 
            noise_Ps = {t: sum([DPs[d_i]*(1. if d_i in self._D[t[0]] else 0.) for d_i in self._Tds]) for t in LM_Ps}
            norm_const =  sum(noise_Ps.values())
            noise_Ps = Counter({t: (noise_Ps[t]/norm_const)# *damping + (1. - damping)*(len(self._D[t[0]])/len(self._Tds))
                                for t in LM_Ps})
        else:
            # uses the document-frequency distribution (truncated uniform, by rank ---> uniform-ish distribution)
            noise_Ps = {t: len(self._D[t[0]])/len(self._Tds) for t in LM_Ps}
            norm_const =  sum(noise_Ps.values())
            noise_Ps = Counter({t: (noise_Ps[t]/norm_const)# *damping + (1. - damping)*(len(self._D[t[0]])/len(self._Tds))
                                for t in LM_Ps})
            # these are other potential noise-sampling models, which operate on unigram and uniform frequencies
            ## noise_Ps = {t: self._T.get(t, 1)/(self._C_tot + 1) for t in LM_Ps} # uses the term-frequency distribution
            ## noise_Ps = {t: 1/(len(self._T) - len(Td) + 1) for t in LM_Ps} # uses the uniform distribution
            
        # the novelty clamping, the average (1 - theta) appears to stabilize generation
        alpha = (np.mean(self._alphas.get(int(sum(Td.values())),
                                          [min([x for x in map(np.mean, self._alphas.values()) if x])])) if Td else 1-len(self._D)/self._total_tokens)
        # sigma models the local probability of a novel word introduction, 
        # when whose compliment is blended with the compliment of epsilon,
        # (i.e., the local probability of an oov-token) as well as the coverage rate
        # of the current locality of contexts determines how much smoothing to self-impose with noise
        nov_Ps = Counter({(t, 'nov'): 10**-self.NLL((t, 'nov'), s, i) for t in self._ltypes['nov']})
        nov_Ps = {t: nov_Ps[t]/sum(nov_Ps.values()) for t in nov_Ps}
        oov_Ps = Counter({(t, 'oov'): 10**-self.NLL((t, 'oov'), s, i) for t in self._ltypes['oov']})
        oov_Ps = {t: oov_Ps[t]/sum(oov_Ps.values()) for t in oov_Ps}

        # smooths the local distributional model with the noise model according to a combination of factors:
        # more/less coverage from the contexts, and certainty of generation w/r novelty & vocabulary expansion (not sure if functional)
        # coverage = np.mean([self._Tp.get(c,0) for c in get_context(i, s, m = self._m, positional = self._positional)])
        # epsilon = ((1 - coverage)*(1 - nov_Ps[(True, 'nov')])*oov_Ps[(True, 'oov')])**(1/3)
        epsilon = ((1 - nov_Ps[(True, 'nov')])*oov_Ps[(True, 'oov')])**(1/2)
        P = Counter({t: LM_Ps[t]*(1 - epsilon) + noise_Ps[t]*epsilon for t in LM_Ps})

        # gets the total noise-local smoothed probability covered by the current document
        TdP_tot = sum([P[t]  for t in P if (t[0] in Td)] + [0])
        # clamps unseen words in this document over the noise-local smoothed distribution via alpha
        if TdP_tot:
            P = Counter({t: (P[t]*(1 - alpha)/TdP_tot if (t[0] in Td) else #   and TdP_tot
                             P[t]*alpha)/(1 - TdP_tot) for t in P})
        Ptot = sum(P.values())

        return Counter({ti: P[ti]/Ptot for ti in P}), alpha, epsilon
    
    def predict_layer(self, s, ltype, i = 0): 
        layer_Ps = Counter({(t, ltype): 10**-self.NLL((t, ltype), s, i) # (i if i else len(s)-1)
                            for t in self._ltypes[ltype]})
        Psum = sum(layer_Ps.values())
        return Counter({t: layer_Ps[t]/Psum for t in layer_Ps})
    
    def sample_layer(self, s, ltype, i = 0, seed = None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self._seed)
        layer_Ps = self.predict_layer(s, ltype, i = i) # note: we get as many predictions as there are within the separated component
                                                       # consider 'copies in': `for past_is_sep in is_seps[::-1]:`, below.
                                                       # this means on_sep-type tag predictions, e.g., POS, 
                                                       # can be blended with attention wights as a batch prediction,
                                                       # attention weights can be computed from the _real_ CBOW model, 
                                                       # which utilizes wave superposition to form temporal weights
                                                       # and will be built separately. for now, trailing whatevers inherit
                                                       # the separator's tag prediction, and the POS contexts are recorded via 
                                                       # for all whatevers, regardless of if they segment to the next layer
        layer_ts, layer_ps = map(np.array, zip(*layer_Ps.most_common()))
        layer_that = np.random.choice([x[0] for x in layer_ts], size=None, 
                                      replace=True, p=layer_ps/layer_ps.sum())
        return layer_that, layer_ps
    
    def grok(self, wi, w, eot, nov, oov, vec, seed, pred_eos, pred_eod):
        self._whatevers.append(Whatever(w, ix = self._ix, sep = eot, nov = nov, oov = oov, vec = vec))
        self._w_set.add(w)
        self._ix += len(w)
        # self._talking = False
        # apply token-level information to build the next level sequence
        seps = [eot]
        if eot:
            kwargs = {ltype: ((self._d_layers[self._d_ltypes.index(ltype)][self._d_i][self._s_i][wi] 
                               if ltype in self._d_ltypes else self.sample_layer(self._s, ltype, i = wi, seed = seed)[0])
                              if ltype in self._ltypes else None)
                      for ltype in ['lem', 'sen', 'pos', 'ent', 'dep', 'sup', 'infs']}
            kwargs['sep'] = ((self.sample_layer(self._s, 'eos', i = wi, seed = seed)[0] == "True") 
                             if pred_eos else self._eoss[self._d_i][self._s_i][wi])
            seps.append(kwargs['sep'])
            self._tokens.append(Token(self._whatevers, ix = self._ix - len("".join([w._form for w in self._whatevers])), **kwargs))
            self._whatevers = []
            # if self._generate_next == 't':
            #     self._talking = False
            # apply sentence-level information to build the next level sequence
            if kwargs['sep']:
                kwargs = {'tops': [((self._d_layers[self._d_ltypes.index(ltype)][self._d_i][self._s_i][wi] 
                                    if ltype in self._d_ltypes else self.sample_layer(self._s, ltype, i = wi, seed = seed)[0])
                                   if ltype in self._ltypes else None)
                                   for ltype in self._ltypes if ltype[:3] == 'top']}
                kwargs['sep'] = ((self.sample_layer(self._s, 'eod', i = wi, seed = seed)[0] == "True") 
                                 if pred_eod else self._eods[self._d_i][self._s_i][wi])
                seps.append(kwargs['sep'])
                self._sentences.append(Sentence(self._tokens, 
                                                ix = self._ix - len("".join([w._form for t in self._tokens for w in t._whatevers])), **kwargs))
                self._tokens = []
                self._s = []
                # if self._generate_next == 's':
                #     self._talking = False
                if kwargs['sep']:
                    self._documents.append(Document(self._sentences, 
                                                    ix = self._ix - len("".join([w._form for s in self._sentences 
                                                                                 for t in s._tokens for w in t._whatevers]))))
                    self._w_set = set(); self._ix = 0
                    self._sentences = []
                    # if self._generate_next == 'd':
                    #     self._talking = False
        # who knows perhaps a token sep
        # print(seps, self._eots)
        # self._eots[-1][-1].append(seps[0])
        if len(seps) == 1: 
            self._eoss[-1][-1].append(False)
            self._eods[-1][-1].append(False)
        # definitely a sentence sep, hence a token sep
        if len(seps) == 2: 
            self._eoss[-1][-1].append(seps[1])
            self._eods[-1][-1].append(False)
        # definitely a document sep, hence all seps
        if len(seps) == 3: 
            self._eoss[-1][-1].append(seps[1])
            self._eods[-1][-1].append(seps[2])

    def digest(self):
        if self._whatevers:
            kwargs = {}
            self._tokens.append(Token(self._whatevers, ix = self._whatevers[0]._ix, **kwargs))
            self._whatevers = []
        if self._tokens:
            kwargs = {}
            self._sentences.append(Sentence(self._tokens, ix = self._tokens[0]._ix, **kwargs))
            self._tokens = []
        if self._sentences:
            self._documents.append(Document(self._sentences, ix = self._sentences[0]._ix))
            self._sentences = []

    def interpret(self, docs, Td = Counter(), damping = 0.01, seed = None, resonant = False, top = 0.99, covering = [], 
                  ltypes = [], layers = [], vecs = True, pred_eos = True, pred_eod = True, digest = True):
        if seed is not None:
            np.random.seed(seed)
        #############
        ## assert that _raw sentences_ (to be HR-BPE tokenized) is the only requirement for IaMaN-LM processes
        assert(all([type(text) == str for doc in docs for text in doc]))
        # build the model's whatever sequence
        print('Tokenizing documents...')
        docs = [[self.tokenizer.tokenize(text) for text in doc] for doc in tqdm(docs)]
        # gather the boundary, vocab, and novelty information for the whatevers
        print('Indexing documents...')
        self._novs, self._oovs, self._eoss, self._eods = self.index_documents(docs, modify_model = False)
        # build up any known gold layers
        self._d_ltypes, self._d_layers = self.build_layers(docs, covering, ltypes, layers)
        # self._eots = self.build_eots(docs, covering) if covering else []
        eots = self.build_eots(docs, covering) if covering else []
        # build the document
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        print("Interpreting documents")
        for d_i, doc in tqdm(list(enumerate(docs))):
            self._d_i = d_i
            self._w_set = set(); self._ix = 0
            for s_i, s in enumerate(doc):
                self._s_i = s_i; self._s = s
                # iterate over whatevers to form base sequence
                for wi, w in enumerate(self._s):
                    eot = eots[self._d_i][self._s_i][wi] if eots else (self.sample_layer(s, 'eot', i = wi, seed = seed)[0] == "True")
                    vec = None
                    if vecs:
                        P, _, _ = self.predict(self._s, wi, Td = Td + (Counter(self._s[:-1]) if resonant else Counter()), 
                                               damping = damping, seed = seed)
                        vec = -np.log10([P[t] for t in P]) - np.log10(len(P))
                    
                    self.grok(wi, w, eot, w in self._w_set, (w, 'form') in self._T, vec, seed, pred_eos, pred_eod)
            
        if digest:
            self.digest()
    
    def generate(self, m = 1, docs = [], Td = Counter(), damping = 0.01, seed = None, resonant = False, top = 0.99, covering = [], 
                 ltypes = [], layers = [], pred_eos = True, pred_eod = True, generate_next = 't'):
        if seed is not None:
            np.random.seed(seed)
        #############
        ## assert that _raw sentences_ (to be HR-BPE tokenized) is the only requirement for IaMaN-LM processes
        assert(all([type(text) == str for doc in docs for text in doc]))
        if docs:
            self.interpret(docs, Td = Td, damping = damping, seed = seed, resonant = resonant, top = top, covering = covering, 
                           ltypes = ltypes, layers = layers, pred_eos = pred_eos, pred_eod = pred_eod, vecs = True, digest = False)
        else:
            self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        try:
            self._s
        except:
            self._s = [""]

        sampled = 0; talking = True
        while talking:
            
            if not self._s:
                self._s = list(self._padding) + [""]
            elif self._s[-1]:
                self._s = list(self._s) + [""]
            
            wi = len(self._s) - 1
            eot = self.sample_layer(self._s, 'eot', i = wi, seed = seed)[0] == "True"
            P, _, _ = self.predict(self._s, wi, Td = Td + (Counter(self._s[:-1]) if resonant else Counter()), 
                                   damping = damping, seed = seed)
            vec = -np.log10([P[t] for t in P]) - np.log10(len(P))
            ts, ps = map(np.array, zip(*P.most_common()))
            if type(top) != int:
                top = len(ps[ps[::-1].cumsum() <= top])
            ts, ps = ts[:top], ps[:top]
            self._s[-1] = np.random.choice([t[0] for t in ts], size=None, replace=True, p=ps/ps.sum()) # note: there's no reason why whatever-type
            sampled += 1                                                                               # has to come before other layers, i.e., 
            w = self._s[-1]                                                                            # we can co-predict with layers, 
                                                                                                       # or reverse prediction orders
            self.grok(wi, w, eot, w in self._w_set, (w, 'form') in self._T, vec, seed, pred_eos, pred_eod)
            if not self._whatevers: # just made a new token
                if generate_next == 't':
                    talking = False
                if not self._tokens: # just made a new sentence
                    if generate_next == 's':
                        talking = False
                    self._s_i += 1
                    if not self._sentences: # just made a new document
                        talking = False
                        self._s_i = 0
                        self._d_i += 1
            if sampled == m: talking = False
        self.digest()

    