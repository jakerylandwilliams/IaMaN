import json, re, torch
import numpy as np
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
from itertools import groupby
from functools import reduce, partial
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from .types import Dummy, Whatever, Token, Sentence, Document
from ..utils.fnlp import get_context, wave_index, get_wave
from ..utils.munge import process_document, aggregate_processed, build_layers, build_document, build_eots, build_iats, count
from ..utils.stat import agg_vecs, blend_predictions
from ..utils.hr_bpe.src.bpe import HRBPE
from ..utils.hr_bpe.src.utils import tokenize, sentokenize
from pyspark import SparkContext

class LM(ABC):
    def __init__(self, m = 10, tokenizer = 'hr-bpe', noise = 0.001, positionally_encode = 'c', seed = None, positional = 'dependent',
                 space = True, attn_type = [False], do_ife = True, runners = 0, hrbpe_kwargs = {}, gpu = False): 
        self._tokenizer_name = str(tokenizer); self._runners = int(runners); self._gpu = bool(gpu)
        if self._tokenizer_name == 'hr-bpe':
            self._hrbpe_kwargs = {'method': 'char', 'param_method': 'est_theta', 'reg_model': 'mixing', 
                                  'early_stop': True, 'num_batches': 100, 'batch_size': 10_000,
                                  'action_protect': ["\n","[*\(\{\[\)\}\]\.\?\!\,\;][ ]*\w", 
                                                     "\w[ ]*[*\(\{\[\)\}\]\.\?\!\,\;]"],
                                 } if not hrbpe_kwargs else dict(hrbpe_kwargs)        
        self._seed = int(seed); self._space = bool(space); self._attn_type = list(attn_type); self._do_ife = bool(do_ife)
        self._cltype = 'frq' if self._do_ife else 'form'
        self._lorder = list(); self._ltypes = defaultdict(set) 
        if self._seed:
            np.random.seed(seed=self._seed)
        self._positional = str(positional); self._positionally_encode = str(positionally_encode)
        self._noise = noise; self._m = m; self._As = {}; self._Ls = {}
        self._X = Counter(); self._F = Counter(); self._Xs = {}; self._Fs = defaultdict(Counter)
        self._ife = Counter(); self._tru_ife = Counter(); self._f0 = Counter()
        self._trXs = {}; self._trIs = {}
        self._Tds, self._Tls = defaultdict(Counter), defaultdict(lambda : defaultdict(Counter))
        self._D = defaultdict(set); self._L = defaultdict(lambda : defaultdict(set))
        self._alphas = defaultdict(list); self._total_sentences = 0; self._total_tokens = 0; self._max_sent = 0
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        self.array = partial(torch.tensor, dtype = torch.double) if self._gpu else np.array
        self.cat = partial(torch.cat,  axis = -1) if self._gpu else np.concatenate

    # coverings and layers are tokenizations and tag sets
    def fit(self, docs, docs_name, covering = [], all_layers = defaultdict(lambda : defaultdict(list)), 
            fine_tune = False):
        # assure all covering segmentations match their documents
        if not all([len("".join(s_c)) == len(s) for d_c, d in zip(covering, docs) for s_c, s in zip(d_c, d)]):
            covering = []
        else:
            docs = [["".join(s) for s in d] for di, d in enumerate(docs) if len(covering[di])]
            covering = [list(cover) for di, cover in enumerate(covering) if len(covering[di])]
            all_layers = ({di: {ltype: all_layers[di][ltype] for ltype in all_layers[di] 
                                if len(covering[di])} for di in all_layers} 
                          if all_layers else defaultdict(lambda : defaultdict(list)))
        
        # train hr-bpe, if necessary
        if self._tokenizer_name == 'hr-bpe':
            if 'seed' not in self._hrbpe_kwargs:
                self._hrbpe_kwargs['seed'] = self._seed
            self._hrbpe_kwargs['covering'] = []; self._hrbpe_kwargs['covering_vocab'] = set()
            if covering:
                self._hrbpe_kwargs['covering'] = list([s for d in covering for s in d])
                if 'covering_vocab' not in self._hrbpe_kwargs:
                    self._hrbpe_kwargs['covering_vocab'] = set([t for d in covering for s in d for t in s])
            self._hrbpe_kwargs['actions_per_batch'] = int(self._hrbpe_kwargs['batch_size']/1)    
            print("Training tokenizer...")
            self.tokenizer = HRBPE(**{kw: self._hrbpe_kwargs[kw] 
                                      for kw in ['param_method', 'reg_model', 'early_stop', 'covering_vocab']})
            self.tokenizer.init([s for d in docs for s in d], 
                                **{kw: self._hrbpe_kwargs[kw] 
                                   for kw in ['seed', 'method', 'covering', 'action_protect']})
            self.tokenizer.fit(self._hrbpe_kwargs['num_batches'], self._hrbpe_kwargs['batch_size'], 
                               **{kw: self._hrbpe_kwargs[kw] for kw in ['seed', 'actions_per_batch']})
        elif self._tokenizer_name == 'sentokenizer':
            self.tokenizer = Dummy()
            self.tokenizer.tokenize = tokenize
            self.tokenizer.sentokenize = sentokenize
        else:
            self.tokenizer = Dummy()
            self.tokenizer.tokenize = lambda text: [t for t in re.split('( )', text) if t]
        # define the tokenizer   
        def tokenize(text):
            if self._space:
                return self.tokenizer.tokenize(text)
            else:
                stream = list(self.tokenizer.tokenize(text))
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
        # attach the tokenizer
        self.tokenize = tokenize
        # tokenize documents and absorb co-occurrences
        all_data = self.process_documents(docs, all_layers, covering) # , modify_model = True
        # compute the marginals
        print('Computing marginal statistics...')
        self.compute_marginals()
        # build dense models
        print('Building dense output heads...')
        self.build_dense_output()
        # build transition matrices for tag decoding
        self.build_trXs(all_data)
        # fine-tune a model for/using attention over predicted contexts
        if fine_tune:
            self.fine_tune(docs, covering = covering, all_layers = all_layers)
        # report model statistics
        print('Done.')
        print('Model params, types, encoding size, contexts, vec dim, max sent, and % capacity used:', 
              len(self._Fs[self._cltype]), len(self._ltypes['form']), len(self._ltypes[self._cltype]), 
              len(self._con_vocs[self._cltype]), self._vecdim, self._max_sent,
              round(100*len(self._Fs[self._cltype])/(len(self._con_vocs[self._cltype])*len(self._ltypes['form'])), 3))
        
    ## now re-structured to train on layers by document
    def process_documents(self, docs, all_layers = defaultdict(lambda : defaultdict(list)), 
                          covering = [], update_ife = False, update_bow = False):
        if (covering and self._tokenizer_name == 'hr-bpe') or (not covering): print('Tokenizing documents...')
        docs = ([json.dumps([self.tokenize(s) for s in doc]) for doc in tqdm(docs)]
                if (covering and self._tokenizer_name == 'hr-bpe') or (not covering) else [json.dumps(d) for d in covering])
        d_is = range(len(docs))
        all_data = list(zip(docs, [covering[d_i] if covering else [] for d_i in d_is], # docs and covering
                            [list(all_layers[d_i].keys()) for d_i in d_is], # ltypes
                            [list(all_layers[d_i].values()) for d_i in d_is])) # layers
        Fs = []; old_ife = Counter(self._ife)
        bc = {'m': int(self._m), # 'tokenizer': self.tokenize,
              'old_ife': Counter(self._ife)} 
        if self._runners:
            print('Counting documents and aggregating counts...')
            sc = SparkContext(f"local[{self._runners}]", "IaMaN", self._runners)
            bc = sc.broadcast(bc)
            Fs = Counter({ky: ct for (ky, ct) in tqdm(sc.parallelize(all_data)\
                                                        .flatMap(partial(count, bc = bc), preservesPartitioning=False)\
                                                        .reduceByKey(lambda a, b: a+b, numPartitions = int(len(docs)/10) + 1)\
                                                        .toLocalIterator())})
            sc.stop()
        else:
            print('Counting documents...')
            all_Fs = [Counter({ky: ct for (ky, ct) in count([doc, cover, ltypes, layers], bc = bc)})
                      for d_i, (doc, cover, ltypes, layers) in tqdm(list(enumerate(all_data)))]
            print("Aggregating counts...")
            Fs = reduce(lambda a, b: a + b, tqdm(all_Fs))
        print('Collecting metadata...')
        processed = [process_document([doc, cover, ltypes, layers])[3]
                     for d_i, (doc, cover, ltypes, layers) in tqdm(list(enumerate(all_data)))]
        print("Aggregating metadata...")
        meta = reduce(aggregate_processed, tqdm(processed))
        ## now modifies the model outside of the lower-level functions
        self._total_sentences += meta['total_sentences']
        self._total_tokens += meta['M']
        for Mt in meta['alphas']:
            for alpha in meta['alphas'][Mt]:
                self._alphas[Mt].append(alpha)
        for Td, Tl in zip(meta['Tds'], meta['Tls']):                
            d_i = len(self._Tds)
            self._Tds[d_i] = Td
            for t in Td:
                self._D[t].add(d_i)
            for ltype in Tl:
                for l in Tl[ltype]:
                    self._Tls[ltype][l] += Tl[ltype][l]
        for ltype in meta['L']:
            for t in meta['L'][ltype]:
                self._L[ltype][t] = self._L[ltype][t].union(meta['L'][ltype][t])
        self._max_sent = max([meta['max_sent'], self._max_sent])
        for ltype in meta['lorder']:
            if ltype not in self._lorder: 
                self._lorder.append(ltype)
        if not self._f0 or update_bow:
            self._f0 += meta['f0']
            self._M0 = sum(self._f0.values())
            self._p0 = Counter({t: self._f0[t]/self._M0 for t in self._f0})
        
        ## sets the ife for encoding
        self._ltypes = defaultdict(set); self._ctypes = set()
        self._ltypes['form'].add(''); self._ltypes['frq'].add(0)
        self._ltypes['form-attn'].add(''); self._ltypes['frq-attn'].add(0)
        self._F += Fs
        if not self._ife or update_ife:
            self._ife[''] = 0
            for t, c in Fs:
                if c[-1] == 'form': 
                    self._ife[c[0]] += Fs[(t,c)]
            self._ife_tot = sum(self._ife.values())
        self._Fs = defaultdict(Counter)
        print("Encoding parameters...")
        for t, c in tqdm(self._F):
            ent, enc, intensity = self.encode(t, c)
            impact = intensity*self._F[(t,c)]
            if not impact: print(t)
            ltype = t[1]+'-'+enc[-1] if enc[-1] == 'attn' else t[1]
            self._Fs[ltype][((t[0], ltype),enc)] += impact
            self._ltypes[ltype].add(t[0])
            self._ctypes.add(enc[-1])
            if enc[-1] == 'attn':
                self._ltypes['attn'].add(enc[0])
            if t[1] == 'form':
                ltype = ent[1]+'-'+enc[-1] if enc[-1] == 'attn' else ent[1]; ent = (ent[0], ltype)
                self._Fs[ltype][(ent,enc)] += impact
                self._ltypes[ltype].add(ent[0])

        # sets the vocabularies according to the current accumulated collection of data in a unified way
        # these are the individual layers' vocabularies
        self._zeds, self._idxs, self._vocs = {}, {}, {}
        for ltype in self._ltypes:
            self._vocs[ltype] = [(t, ltype) for t in self._ltypes[ltype]]
            self._idxs[ltype] = {t: idx for idx, t in enumerate(self._vocs[ltype])}
            self._zeds[ltype] = np.zeros(len(self._vocs[ltype]))
        self._vocs['attn'] = sorted(self._vocs['attn'], key = lambda t: int(t[0]))
        # set dense bow vector-probabilities for the language model to fall back to when contextless
        self._P0 = {}
        self._P0['form'] = self.array([self._f0.get(t[0], 0) for t in self._vocs['form']])
        self._P0['form'][self._idxs['form'][('', 'form')]] = 1
        self._P0['form'] = self._P0['form']/self._P0['form'].sum()
        f_ife = Counter()
        for t in self._f0:
            f_ife[self._ife[t]] += self._f0[t]
        self._P0['frq'] = self.array([f_ife.get(t[0], 0) for t in self._vocs['frq']])
        self._P0['frq'][self._idxs['frq'][(0, 'frq')]] = 1
        self._P0['frq'] = self._P0['frq']/self._P0['frq'].sum()
        
        # these are the combined context vocabularies, which spread the vocabs across the entire (2m+1) positional range
        self._con_vocs = defaultdict(list); self._con_idxs = defaultdict(dict)
        
        for ctype in self._ctypes:
            if self._positional == 'dependent':
                self._con_vocs[ctype] = [(t,rad,ltype) for rad in range(-self._m,self._m+1) for t, ltype in self._vocs[ctype]] 
            else:
                self._con_vocs[ctype] = [(t,0,ltype) for t, ltype in self._vocs[ctype]] 
            self._con_idxs[ctype] = {c: idx for idx, c in enumerate(self._con_vocs[ctype])}
        ##
        ##
        # these are pre-computed for the positional encoding
        if self._positional == 'dependent':
            # if 'c' in self._positionally_encode:
            self._positional_intensities = self.array([self.intensity(t, abs(rad)) for rad in range(-self._m,self._m+1) 
                                                        for t, _ in self._vocs[self._cltype]])
        elif self._positional == 'independent':
            self._positional_intensities = [self.array([self.intensity(t, abs(rad)) for t, _ in self._vocs[self._cltype]])
                                            for rad in range(-self._m,self._m+1)]
        else:
            self._positional_intensities = [self.array(np.zeros(len(self._con_vocs[self._cltype]))) + 1
                                            for rad in range(-self._m,self._m+1)]
        # these are the combined output vocabulary (indexing the combined output vectors)
        # note: by not using con_voc for c-dist's output vocabulary, fine-tuning will mis-align
        self._vecsegs = {}; self._vecdim = 0; self._allvocs = []
        # this is only for fine tuning, and somehow should/could be witheld to speed calculations
        if self._cltype not in self._lorder:
            self._lorder += [self._cltype]
        for li, ltype in enumerate(self._lorder):
            self._vecsegs[ltype] = (self._vecdim, self._vecdim + len(self._vocs[ltype]))
            self._vecdim += len(self._vocs[ltype])
            self._allvocs += self._vocs[ltype]
        return all_data
    
    def encode(self, t, c):
        intensity = 1; ent = None
        if 'c' in self._positionally_encode:
            intensity *= self.intensity(c[0], abs(c[1]))
        if 't' in self._positionally_encode: 
            intensity *= self.intensity(t[0], abs(c[1]))
        enc = self.if_encode_context(c) if (self._cltype == 'frq' and c[-1] == 'form') else c
        if ('attn' in c[-1]) or (self._positional == 'independent'):
            enc = (enc[0], 0, enc[-1])
        if t: ent = (self._ife[t[0]], 'frq') if t[1] == 'form' else None
        return ent, enc, intensity
    
    def intensity(self, c, dm):
        if type(c) == int:
            theta = c/self._M0
        else:
            theta = self._p0.get(c, 0)
        return (np.cos((2*np.pi*dm*theta) if (c or type(c) == bool) else np.pi) + 1)/2
    
    ## it looks like context-forms can be masked with ife both here in grokdoc() and count() 
    ## to control the encoding for the whole system, with exception of generation
    def if_encode_context(self, c):
        return(tuple([self._ife.get(c[0], 0), c[1], 'frq']))
        
    def compute_marginals(self):    
        numcon = (len(self._ife) if self._cltype == 'form' else len(set(list(self._ife.values()))))*2*(self._m+1)
        self._beta, self._zeta, self._Tn, self._Cn, self._Tvs, self._Cvs = {}, {}, {}, {}, {}, {}
        for ltype in tqdm(self._Fs):
            T = Counter(); C = Counter(); C_tot = 0
            TCs = defaultdict(set); CTs = defaultdict(set)
            ctype = 'attn' if 'attn' in ltype else self._cltype
            for t, c in self._Fs[ltype]:
                T[t] += self._Fs[ltype][(t,c)]; C[c] += self._Fs[ltype][(t,c)]
                C_tot += self._Fs[ltype][(t,c)]; TCs[t].add(c); CTs[c].add(t)
            numtok = len(T); # numcon = len(C); 
            self._Tvs[ltype] = self.array([T.get(t,1.) for t in self._vocs[ltype]])
            if ctype not in self._Cvs: self._Cvs[ctype] = self.array([C.get(c,1.) for c in self._con_vocs[ctype]])
            Cp = {t: sum([C[c] for c in TCs[t]] + [0]) for t in T}
            Tp = {c: sum([T[t] for t in CTs[c]] + [0]) for c in C}
            self._beta[ltype] = defaultdict(lambda : 0, {t: Cp[t]/C_tot for t in T})
            self._zeta[ltype] = defaultdict(lambda : 0, {c: Tp[c]/C_tot for c in C})
            self._Cn[ltype] = defaultdict(lambda : numcon, {t: numcon - len(TCs[t]) for t in T})
            self._Tn[ltype] = defaultdict(lambda : numtok, {c: len(T) - len(CTs[c]) for c in C})
            if ltype == 'form':
                self._f = Counter({t[0]: T[t] for t in T if t[1] == 'form'}); self._f[''] = 1
                self._M = sum(self._f.values())
            
    def build_dense_output(self):
        for ltype in tqdm(self._Fs):
            ctype = 'attn' if 'attn' in ltype else self._cltype
            fl = Counter(); cl = Counter()
            for t, c in self._Fs[ltype]:
                fl[t] += self._Fs[ltype][(t,c)]
                cl[c] += self._Fs[ltype][(t,c)]
            # first build the emission matrices
            X = np.zeros((len(self._vocs[ltype]), len(self._con_vocs[ctype])))
            for i in range(X.shape[0]): ## add the negative information
                if self._Cn[ltype][self._vocs[ltype][i]]:
                    X[i,:] = (1 - self._beta[ltype][self._vocs[ltype][i]]
                              )/self._Cn[ltype][self._vocs[ltype][i]]
            for t, c in self._Fs[ltype]: ## add the positive information
                X[self._idxs[ltype][t], self._con_idxs[ctype][c]] = self._beta[ltype][t]*self._Fs[ltype][(t,c)]/fl[t]
            X[X==0] = self._noise; X /= X.sum(axis = 1)[:,None]; X = np.nan_to_num(-np.log10(X))
            self._Xs[ltype] = self.array(X)
            # now build the transition matrices
            A = np.zeros((len(self._vocs[ltype]), len(self._con_vocs[ctype])))
            for j in range(A.shape[1]): ## add the negative information
                if self._Tn[ltype][self._con_vocs[ctype][j]]:
                    A[:,j] = (1 - self._zeta[ltype][self._con_vocs[ctype][j]])/self._Tn[ltype][self._con_vocs[ctype][j]]
            for t, c in self._Fs[ltype]: ## add the positive information
                A[self._idxs[ltype][t], self._con_idxs[ctype][c]] = self._zeta[ltype][c]*self._Fs[ltype][(t,c)]/cl[c]
            A[A==0] = self._noise; A /= A.sum(axis = 0); A = np.nan_to_num(-np.log10(A))
            self._As[ltype] = self.array(A)
        
    def pre_train(self, ptdocs, update_ife = False, update_bow = False):
        if ptdocs:
            ptdocs = [["".join(s) for s in d] for d in ptdocs]
            print("Processing pre-training documents...")
            self.process_documents(ptdocs, update_ife = update_ife, update_bow = update_bow)
            print("Re-computing marginal statistics...")
            self.compute_marginals()
            print("Re-building dense output heads...")
            self.build_dense_output()
            # report model statistics
            print('Done.')
            print('Model params, types, encoding size, contexts, vec dim, max sent, and % capacity used:', 
                  len(self._Fs[self._cltype]), len(self._ltypes['form']), len(self._ltypes[self._cltype]), 
                  len(self._con_vocs[self._cltype]), self._vecdim, self._max_sent,
                  round(100*len(self._Fs[self._cltype])/(len(self._con_vocs[self._cltype])*len(self._ltypes['form'])), 3))

    def build_trXs(self, all_data): # viterbi decoding will work best with token-level transitions
        print("Counting for transition matrices...")
        self._trFs = defaultdict(Counter)
        for d_i, (doc, cover, ltypes, layers) in tqdm(list(enumerate(all_data))):          
            for ltype, layer in zip(ltypes, layers):
                lstream = [''] + [lt for ls in layer for lt in ls]
                self._trFs[ltype] += Counter(list(zip(lstream[1:], lstream[:-1])))
            d, layers, ltypes, _ = process_document((doc, cover, ltypes, layers))
            
            for ltype, layer in zip(ltypes, layers):
                if ltype not in ['nov', 'iat', 'bot', 'eot', 'eos', 'eod']: continue
                lstream = [''] + [lt for ls in layer for lt in ls]
                self._trFs[ltype] += Counter(list(zip(lstream[1:], lstream[:-1])))
        print("Building transition matrices for Viterbi tag decoding...")
        for ltype in tqdm(self._trFs):
            fl = Counter()
            self._trXs[ltype] = np.zeros((len(self._vocs[ltype]), len(self._vocs[ltype])))
            self._trIs[ltype] = np.zeros(len(self._vocs[ltype]))
            for t, c in self._trFs[ltype]: 
                if c:
                    fl[t] += self._trFs[ltype][(t,c)]
                else:
                    self._trIs[ltype][self._idxs[ltype][(t, ltype)]] += self._trFs[ltype][(t,c)]
            self._trIs[ltype] /= self._trIs[ltype].sum() # record the initial state probabilities
            for t, c in self._trFs[ltype]: ## add the positive information
                if not c: continue
                self._trXs[ltype][self._idxs[ltype][(t, ltype)], 
                                  self._idxs[ltype][(c, ltype)]] = self._beta[ltype][(t, ltype)]*self._trFs[ltype][(t,c)]/fl[t]
            for j in range(self._trXs[ltype].shape[1]): ## add the negative information
                zees = self._trXs[ltype][:,j] == 0.; Cn = sum(zees)
                nonz = self._trXs[ltype][:,j] != 0.; Cp = sum(nonz)
                bet = Cp/(Cn + Cp)
                if Cn: 
                    self._trXs[ltype][zees,j] = (1 - bet)/Cn
                    self._trXs[ltype][nonz,j] *= bet
                self._trXs[ltype][:,j] /= self._trXs[ltype][:,j].sum()
                
    def hot_encode(self, w):
        if len(w) == 2:
            vec = np.array(self._zeds[w[1]]); vec[self._idxs[w[1]][w]] = 1.
        else:
            vec = np.zeros(len(self._con_vocs[self._cltype])); vec[self._con_idxs[self._cltype][w]] = 1. 
        return self.array(vec)
    
    def dense_encode(self, t, c):
        _, enc, intensity = self.encode(t, c)
        return (self.hot_encode(enc)*intensity if enc in self._con_idxs[self._cltype] else 
                self.array(np.zeros(len(self._con_vocs[self._cltype]))))
    
    def hot_context(self, contexts, t = ('', 'form')):
        return sum([self.dense_encode(t, c) for c in contexts])
        
    def hot_conmat(self, contexts, ltype):
        conmat = np.zeros((len(self._vocs[ltype]), len(self._con_vocs[self._cltype])))
        for c in contexts:
            enc = self.encode(('', 'form'), c)[1]
            if enc in self._con_idxs[self._cltype]:
                conmat[:,self._con_idxs[self._cltype][enc]] = self.array([self.encode(t, c)[-1] for t in self._vocs[ltype]])
        return conmat
    
    def dot(self, x, y):
        return x.inner(y) if self._gpu else x.dot(y)
    
    def P1(self, ltype, cv, cm = None):
        if cm is not None:
            Csum = self.dot(cm, self._Cvs[self._cltype]); Csum[Csum == 0] = min(Csum[Csum != 0])
            P = (cm * self._Xs[ltype]).sum(axis = 1)
        else:
            Csum = self.dot(cv, self._Cvs[self._cltype])
            P = self.dot(self._Xs[ltype], cv)
        P -= np.log10(self._Tvs[ltype]/Csum); P = 10**-(P + P.min()); P = P/P.sum()
        return P
            
    def represent(self, stream, predict_contexts = False):
        vecs = []; ltypes = list(self._lorder) if not predict_contexts else [self._cltype]
        if False not in self._attn_type:
            wav_idx, wav_f, wav_M, wav_T = wave_index(stream)
            atns = sum([get_wave(w, wav_idx, self._f, self._M, wav_T, # wav_f, wav_M, 
                                 'accumulating' in self._attn_type, 'backward' in self._attn_type) 
                        for w in wav_f if w])
        else:
            atns = np.ones(len(stream))
        for wi, w in enumerate(stream):
            atn = abs(atns[wi])
            contexts = get_context(wi, stream, m = self._m)
            convec = self.hot_context(contexts, (w, 'form'))
            # collect the necessary vectors for the prediction of this model's tags
            vs = []
            for li, ltype in enumerate(ltypes):
                vs.append(self.P1(ltype, convec))
            vecs.append(self.cat(vs))
        return vecs, atns
            
    def fine_tune(self, docs, covering = [], all_layers = defaultdict(lambda : defaultdict(list))):
        docs = [["".join(s) for s in d] for d in docs]
        self._Ls = {ltype: self.array(np.zeros((len(self._vocs[ltype]), len(self._con_vocs[self._cltype])))) 
                    for ltype in self._ltypes} ############# check to make sure we really want to do this for all _ltypes
        print("Fine-tuning dense output heads...")
        for d_i, doc in tqdm(list(enumerate(docs))):
            doc = (json.dumps([self.tokenize(s) for s in doc]) 
                   if (covering and self._tokenizer_name == 'hr-bpe') or (not covering) else json.dumps(covering[d_i]))
            d, d_layers, d_ltypes, meta = process_document([doc, covering[d_i] if covering else [], 
                                                            list(all_layers[d_i].keys()), list(all_layers[d_i].values())])
            stream = list([t for s in d for t in s])
            lstreams = [list([lt for ls in ld for lt in ls]) for ld in d_layers]
            # get the vectors and attention weights for the document
            vecs, _ = self.represent(stream, predict_contexts = True)
            ## this is where we have to go through the vecs and scoup up to 2*m + 1 vectors for the context distribution
            ## these should be stacked and applied to a contiguous portion of the dense distribution
            for wi in list(range(len(stream))):
                contexts = get_context(wi, stream, m = self._m)
                hot_con = self.hot_context(contexts, (stream[wi], 'form')) 
                
                # the likelihood context vector from all positional predictions
                dcv = self.get_dcv(vecs, wi)
                dcv = ((dcv*sum(hot_con)/sum(dcv)) + hot_con)/2 if sum(dcv) else hot_con
                for ltype in self._Ls:
                    if ltype not in d_ltypes:
                        continue
                    l = lstreams[d_ltypes.index(ltype)][wi] if ltype != 'form' else stream[wi]
                    self._Ls[ltype][self._idxs[ltype][(l, ltype)],:] += dcv
        for ltype in self._Ls:
            self._Ls[ltype][self._Ls[ltype]==0] = self._noise
            self._Ls[ltype] /= self._Ls[ltype].sum(axis=1)[:,None]
            # this just weights a geometric average of probabilities
            self._Ls[ltype] = self.array(np.nan_to_num(-np.log10(self._Ls[ltype])))
        
    def get_dcv(self, vecs, wi):        
        a = self.array(np.zeros(len(self._con_vocs[self._cltype]))) + 1
        dcv = self.array(np.zeros(len(self._con_vocs[self._cltype]))) # note this sets and keeps zero
        ##
        if self._positional == 'independent':
            # gather the (2*m+1)-location positional attention distribution
            pa = self.dot(self._As[self._cltype+'-attn'].T, vecs[wi])
            pa = 10**-(pa - pa.max())
            if pa.sum(): pa = pa/pa.sum()
            # gather the |V|-context semantic distribution
            sa = self.dot(self._As[self._cltype].T, vecs[wi])
            sa = 10**-(sa - sa.max())
            if sa.sum(): sa = sa/sa.sum()
            dcv = sum([vecs[ci]*pa[ci-wi+self._m]*sa*self._positional_intensities[ci-wi+self._m] 
                       for ci in range(max([wi-self._m, 0]), min([wi+self._m+1, len(vecs)]))])
        elif self._positional == 'dependent':
            window = np.array(range(max([wi-self._m, 0]),min([wi+self._m+1, len(vecs)])))
            radii = window - wi
            mindex = self._m + radii[0]
            mincol = mindex*len(self._vocs[self._cltype])
            maxcol = mincol + len(window)*len(self._vocs[self._cltype])
            # concatenate the likelihood context vector from all positional predictions
            dcv[mincol:maxcol] += self.cat([vecs[ci] for ci in range(max([wi-self._m, 0]), min([wi+self._m+1, len(vecs)]))])
            # if 'c' in self._positionally_encode:
            dcv[mincol:maxcol] *= self.array(self._positional_intensities[mincol:maxcol])
            a = self.dot(self._As[self._cltype].T, vecs[wi])
            a = 10**-(a - a.max()); a /= a.sum()
            dcv *= a
        return dcv
    
    def P2(self, ltypes, vecs, wi, cv, cm = None): 
        dcv = self.get_dcv(vecs, wi)
        if cm is not None:
            dcv = ((dcv*cm.sum(axis = 0)/sum(dcv)) + cm)/2 if sum(dcv) else cm
            Csum = self.dot(dcv, self._Cvs[self._cltype]); Csum[Csum == 0] = min(Csum[Csum != 0])
        else:
            dcv = ((dcv*sum(cv)/sum(dcv)) + cv)/2 if sum(dcv) else cv
            Csum = self.dot(dcv, self._Cvs[self._cltype]) 
        Ps = []
        for ltype in ltypes:
            if cm is not None:
                Ps.append((dcv * self._Ls[ltype]).sum(axis = 1))
            else:
                Ps.append(self.dot(self._Ls[ltype], dcv))
            Ps[-1] -= np.log10(self._Tvs[ltype]/Csum); Ps[-1] = 10**-(Ps[-1] - Ps[-1].min()); Ps[-1] /= Ps[-1].sum()
        return self.cat(Ps)
    
    def attend(self, vecs, stream, predict_contexts = False):
        avecs = []; ltypes = list(self._lorder) if not predict_contexts else [self._cltype]
        for wi in list(range(len(vecs))):
            contexts = get_context(wi, stream, m = self._m)
            cv = self.hot_context(contexts, (stream[wi], 'form'))
            avecs.append(self.P2(ltypes, vecs, wi, cv))
        return avecs
    
    def grok(self, wi, w, eot, eos, eod, nov, atn, vec, seed, 
             all_vecs, all_atns, all_nrms, all_ixs, tags = {}):
        self._whatevers.append(Whatever(w, ix = self._ix, sep = eot, nov = nov, 
                                        atn = atn, vec = all_vecs[0][wi])) # vec
        self._w_set.add(w)
        self._ix += len(w)
        # apply token-level information to build the next level sequence
        seps = [eot, eos, eod]
        if eot:
            wis = [wi - wix for wix in range(len(self._whatevers))]
            kwargs = {'sep': seps[1], 'nrm': all_nrms[1][all_ixs[1][wi]], 
                      'atn': all_atns[1][all_ixs[1][wi]], 'vec': all_vecs[1][all_ixs[1][wi]]}
            for ltype in ['lem', 'sen', 'pos', 'ent', 'dep', 'sup', 'infs']:
                kwargs[ltype] = tags.get(ltype, None)
            self._tokens.append(Token(self._whatevers, ix = self._ix - len("".join([w._form for w in self._whatevers])), **kwargs))                
            self._whatevers = []
            # apply sentence-level information to build the next level sequence
            if kwargs['sep']: 
                kwargs = {'sep': seps[2], 'nrm': all_nrms[2][all_ixs[2][wi]],
                          'atn': all_atns[2][all_ixs[2][wi]], 'vec': all_vecs[2][all_ixs[2][wi]]}
                # 'sty' prediction is a sentence-level mult-class classification tag (sentence type)
                for ltype in ['sty']:
                    kwargs[ltype] = tags.get(ltype, None)
                self._sentences.append(Sentence(self._tokens, 
                                                ix = self._ix - len("".join([w._form for t in self._tokens for w in t._whatevers])),
                                                **kwargs))
                self._tokens = []
                # apply document-level information to build the next level sequence
                if kwargs['sep']:
                    kwargs = {'nrm': all_nrms[3][all_ixs[3][wi]], 'atn': all_atns[3][all_ixs[3][wi]], 'vec': all_vecs[3][all_ixs[3][wi]]}
                    self._documents.append(Document(self._sentences, 
                                                    ix = self._ix - len("".join([w._form for s in self._sentences 
                                                                                 for t in s._tokens for w in t._whatevers])),
                                                    **kwargs))
                    self._w_set = set(); self._ix = 0
                    self._sentences = []
    
    def viterbi(self, vecs, ltype):
        # start off the chain of probabilities 
        V = [{}]; segsum = sum(vecs[0][self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
        for t in self._vocs[ltype]:
            V[0][t] = {"P": (self._trIs[ltype][self._idxs[ltype][t]] * # initial state
                             (vecs[0][self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._idxs[ltype][t]]/segsum)), # emission potential
                       "pt": None}
        # continue with chains for each branching point
        for vec in vecs[1:]:
            V.append({}); i = len(V) - 1
            maxP = 0.
            segsum = sum(vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
            for t in self._vocs[ltype]:
                P, pt = max([(V[i - 1][pt]["P"] * # probability of chain to this point
                              (vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._idxs[ltype][t]]/segsum) * # emission potential
                              self._trXs[ltype][self._idxs[ltype][t],self._idxs[ltype][pt]], pt) # transmission potential
                              for pt in self._vocs[ltype]]); V[i][t] = {"P": P, "pt": pt}
                if P > maxP:
                    maxP = P
            for t in V[i]:
                V[i][t]['P'] /= maxP
        # get most probable state and its backtrack
        tags = [max(V[-1], key = lambda t: V[-1][t]['P'])]
        pt = tags[-1]
        # follow the backtrack till the first observation
        for i in range(len(V) - 2, -1, -1):
            tags.insert(0, V[i + 1][pt]["pt"])
            pt = V[i + 1][pt]["pt"]
        return [t[0] for t in tags]
    
    def output(self, vec, ltype):
        segsum = sum(vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
        layer_Ps = Counter({t: vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._idxs[ltype][t]]/segsum
                            for ix, t in enumerate(self._vocs[ltype])})
        layer_that = list(layer_Ps.most_common(1))[0][0]
        return(layer_that, layer_Ps)
    
    def yield_branch(self, sentence, t_i):
        yield t_i
        for i_t_i in sentence[t_i]._infs:
            yield from self.yield_branch(sentence, i_t_i)
    
    def tag_tree(self, tvecs):
        # dep/sup/infs prediction only engages upon sentence completion
        tags = []; sups = ['']*len(tvecs); deps = ['']*len(tvecs); infss = [[] for _ in range(len(tvecs))]
        for ti in range(len(tvecs)):
            tags.append({ltype: self.output(tvecs[ti], ltype) for ltype in ['dep', 'sup']})
        # time to find the root and other parse tags
        root_ix = max(range(len(tags)), key = lambda x: ((tags[x]['dep'][1][('root', 'dep')] 
                                                          if ('root', 'dep') in tags[x]['dep'][1] else 0)*
                                                         (tags[x]['sup'][1][('0', 'sup')] 
                                                          if ('0', 'sup') in tags[x]['sup'][1] else 0))**0.5)
        deps[root_ix] = 'root'; sups[root_ix] = '0'
        nonroot = set([ti for ti in range(len(tvecs)) if ti != root_ix])
        untagged = nonroot; taggable = set([ti for ti in range(len(tvecs))])
        tagged = taggable - untagged
        tag_max_vals = []
        while untagged:
            tagged = taggable - untagged
            tag_vals = []
            for ti in untagged:
                tag = tags[ti]
                if tag['dep'][1] and set([int(x[0]) + ti for x in tag['sup'][1]]).intersection(tagged): # taggable
                    layer_sups, sups_ps = map(np.array, zip(*[x for x in tag['sup'][1].most_common() 
                                                              if int(x[0][0])+ti in tagged and int(x[0][0])])) # taggable
                    sup_that = layer_sups[0][0]
                    layer_deps, deps_ps = map(np.array, zip(*[x for x in tag['dep'][1].most_common() if x[0][0] != 'root']))
                    dep_that = layer_deps[0][0]
                    tag_vals.append([(tag['sup'][1][(sup_that, 'sup')]*tag['dep'][1][(dep_that, 'dep')])**0.5, 
                                     [ti, ti, sup_that, dep_that]])
                else:
                    tag_vals.append([0., [ti, ti, tag['sup'][0], tag['dep'][0]]])
            max_p, max_vals = max(tag_vals)
            tag_max_vals.append((max_p, max_vals))
            deps[max_vals[0]] = max_vals[3]
            sups[max_vals[0]] = max_vals[2]
            infss[max_vals[0]+int(max_vals[2])].append(max_vals[0])
            untagged.remove(max_vals[1])
        new_tag_max_vals = []# ; its = 1
        while (new_tag_max_vals != tag_max_vals): # and its <= maxits:
            if new_tag_max_vals:
                tag_max_vals = list(new_tag_max_vals); new_tag_max_vals = []
            for max_p, max_vals in sorted(tag_max_vals): # , reverse = True
                if not max_p:
                    new_tag_max_vals.append((max_p, max_vals))
                else:
                    dummies = []
                    for infs in infss:
                        dummies.append(Dummy()); dummies[-1]._infs = infs
                    sup_candidates = nonroot - set(self.yield_branch(dummies, max_vals[0]))
                    if sup_candidates:
                        max_candidate = max(sup_candidates, 
                                            key = lambda x: tags[max_vals[0]]['dep'][1][(str(x-max_vals[0]), 'sup')])
                        new_max_p = (tags[max_vals[0]]['sup'][1][(str(max_candidate-max_vals[0]), 'sup')]*
                                     tags[max_vals[0]]['dep'][1][(deps[max_vals[0]], 'dep')])**0.5
                        if new_max_p > max_p:
                            new_max_vals = [max_vals[0], max_vals[1], str(max_candidate-max_vals[0]), deps[max_vals[0]]]
                            new_tag_max_vals.append((new_max_p, new_max_vals))
                            sups[max_vals[0]] = new_max_vals[2]
                            old_sup = int(sups[max_vals[0]])+max_vals[0]
                            infss[max_candidate].append(infss[old_sup].pop(max_vals[0]))
                        else:
                            new_tag_max_vals.append((max_p, max_vals))
                    else:
                        new_tag_max_vals.append((max_p, max_vals))
            # its += 1
        return sups, deps, infss
    
    def decode_eots(self, vecs, decode_method = 'viterbi'):
        # viterbi decode is_token status for whatevers
        if decode_method == 'viterbi':
            iats = self.viterbi(vecs, 'iat')
        else: # argmax decode is_token status for whatevers
            iats = [list(self.output(vec, 'iat')[1].most_common(1))[0][0][0] for vec in vecs]
        # fill in determined details, assuming is_token prediction (iat) is accurate
        eots = []
        for wi in range(len(vecs)):
            eots.append(iats[wi])
            if iats[wi] and wi:
                if not eots[-2]: eots[-2] = True
        # enforce boundary constraints (final whatever has to end a token)
        if not eots[-1]: eots[-1] = True
        # predict eot status for all of the rest
        false_vecs = []
        for wi in range(len(vecs)):
            if not eots[wi]:
                false_vecs.append((wi, vecs[wi]))
            else:
                if false_vecs:
                    if decode_method == 'viterbi': # viterbi decode eot status for whatevers
                        bots = self.viterbi([vec for wii, vec in false_vecs], 'bot')
                        new_eots = self.viterbi([vec for wii, vec in false_vecs], 'eot')
                    else: # argmax decode eot status for whatevers
                        bots = [list(self.output(vec, 'bot')[1].most_common(1))[0][0][0] for wii, vec in false_vecs]
                        new_eots = [list(self.output(vec, 'eot')[1].most_common(1))[0][0][0] for wii, vec in false_vecs]
                    bot_i = 0
                    for bot, eot, wi_vec in zip(bots, new_eots, false_vecs):
                        wii, vec = wi_vec
                        eots[wii] = eot
                        if bot_i:
                            eots[wii-1] = bot
                        bot_i += 1
                    false_vecs = []
        if false_vecs:
            if decode_method == 'viterbi': # viterbi decode eot status for whatevers
                bots = self.viterbi([vec for wii, vec in false_vecs], 'bot')
                new_eots = self.viterbi([vec for wii, vec in false_vecs], 'eot')
            else: # argmax decode eot status for whatevers
                bots = [list(self.output(vec, 'bot')[1].most_common(1))[0][0][0] for wii, vec in false_vecs]
                new_eots = [list(self.output(vec, 'eot')[1].most_common(1))[0][0][0] for wii, vec in false_vecs]
            bot_i = 0
            for bot, eot, wi_vec in zip(bots, new_eots, false_vecs):
                wii, vec = wi_vec
                eots[wii] = eot
                if bot_i:
                    eots[wii-1] = bot
                bot_i += 1
        return eots
    
    def decode_eoss(self, tvecs, twis):
        # decoding the sentence segmentation
        tokens = []; teoss = []
        for ti in range(len(tvecs)):
            tokens.append(ti)
            teoss.append(self.output(tvecs[ti], 'eos')[0][0] if 'eos' in self._ltypes else True)
        if (not teoss[-1]) and (len(tokens) >= self._max_sent) and self._max_sent and predict_tags:
                teoss[max([(self.output(tvecs[tix], 'eos')[1][(True, 'eos')], tix) for tix in tokens])[1]] = True
        for ti, teos in enumerate(teoss):
            eoss.extend([teos]*len(twis[ti]))
        if not eoss[-1]: eoss[-1] = True
        return eoss
    
    def decode_poss(self, tvecs, twis, decode_method = 'viterbi'):
        # viterbi decode pos tags
        if decode_method == 'viterbi':
            tposs = self.viterbi(tvecs, 'pos')
        # argmax decode pos tags
        if decode_method == 'argmax':
            tposs = [list(self.output(vec, 'pos')[1].most_common(1))[0][0][0] for vec in tvecs]
        poss = []
        for ti in range(len(tposs)):
            poss.extend([tposs[ti]]*len(twis[ti]))
        return poss
    
    def decode_parse(self, tvecs, twis, eoss):
        ssts = [0]; seds = []
        tsups, tdeps, tinfss = [], [], []; sups, deps, infss = [], [], []
        for ti, twi in enumerate(twis):
            if eoss[twi[-1]] or twi[-1] + 1 == len(eoss):
                seds.append(ti + 1)
                if ti + 1 < len(twis):
                    ssts.append(ti + 1)
        for sst, sed in zip(ssts, seds):
            ssups, sdeps, sinfss = self.tag_tree(tvecs[sst:sed])
            tsups += ssups; tdeps += sdeps; tinfss += sinfss
        for ti in range(len(tvecs)):
            sups.extend([tsups[ti]]*len(twis[ti]))
            deps.extend([tdeps[ti]]*len(twis[ti]))
            infss.extend([tinfss[ti]]*len(twis[ti]))
        return sups, deps, infss
    
    def decode_stys(self, svecs, stis, twis):
        sstys = []; stys = []
        for svec in svecs:
            sstys.append(list(self.output(svec, 'sty')[1].most_common(1))[0][0][0])
        for si in range(len(svecs)):
            stys.extend([sstys[si]]*sum([len(twis[ti]) for ti in stis[si]]))
        return stys
    
    def interpret(self, docs, covering = [], all_layers = defaultdict(lambda : defaultdict(list)),
                  seed = None, predict_tags = True, predict_contexts = False):
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        if seed is not None:
            np.random.seed(seed)
        for d_i, doc in tqdm(list(enumerate(docs))):
            doc = (json.dumps([self.tokenize(s) for s in doc]) 
                   if (covering and self._tokenizer_name == 'hr-bpe') or (not covering) else json.dumps(covering[d_i]))
            d, d_layers, d_ltypes, meta = process_document([doc, covering[d_i] if covering else [], 
                                                            list(all_layers[d_i].keys()), list(all_layers[d_i].values())])
            self._w_set = set(); self._ix = 0
            stream = list([t for s in d for t in s]) if d else []
            eots = ([eot for seot in d_layers[d_ltypes.index('eot')] 
                     for eot in seot] if d else []) if 'eot' in d_ltypes else []
            eoss = [eos for seos in d_layers[d_ltypes.index('eos')] for eos in seos] if d else []
            eods = [eod for seod in d_layers[d_ltypes.index('eod')] for eod in seod] if d else []
            # if a fine-tuned layer exists, use it to attend the prediction vectors
            if self._Ls:
                vecs, atns = self.represent(stream, predict_contexts = True)
                vecs = self.attend(vecs, stream, predict_contexts = False)
            else:
                # get the vectors and attention weights for the sentence
                vecs, atns = self.represent(stream)
            # determine the token segmentation, if necessary
            if not eots: # eot == True means the whatever is a singleton itself; others are compound
                eots = self.decode_eots(vecs)
            tvecs, tnrms, tatns, twis = agg_vecs(vecs, atns, eots) 
            # determine the sentence segmentation, if necessary
            if not eoss:
                eoss = self.decode_eoss(tvecs, twis)
            teoss = [eos for eot, eos in zip(eots, eoss) if eot]
            svecs, snrms, satns, stis = agg_vecs(tvecs, tatns, teoss, tnrms) 
            # determine the document segmentation, if necessary
            if not eods:
                eods = [False for _ in range(len(stream))]; eods[-1] = True
            seods = [eod for eot, eos, eod in zip(eots, eoss, eods) if (eos and eot)]
            dvecs, dnrms, datns, dsis = agg_vecs(svecs, satns, seods, snrms)
            # determine the part of speech tags, if necessary
            poss = [None for _ in range(len(vecs))]
            if 'pos' in self._lorder and 'pos' not in all_layers:
                poss = self.decode_poss(tvecs, twis)
            # determine the parse tags, if necessary
            sups, deps, infss = [None for _ in range(len(vecs))], [None for _ in range(len(vecs))], [[] for _ in range(len(vecs))]
            if 'sup' in self._lorder and 'sup' not in all_layers[d_i]:
                sups, deps, infss = self.decode_parse(tvecs, twis, eoss)
            # determine the sentence type tags, if necessary
            stys = [None for _ in range(len(vecs))]
            if 'sty' in self._lorder and 'sty' not in all_layers:
                stys = self.decode_stys(svecs, stis, twis)
            # feed the tags and data into interaction the framework 
            wi = 0
            all_ixs = [list(range(len(stream))),
                       [ti for ti, wis in enumerate(twis) for _ in wis],
                       [si for si, tis in enumerate(stis) for ti in tis for _ in twis[ti]],
                       [di for di, sis in enumerate(dsis) for si in sis for ti in stis[si] for _ in twis[ti]]]
            all_vecs = (vecs, tvecs, svecs, dvecs); all_atns = (atns, tatns, satns, datns) 
            all_nrms = ([1 for _ in range(len(vecs))], tnrms, snrms, dnrms)
            for w, eot, eos, eod, pos, sup, dep, infs, sty in zip(stream, eots, eoss, eods, poss, sups, deps, infss, stys):
                tags = {'pos': pos, 'sup': sup, 'dep': dep, 'infs': infs, 'sty': sty}
                self.grok(wi, w, eot, eos, eod, w in self._w_set, atns[wi], vecs[wi], seed, 
                          all_vecs, all_atns, all_nrms, all_ixs, tags = tags)
                wi += 1
    
    def predict_document(self, Td):
        if Td: # Dsums should be a marginalized pre-compute
            Dsums = {d_i: sum(self._Tds[d_i].values()) + len(self._D)*self._noise for d_i in self._Tds} 
            DPs = {d_i: -sum([Td[t]*np.log10((self._Tds[d_i].get(t, 0) + self._noise)/Dsums[d_i]) 
                              for t in Td]) 
                   for d_i in self._Tds}
            MaxPs_list = [DPs[d_i] for d_i in DPs if not np.isinf(DPs[d_i])]
            MaxPs = np.max(MaxPs_list) if MaxPs_list else 0
            if MaxPs:
                DPs = {d_i: (10**-(DPs[d_i] - MaxPs) if not np.isinf(DPs[d_i]) else DPs[d_i]) for d_i in DPs}
                Psum = sum([xyz for xyz in DPs.values() if not np.isinf(xyz)] + [0])
                DPs = Counter({d_i: (DPs[d_i]/Psum if not np.isinf(DPs[d_i]) else 0) for d_i in DPs})
            else:
                DPs = Counter({d_i:1/len(self._Tds) for d_i in self._Tds})
        else:
            DPs = Counter({d_i:1/len(self._Tds) for d_i in self._Tds})
        return DPs
    
    def blend_layer(self, Ps, transfer_p, ltype, cv, Td = Counter()):
        if ltype == 'doc':
            layer_Ps = self.predict_document(Td)
            LAY_Ps = Counter({t: sum([layer_Ps[d_i]*((self._Tds[d_i].get(t[0], 0) + self._noise)/
                                                    (self._T[d_i] + self._noise*len(Ps)))
                                      for d_i in layer_Ps]) for t in Ps})
        else:
            layer_Ps = Counter({self._vocs[ltype][pix]: p for pix, p in enumerate(np.array(self.P1(ltype, cv)))})
            LAY_Ps = Counter({t: sum([layer_Ps[LAY]*((self._Tls[ltype][LAY[0]].get(t[0], 0) + self._noise)/
                                                    (self._T[LAY] + self._noise*len(Ps)))
                                      for LAY in layer_Ps]) for t in Ps})
        return blend_predictions(Ps, LAY_Ps, transfer_p)
    
    def novelty_clamp(self, P, transfer_p, Td): # transfer_p is the novel mass
        # gets the total probability covered by the current document
        TdP_tot = sum([P[t]  for t in P if (t[0] in Td)] + [0])
        # clamps towards words unseen in this document by the transfer probabilty
        P = Counter({t: ((P[t]*(1 - transfer_p)/TdP_tot if TdP_tot else 1/len(self._vocs['form'])) if (t[0] in Td) else # 
                        (P[t]*transfer_p)/(1 - TdP_tot) if (1 - TdP_tot) else 1/len(self._vocs['form'])) for t in P}) # 
        Ptot = sum(P.values())
        return Counter({ti: P[ti]/Ptot for ti in P})
    
    def generate(self, m = 1, prompt = "", docs = [], Td = Counter(), revise = [], top = 1., covering = [], 
                 rhyme = False, slang = 0., focus = 0., prose = 0., style = 0., punct = 0., chunk = 0.,
                 seed = None, verbose = True, return_output = False):
        test_ps = [[]]; d_i, w_i = 0, 0
        if docs:
            if (covering and self._tokenizer_name == 'hr-bpe') or (not covering): print('Tokenizing documents...')
            docs = ([[self.tokenize(s) for s in doc] for doc in tqdm(docs)]
                    if (covering and self._tokenizer_name == 'hr-bpe') or (not covering) else list(covering))
            eots = [build_eots(d, cover) for d, cover in zip(docs, covering)] if covering else []
            docs = [[t for s in doc for t in s] for doc in docs]
            eots = [[et for es in edoc for et in es] for edoc in eots]
            tok_ps = []
            print("Evaluating language model..")
            pbar = tqdm(total=len(docs[d_i]))
        if seed is not None:
            np.random.seed(seed)
        stream = list(self.tokenize(prompt)) if prompt else []
        vecs = self.represent(stream, predict_contexts = True)[0] if stream and self._Ls else []
        output = []; sampled = 0; talking = True; wis = []
        if revise and not docs:
            numchars = np.cumsum([len(w) for w in stream])
            wis = [wi for wi in range(len(stream)) 
                   if ((revise[0] <= numchars[wi] - 1 < revise[1]) or 
                       (revise[0] <= numchars[wi] - len(stream[wi]) < revise[1]))]
        if stream: 
            Td += Counter(stream)
            if verbose: 
                if wis:
                    print("".join(stream[:wis[0]]), end = '')
                else:        
                    print("".join(stream), end = '')
        while talking:
            if revise and not docs:
                wi = wis.pop(0); stream[wi] = ""
                if not wis: talking = False
            else:                
                stream = list(stream) + [""] 
                wi = len(stream) - 1
            # get the current context vector
            contexts = get_context(wi, stream, m = self._m)
            convec = self.hot_context(contexts) 
            conmat = self.hot_conmat(contexts, 'form') if 't' in self._positionally_encode else None
            contexts_conmat = self.hot_conmat(contexts, 'frq') if 't' in self._positionally_encode else None
            # only make informed decisions when some context is available
            if sum(convec):
                # get the base prediction distribution for the surface ('form') vocabulary
                if self._Ls:
                    if revise and not docs:
                        vecs[wi] = self.P1(self._cltype, convec, cm = contexts_conmat)
                    else:
                        vecs.append(self.P1(self._cltype, convec, cm = contexts_conmat))
                    P = Counter(dict(zip(self._vocs['form'], np.array(self.P2(['form'], vecs, wi, convec, cm = conmat)))))
                else:
                    P = Counter(dict(zip(self._vocs['form'], np.array(self.P1('form', convec, cm = conmat))))) 
                # clamps the vocabulary according to in-siteu generative statistics
                if rhyme:
                    # measures the local probability of an novel token, i.e., one that's new to the document        
                    alpha = Counter({self._vocs['nov'][pix]: p for pix, p in enumerate(self.P1('nov', convec))})[(True, 'nov')]
                    # in theory, theta (the average replication rate) should stabilize/temper generation over longer documents
                    if any([(t, 'form') in P for t in Td]):
                        theta = 1 - (np.mean(self._alphas.get(int(sum(Td.values())),
                                                              [min([x for x in map(np.mean, self._alphas.values()) if x])])) 
                                     if Td else 1-len(self._D)/self._total_tokens)
                        pnew = ((1 - theta)*alpha)**(1/2)
                        P = self.novelty_clamp(P, pnew, Td)
                # slang is a noise-sampling rate, and transfers probability between locality and topic
                if slang: # coverage from Td operates a bernoulli-bayes document model, which is then blended as
                    # dynamic model of document-frequency (topical) information, elevating document-specific slang
                    P = self.blend_layer(P, slang, 'doc', convec, Td) 
                # blend various predictable layers, as desired
                if 'lem' in self._ltypes and focus: # lem blending helps with semantic stabilization
                    P = self.blend_layer(P, focus, 'lem', convec) 
                if 'pos' in self._ltypes and prose: # pos blending supports prose-like improvements
                    P = self.blend_layer(P, prose, 'pos', convec) 
                if 'sty' in self._ltypes and style: # sty blending crystalizes sentence types
                    P = self.blend_layer(P, style, 'sty', convec) 
                if 'eot' in self._ltypes and chunk: # eot blending builds end of chunk awareness
                    P = self.blend_layer(P, chunk, 'eot', convec) 
                if 'eos' in self._ltypes and punct: # eos blending builds end of sentence awareness
                    P = self.blend_layer(P, punct, 'eos', convec) 
            else: # otherwise output according to the initialization prior
                if self._Ls:
                    vecs.append(self.array(self._P0[self._cltype]))
                P = Counter(dict(zip(self._vocs['form'], np.array(self._P0['form']))))
            # sample from the resulting distribution over the language model's vocabulary
            ts, ps = map(np.array, zip(*P.most_common()))
            if type(top) == float:
                top = len(ps) if top == 1.0 else len(P) - len(ps[ps[::-1].cumsum() <= top])
                ts, ps = ts[:top], ps[:top]
            elif type(top) == list:
                mask = (top[0] <= ps) & (ps <= top[1])
                if any(mask):
                    ts, ps = ts[mask], ps[mask]
            else:
                assert(type(top) == int)
                ts, ps = ts[:top], ps[:top]
            # sample a type base on the blended probabilities
            what = np.random.choice([t[0] for t in ts], size=None, replace=True, p=ps/ps.sum()) 
            # gather stenciling information
            if docs:
                w = docs[d_i][w_i]
                test_p = P.get((w,'form'), P[('','form')])
                if covering:
                    tok_ps.append(test_p)
                    if eots[d_i][w_i]:
                        test_ps[-1].append([np.exp(np.log(tok_ps).sum()/len(tok_ps)), np.exp(np.log(tok_ps).sum()/len(tok_ps))])
                        if w == ' ':
                            test_ps[-1][-1][1] = None
                        tok_ps = []
                else:
                    test_ps[-1].append([test_p, test_p])
                    if w == ' ':
                        test_ps[-1][-1][1] = None
                w_i += 1
                if w_i == len(docs[d_i]):
                    w_i = 0; d_i += 1
                    if d_i == len(docs):
                        talking = False
            # replace the last/empty element with the sampled (or stenciled) token
            stream[wi] = w if docs else what
            # update the process with the sampled type
            sampled += 1; Td[stream[wi]] += 1
            if return_output: output.append([what, P])
            if verbose:
                if docs:
                    pbar.update(1)
                    if not w_i: 
                        pbar.close()
                        print("document no., pct. complete, 1/<P>, 1/<P> (nsp), M, M (nsp): ", d_i, 100*round((d_i)/len(docs), 2), 
                              round(1/(10**(np.log10([test_p[0] for test_p in test_ps[-1]]).mean())), 2), 
                              round(1/(10**(np.log10([test_p[1] for test_p in test_ps[-1] if test_p[1] is not None]).mean())), 2),
                              len(test_ps[-1]), len([test_p[1] for test_p in test_ps[-1] if test_p[1] is not None])
                             )
                        if talking: pbar = tqdm(total=len(docs[d_i]))
                else:
                    print(what, end = '')
            if docs and not w_i: test_ps.append([])
            if sampled == m and not (docs or revise): talking = False
        if revise and not docs:
            if verbose and (wi+1 < len(stream)): print("".join(stream[wi+1:]), end = '')
        if verbose and not docs: print('\n', end = '')
        if return_output: return output