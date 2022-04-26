import json, re
import numpy as np
from tqdm import tqdm, trange
from abc import ABC
from abc import abstractmethod
from itertools import groupby
from functools import reduce, partial
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from ..utils.fnlp import get_context, wave_index, get_wave
from ..utils.munge import process_document, aggregate_processed, build_layers, build_document, build_eots, count
from ..utils.stat import softmax
from ..utils.hr_bpe.src.bpe import HRBPE
from multiprocessing import Pool
from pyspark import SparkContext

class Whatever:
    def __init__(self, form, ix = None, sep = None, nov = None, nrm = None, atn = None, vec = None): 
        self._form = str(form) 
        self._ix = int(ix) if ix is not None else None # note: all ix are scoped as referential 
        self._sep = bool(sep) if sep is not None else None  #  to the objects' starting character indices
        self._nov = bool(nov) if nov is not None else None
        self._nrm = int(nrm) if nrm is not None else 1
        self._atn = float(atn) if atn is not None else 1.
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
    def __init__(self, whatevers, ix = None, sep = None, nrm = None, atn = None, vec = None, 
                 lem = None, sen = None, pos = None, ent = None, dep = None, sup = None, infs = None):
        super(Token, self).__init__("".join([w._form for w in whatevers]), ix = ix, sep = sep,
                                    nrm = nrm, atn = atn, vec = vec)
        self._whatevers = list(whatevers)
        self._lem = str(lem) if lem is not None else None
        self._sen = str(sen) if sen is not None else None
        self._pos = str(pos) if pos is not None else None
        self._ent = str(ent) if ent is not None else None
        self._dep = str(dep) if dep is not None else None
        self._sup = int(sup) if sup is not None else None
        self._infs = [int(inf) for inf in infs] if infs is not None else []
        
class Sentence(Token):
    def __init__(self, tokens, ix = None, sep = None, nrm = None, atn = None, vec = None, sty = None):
        super(Sentence, self).__init__([w for t in tokens for w in t._whatevers], ix = ix, sep = sep, 
                                       nrm = nrm, atn = atn, vec = vec) 
        self._tokens = list(tokens)
        self._sty = str(sty) if sty is not None else None
    def yield_branch(self, t_i):
        yield t_i
        for i_t_i in self._tokens[t_i]._infs:
            yield from self.yield_branch(i_t_i)
        
class Document(Sentence):
    def __init__(self, sentences, ix = None, nrm = None, atn = None, vec = None):
        super(Document, self).__init__([t for s in sentences for t in s._tokens], ix = ix, 
                                       nrm = nrm, atn = atn, vec = vec)
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

    ########## likely, none of these are necessary
    def encode(self, text):
        return self.to_indices(self.tokenizer.tokenize(text))    

    def decode(self, indices):
        return ''.join(self.to_forms(indices))

    def to_indices(self, forms):
        return [self._form2ind[w] for w in forms]

    def to_forms(self, indices):
        return [self._ind2form[i] for i in indices]
    ##########
                
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

    def init(self, m = 10, noise = 0.001, positional = True, seed=None, attn_type = [False], do_ife = True, runners = 0):
        self._seed = int(seed)
        self._attn_type = list(attn_type)
        self._do_ife = bool(do_ife)
        if self._do_ife:
            self._cltype = 'frq'
        else:
            self._cltype = 'form'
        self._lorder = list()
        self._ltypes = defaultdict(set)
        if self._seed:
            np.random.seed(seed=self._seed)
        self._noise = noise
        self._positional = positional
        self._m = m
        self._ife = Counter()
        self._tru_ife = Counter()
        self._X = Counter(); self._F = Counter()
        self._Xs = {}; self._Fs = defaultdict(Counter)
        self._T, self._C, self._Tds, self._Tls = Counter(), Counter(), defaultdict(Counter), defaultdict(lambda : defaultdict(Counter))
        self._TCs = defaultdict(set); self._CTs = defaultdict(set); 
        self._D = defaultdict(set); self._L = defaultdict(lambda : defaultdict(set))
        self._Cp = Counter(); self._C_tot = 0
        for t, c in tqdm(self._X):
            self._X[(t,c)] = self._X[(t,c)]*self._noise; self._T[t] += self._X[(t,c)]; self._C[c] += self._X[(t,c)]
            self._TCs[t].add(c); self._CTs[c].add(t)
            self._Cp[t] += self._X[(t,c)]; self._C_tot += self._X[(t,c)]
        self._alphas = defaultdict(list); self._total_sentences = 0
        self._total_tokens = 0; self._max_sent = 0
        self._runners = int(runners)
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []

    # coverings and layers are tokenizations and tag sets, ltypes keys which
    def fit(self, docs, docs_name, covering = [], 
            all_layers = defaultdict(lambda : defaultdict(list)), # tune_output_heads = 0,
            method = 'hr-bpe', init_method = 'char', param_method = 'est_theta', 
            reg_model = 'mixing', early_stop = True, num_batches = 100, batch_size = 10_000,
            action_protect = ["\n","[*\(\{\[\)\}\]\.\?\!\,\;][ ]*\w", "\w[ ]*[*\(\{\[\)\}\]\.\?\!\,\;]"]):
        # assure all covering segmentations match their documents
        if not all([len("".join(s_c)) == len(s) for d_c, d in zip(covering, docs) for s_c, s in zip(d_c, d)]):
            covering = []
        else:
            self._covering_vocab = set()
            docs = [["".join(s) for s in d] for d in docs]
        # train tokenizer
        print("Training tokenizer...")
        actions_per_batch = int(batch_size/1)
        self._tokenizer_str = f'{method}_{init_method}_{num_batches}_{batch_size}_{actions_per_batch}_{reg_model}_{param_method}_{self._seed}_{docs_name}'
        self.tokenizer = HRBPE(param_method = param_method, reg_model = reg_model, early_stop = early_stop,
                               covering_vocab = self._covering_vocab)
        self.tokenizer.init([s for d in docs for s in d], seed = self._seed, method = init_method, 
                            covering = [s for d in covering for s in d], action_protect = action_protect)
        self.tokenizer.fit(num_batches, batch_size, actions_per_batch=actions_per_batch, seed=self._seed)
        # tokenize documents and absorb co-occurrences
        self.process_documents(docs, all_layers, covering) # , modify_model = True
        # compute the marginals
        print('Computing marginal statistics...')
        self.compute_marginals()
        # build dense models
        print('Building dense output heads...')
        self.build_dense_output()
        
    ## now re-structured to train on layers by document
    def process_documents(self, docs, all_layers = defaultdict(lambda : defaultdict(list)), 
                          covering = [], update_ife = False):
        print('Tokenizing documents...')
        docs = [json.dumps([self.tokenizer.tokenize(s) for s in doc]) for doc in tqdm(docs)]
        d_is = range(len(docs))
        all_data = list(zip(docs, [covering[d_i] if covering else [] for d_i in d_is], # docs and covering
                            [list(all_layers[d_i].keys()) for d_i in d_is], # ltypes
                            [list(all_layers[d_i].values()) for d_i in d_is])) # layers
        Fs = []; old_ife = Counter(self._ife)
        bc = {'m': int(self._m), 'positional': bool(self._positional), # 'tokenizer': self.tokenizer,
              'old_ife': Counter(self._ife)} 
        if self._runners:
            print('Counting documents and aggregating counts...')
#             SparkContext.setSystemProperty('spark.executor.memory', '2g')
#             SparkContext.setSystemProperty('spark.driver.memory', '32g')
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
        ##
        self._ltypes = defaultdict(set)
        self._ltypes['form'].add(''); self._ltypes['frq'].add(0)
        self._F += Fs
        if not self._ife or update_ife:
            self._ife[''] = 0
            for t, c in Fs:
                self._ife[c[0]] += Fs[(t,c)]
                    
        self._Fs = defaultdict(Counter); self._X = Counter()
        print("Encoding parameters...")
        for t, c in tqdm(self._F):
            enc = self.if_encode_context(c) if self._cltype == 'frq' else c
            self._X[(t,enc)] += self._F[(t,c)]
            self._Fs[t[1]][(t,enc)] += self._F[(t,c)]
            self._ltypes[t[1]].add(t[0])
            if t[1] == 'form':
                ent = (self._ife[t[0]], 'frq')
                self._X[(ent,enc)] += self._F[(t,c)]
                self._Fs['frq'][(ent,enc)] += self._F[(t,c)]
                self._ltypes[ent[1]].add(ent[0])
            
        # sets the vocabularies according to the current accumulated collection of data in a unified way
        # these are the individual layers' vocabularies
        self._zeds, self._idxs, self._vocs = {}, {}, {}
        for ltype in self._ltypes:
            self._vocs[ltype] = [(t, ltype) for t in self._ltypes[ltype]]
            self._idxs[ltype] = {t: idx for idx, t in enumerate(self._vocs[ltype])}
            self._zeds[ltype] = np.zeros(len(self._vocs[ltype]))
        # these are the combined context vocabulary, which spreads the vocab across the entire (2m+1) positional range
        self._con_voc = [(t,rad,ltype) for rad in range(-self._m,self._m+1) for t, ltype in self._vocs[self._cltype]] # con_voc
        self._con_idx = {c: idx for idx, c in enumerate(self._con_voc)}
        # these are the combined output vocabulary (indexing the combined output vectors)
        # note: by not using con_voc for c-dist's output vocabulary, fine-tuning will mis-align
        self._vecsegs = {}; self._vecdim = 0; self._allvocs = []
        if self._cltype not in self._lorder:
            self._lorder +=  [self._cltype]
        for li, ltype in enumerate(self._lorder):
            self._vecsegs[ltype] = (self._vecdim, self._vecdim+len(self._vocs[ltype]))
            self._vecdim += len(self._vocs[ltype])
            self._allvocs += self._vocs[ltype]
    
    ## it looks like context-forms can be masked with ife both here in grokdoc() and count() 
    ## to control the encoding for the whole system, with exception of generation
    def if_encode_context(self, c):
        return(tuple([self._ife.get(c[0], 0), c[1], 'frq']))
        
    def compute_marginals(self):
        numcon = (len(self._ife) if self._cltype == 'form' else len(set(list(self._ife.values()))))*2*(self._m+1)
#         numcon = sum(self._X.values())#*2*(self._m+1) # len(self._C) # len(self._con_voc)
        for t, c in tqdm(self._X):
            self._T[t] += self._X[(t,c)]; self._C[c] += self._X[(t,c)]; self._C_tot += self._X[(t,c)]
            self._TCs[t].add(c); self._CTs[c].add(t)
        self._Cp = {t: sum([self._C[c] for c in self._TCs[t]] + [0]) for t in self._T}
        self._Tp = {c: sum([self._T[t] for t in self._CTs[c]] + [0])/self._C_tot for c in self._C}
        self._beta = {t: self._Cp[t]/self._C_tot for t in self._T}
        self._beta[('', 'form')] = 0; self._beta[(0, 'frq')] = self._beta[('', 'form')]
        self._Tn = {c: len(self._T) - len(self._CTs[c]) for c in self._C}
        self._Cn = {t: numcon - len(self._TCs[t]) for t in self._T} 
        self._Cn[('', 'form')] = numcon; self._Cn[(0, 'frq')] = self._Cn[('', 'form')]
        self._missing_mass = sum([(1 - self._beta[t])/self._Cn[t] for t in self._Cn if self._Cn[t]])
        self._f = Counter({t[0]: self._T[t] for t in self._T if t[1] == 'form'})
        self._f[''] = 1
        self._M = sum(self._f.values())
            
    def build_dense_output(self):
        for ltype in tqdm(self._Fs):
            fl = Counter()
            for t, c in self._Fs[ltype]:
                fl[t] += self._Fs[ltype][(t,c)]   
            X = np.zeros((len(self._vocs[ltype]), len(self._con_voc)))
            for i in range(X.shape[0]): ## add the negative information
                if self._Cn.get(self._vocs[ltype][i], self._Cn[('', 'form')]):
                    X[i,:] = (1 - self._beta.get(self._vocs[ltype][i], self._beta[('', 'form')])
                              )/self._Cn.get(self._vocs[ltype][i], self._Cn[('', 'form')]) 
            for t, c in self._Fs[ltype]: ## add the positive information
                X[self._idxs[ltype][t], self._con_idx[c]] = self._beta[t]*self._Fs[ltype][(t,c)]/fl[t]
            self._Xs[ltype] = X
            self._Xs[ltype][self._Xs[ltype]==0] = self._noise
            self._Xs[ltype] /= self._Xs[ltype].sum(axis=1)[:,None]
            self._Xs[ltype] = np.nan_to_num(-np.log10(self._Xs[ltype]))
            
        # report model statistics
        print('Done.')
        print('Model params, types, encoding size, contexts, vec dim, max sent, and % capacity used:', 
              len(self._F), len(self._ltypes['form']), len(self._ltypes[self._cltype]), len(self._con_voc), self._vecdim, self._max_sent,
              round(100*len(self._F)/(len(self._con_voc)*len(self._ltypes['form'])), 3))
        
    def pre_train(self, ptdocs, update_ife = False):
        if ptdocs:
            ptdocs = [["".join(s) for s in d] for d in ptdocs]
            print("Processing pre-training documents...")
            self.process_documents(ptdocs, update_ife = update_ife)
            print("Re-computing marginal statistics...")
            self.compute_marginals()
            print("Re-building dense output heads...")
            self.build_dense_output()
            
    def fine_tune(self, docs, covering = defaultdict(list), all_layers = defaultdict(lambda : defaultdict(list))): 
        docs = [["".join(s) for s in d] for d in docs]
        Xs = {ltype: np.zeros((len(self._vocs[ltype]), len(self._con_voc))) for ltype in self._lorder if ltype != 'frq' and ltype != 'form'}
        print("Fine-tuning dense output heads...")
        for d_i, doc in tqdm(list(enumerate(docs))):
            self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
            self.grokdoc(doc, d_i, seed = self._seed, covering = covering, all_layers = all_layers, 
                         digest = False, predict_tags = False, predict_contexts = True)
            d = self._documents[0]
            vecs, atns, strm, s_is, w_is = list(map(list, zip(*[(w._vec[self._vecsegs[self._cltype][0]:self._vecsegs[self._cltype][1]], 
                                                                 w._atn, w._form, s_i, w_i) for s_i, s in enumerate(d._sentences)
                                                                for w_i, w in enumerate([w for t in s._tokens for w in t._whatevers]) ])))
            vecs = np.array(vecs)
            ## this is where we have to go through the vecs and scoup up to 2*m + 1 vectors for the context distribution
            ## these should be stacked and applied to a contiguous portion of the dense distribution
            for wi in list(range(len(strm))):
                window = np.array(range(max([wi-self._m, 0]),min([wi+self._m+1, len(strm)])))
                radii = window - wi
                mindex = self._m + radii[0]
                mincol = mindex*len(self._vocs[self._cltype])
                maxcol = mincol + len(window)*len(self._vocs[self._cltype])
                convec = np.concatenate(vecs[window])
                for ltype in Xs:
                    if ltype not in self._d_ltypes:
                        continue
                    l = self._d_layers[self._d_ltypes.index(ltype)][s_is[wi]][w_is[wi]] if ltype != 'form' else doc[s_is[wi]][w_is[wi]]
                    Xs[ltype][self._idxs[ltype][(l, ltype)],mincol:maxcol] += convec
        for ltype in Xs:
            Xs[ltype] = Xs[ltype]
            Xs[ltype][Xs[ltype]==0] = self._noise
            Xs[ltype] /= Xs[ltype].sum(axis=1)[:,None]
#             self._Xs[ltype] = np.nan_to_num(-np.log10(Xs[ltype]))
            # this just weights a geometric average of probabilities
            Xs[ltype] = np.nan_to_num(-np.log10(Xs[ltype]))
#             mod_nrm, tun_nrm = (self._Xs[ltype]**2).sum()**0.5, (Xs[ltype]**2).sum()**0.5
#             Xs[ltype] = 10**(-(self._Xs[ltype]*mod_nrm + Xs[ltype]*tun_nrm)/(tun_nrm + mod_nrm))
            Xs[ltype] = 10**(-(self._Xs[ltype] + Xs[ltype])/2)
            Xs[ltype] /= Xs[ltype].sum(axis=1)[:,None]
            self._Xs[ltype] = np.nan_to_num(-np.log10(Xs[ltype]))
            
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
    
    def grok(self, wi, w, eot, eos, eod, nov, atn, vec, seed, predict_tags = True):
        self._whatevers.append(Whatever(w, ix = self._ix, sep = eot, nov = nov, 
                                        atn = atn, vec = vec))
        self._w_set.add(w)
        self._ix += len(w)
        # apply token-level information to build the next level sequence
        if eod: 
            eos = True; eot = True
        elif eos: 
            eot = True
        seps = [eot, eos, eod]
        if eot:
            wis = [wi - wix for wix in range(len(self._whatevers))]
            kwargs = {'sep': seps[1]}
            kwargs['nrm'] = sum([w._nrm for w in self._whatevers])
            kwargs['atn'] = sum([w._atn for w in self._whatevers])
            kwargs['vec'] = ( ((kwargs['nrm']/kwargs['atn'])*np.array([w._vec for w in self._whatevers]).T.dot([w._atn for w in self._whatevers]) )
                             if vec is not None else None)
            if predict_tags:
                for ltype in ['lem', 'sen', 'pos', 'ent']:
                    kwargs[ltype] = (([t for sl in self._d_layers[self._d_ltypes.index(ltype)] for t in sl][wi] 
                                      if ltype in self._d_ltypes else list(self.output(kwargs['vec'], ltype)[1].most_common(1))[0][0][0]) 
                                     if ltype in self._ltypes else None)            
            self._tokens.append(Token(self._whatevers, ix = self._ix - len("".join([w._form for w in self._whatevers])), **kwargs))                
            self._whatevers = []
            
            if kwargs['sep']: ##### 'sty' prediction, a.k.a. _sty is a sentence-level mult-class classification tag (sentence type)
                kwargs = {'sep': seps[2]}
                kwargs['nrm'] = sum([t._nrm for t in self._tokens])
                kwargs['atn'] = sum([t._atn for t in self._tokens])
                kwargs['vec'] = ( ((kwargs['nrm']/kwargs['atn'])*np.array([t._vec for t in self._tokens]).T.dot([t._atn for t in self._tokens]) )
                                 if vec is not None else None)
                if predict_tags:
                    for ltype in ['sty']:
                        kwargs[ltype] = (([t for sl in self._d_layers[self._d_ltypes.index(ltype)] for t in sl][wi] 
                                          if ltype in self._d_ltypes else list(self.output(kwargs['vec'], ltype)[1].most_common(1))[0][0][0]) 
                                         if ltype in self._ltypes else None)
                #### dep/sup/infs prediction only engages upon sentence completion
                ##
                if predict_tags and ('dep' in self._ltypes and 'sup' in self._ltypes):
                    tags = []                    
                    all_twis = []
                    twi = wi - len([what._form for tok in self._tokens for what in tok._whatevers])
                    for tok_i, tok in enumerate(self._tokens):
                        twi += len(tok._whatevers)
                        all_twis.append(twi) # each radius will need be measured in a normed capacity, xchar away
                        tags.append({ltype: (([(t, Counter()) for sl in self._d_layers[self._d_ltypes.index(ltype)] for t in sl][wi] 
                                              if ltype in self._d_ltypes else self.output(tok._vec, ltype)) 
                                             if ltype in self._ltypes else None) for ltype in ['dep', 'sup']})
                    ## time to find the root and othe parse tags
                    root_ix = max(range(len(tags)), key = lambda x: ((tags[x]['dep'][1][('root', 'dep')] if ('root', 'dep') in tags[x]['dep'][1] else 0)*
                                                                     (tags[x]['sup'][1][('0', 'sup')] if ('0', 'sup') in tags[x]['sup'][1] else 0))**0.5)
                    self._tokens[root_ix]._dep = 'root'
                    self._tokens[root_ix]._sup = '0'            
                    ####
                    all_twis = range(len(tags))
                    ####
                    nonroot = set([twi for twi in all_twis if twi != all_twis[root_ix]])
                    untagged = nonroot; taggable = set([twi for twi in all_twis])
                    tagged = taggable - untagged
                    tag_max_vals = []
                    while untagged:
                        tagged = taggable - untagged
                        tag_vals = []
                        for twi in untagged:
                            tag_ix = all_twis.index(twi)
                            tag = tags[tag_ix]
                            if tag['dep'][1] and set([int(x[0]) + twi for x in tag['sup'][1]]).intersection(tagged): # taggable
                                layer_sups, sups_ps = map(np.array, zip(*[x for x in tag['sup'][1].most_common() if int(x[0][0])+twi in tagged and int(x[0][0])])) # taggable
                                sup_that = layer_sups[0][0]
                                layer_deps, deps_ps = map(np.array, zip(*[x for x in tag['dep'][1].most_common() if x[0][0] != 'root']))
                                dep_that = layer_deps[0][0]
                                tag_vals.append([(tag['sup'][1][(sup_that, 'sup')]*tag['dep'][1][(dep_that, 'dep')])**0.5, 
                                                 [tag_ix, twi, sup_that, dep_that]])                                
                            else:
                                tag_vals.append([0., [tag_ix, twi, tag['sup'][0], tag['dep'][0]]])
                        max_p, max_vals = max(tag_vals)
                        tag_max_vals.append((max_p, max_vals))
                        self._tokens[max_vals[0]]._dep = max_vals[3]
                        self._tokens[max_vals[0]]._sup = max_vals[2]
                        self._tokens[max_vals[0]+int(max_vals[2])]._infs.append(max_vals[0])
                        untagged.remove(all_twis[max_vals[1]])
                    new_tag_max_vals = []; its = 1
                    while (new_tag_max_vals != tag_max_vals): # and its <= maxits:
                        if new_tag_max_vals:
                            tag_max_vals = list(new_tag_max_vals); new_tag_max_vals = []
                        for max_p, max_vals in sorted(tag_max_vals): # , reverse = True
                            if not max_p:
                                new_tag_max_vals.append((max_p, max_vals))
                            else:
                                sup_candidates = nonroot - set(self.yield_branch(self._tokens, max_vals[0]))
                                if sup_candidates:
                                    max_candidate = max(sup_candidates, key = lambda x: tags[max_vals[0]]['dep'][1][(str(x-max_vals[0]), 'sup')])
                                    new_max_p = (tags[max_vals[0]]['sup'][1][(str(max_candidate-max_vals[0]), 'sup')]*
                                                 tags[max_vals[0]]['dep'][1][(self._tokens[max_vals[0]]._dep, 'dep')])**0.5
                                    if new_max_p > max_p:
                                        new_max_vals = [max_vals[0], max_vals[1], str(max_candidate-max_vals[0]), self._tokens[max_vals[0]]._dep]
                                        new_tag_max_vals.append((new_max_p, new_max_vals))
                                        self._tokens[max_vals[0]]._sup = new_max_vals[2]
                                        old_sup = int(self._tokens[max_vals[0]]._sup)+max_vals[0]
                                        self._tokens[max_candidate]._infs.append(self._tokens[old_sup]._infs.pop(max_vals[0]))
                                    else:
                                        new_tag_max_vals.append((max_p, max_vals))
                                else:
                                    new_tag_max_vals.append((max_p, max_vals))   
                        its += 1
                ##
                #### can access the in-sentence whatever-stream for allowable arc predictions from range(len(wis))
                self._sentences.append(Sentence(self._tokens, 
                                                ix = self._ix - len("".join([w._form for t in self._tokens for w in t._whatevers])),
                                                **kwargs))
                self._tokens = []
                if kwargs['sep']:
                    kwargs = {'nrm': sum([s._nrm for s in self._sentences]),
                              'atn': sum([s._atn for s in self._sentences])}
                    kwargs['vec'] = ( ((kwargs['nrm']/kwargs['atn'])*np.array([s._vec 
                                                                               for s in self._sentences]).T.dot([s._atn for s in self._sentences]) ) 
                                     if vec is not None else None)
                    self._documents.append(Document(self._sentences, 
                                                    ix = self._ix - len("".join([w._form for s in self._sentences 
                                                                                 for t in s._tokens for w in t._whatevers])),
                                                    **kwargs))
                    self._w_set = set(); self._ix = 0
                    self._sentences = []
        
    def yield_branch(self, sentence, t_i):
        yield t_i
        for i_t_i in sentence[t_i]._infs:
            yield from self.yield_branch(sentence, i_t_i)
    
    def interpret(self, docs, covering = [], all_layers = defaultdict(lambda : defaultdict(list)),
                  seed = None, digest = True, predict_tags = True, dense_predict = False, predict_contexts = False):
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        if seed is not None:
            np.random.seed(seed)
        for d_i, doc in tqdm(list(enumerate(docs))):
            self.grokdoc(doc, d_i, seed = seed, covering = covering, all_layers = all_layers, 
                         digest = digest, predict_tags = predict_tags, 
                         dense_predict = dense_predict, predict_contexts = predict_contexts)
            
    def get_vecs(self, stream):
        vecs = []
        for wi, w in enumerate(stream):
            contexts = get_context(wi, stream, m = self._m, positional = self._positional) # set()
            if self._do_ife:
                contexts = [self.if_encode_context(c) for c in contexts] # set()
            vecs.append(self.predict_layer(stream, self._cltype, i = wi, contexts = contexts))
        return np.array(vecs)
            
    def con_vec(self, vecs, wi):
        con_vec = []
        window = np.array(range(max([wi-self._m, 0]),min([wi+self._m+1, vecs.shape[0]])))
        radii = window - wi
        mindex = self._m + radii[0]
        mincol = mindex*len(self._vocs[self._cltype])
        maxcol = mincol + len(window)*len(self._vocs[self._cltype])
        left_padding = np.zeros(mincol)
        con_vec.append(left_padding)
        con_vec.append(np.concatenate(vecs[window]))
        right_padding = np.zeros(len(self._con_voc) - maxcol)
        con_vec.append(right_padding)
        return np.concatenate(con_vec)
            
    def grokdoc(self, doc, d_i, seed = None, covering = [], all_layers = defaultdict(lambda : defaultdict(list)), 
                digest = True, predict_tags = True, dense_predict = False, predict_contexts = False):
        # print('Tokenizing document...')
        doc = json.dumps([self.tokenizer.tokenize(s) for s in doc])
        d, self._d_layers, self._d_ltypes, meta = process_document([doc, covering[d_i] if covering else [], list(all_layers[d_i].keys()), 
                                                                        list(all_layers[d_i].values())])
        self._s_i = 0; self._w_set = set(); self._ix = 0
        self._s = list([t for s in d for t in s]) if d else []
        self._s_is = list([s_i for s_i, s in enumerate(d) for t in s]) if d else []
        eots = ([eot for seot in self._d_layers[self._d_ltypes.index('eot')] 
                 for eot in seot] if d else []) if 'eot' in self._d_ltypes else []
        eoss = [eos for seos in self._d_layers[self._d_ltypes.index('eos')] for eos in seos] if d else []
        eods = [eod for seod in self._d_layers[self._d_ltypes.index('eod')] for eod in seod] if d else []
        if self._s:
            if dense_predict:
                vecs = self.get_vecs(self._s)
            if False not in self._attn_type:
                wav_idx, wav_f, wav_M, wav_T = wave_index(self._s)
                self._attn = sum([get_wave(w, wav_idx, self._f, self._M, wav_T, # wav_f, wav_M, 
                                           'accumulating' in self._attn_type, 'backward' in self._attn_type) 
                                  for w in wav_f if w])
            else:
                self._attn = np.ones(len(self._s))
            for wi, w, s_i in list(zip(range(len(self._s)), self._s, self._s_is)):
                self._s_i = s_i
                
                atn = abs(self._attn[wi])
                if dense_predict:
                    contexts = set()
                    convec = self.con_vec(vecs, wi)
                else:
                    convec = np.array([])
                    contexts = get_context(wi, self._s, m = self._m, positional = self._positional) # set()
                    ## mask the contexts with ife
                    if self._do_ife:
                        contexts = [self.if_encode_context(c) for c in contexts] # set()
                # collect the necessary vectors for the prediction of this model's tags
                vs = []
                for li, ltype in enumerate(self._lorder):
                    if ltype == self._cltype and not predict_contexts:
                        vs.append(self._zeds[ltype])
                    else:
                        v = self.predict_layer(self._s, ltype, i = wi, contexts = contexts, v = convec)
                        vs.append(v)
                vec = np.concatenate(vs)
                eot = (eots[wi] if eots else (self.output(vec, 'eot')[0] if 'eot' in self._ltypes else True))
                eos = (eoss[wi] if eoss else (self.output(vec, 'eos')[0] if 'eos' in self._ltypes else True))
                eod = (eods[wi] if eods else (self.output(vec, 'eod')[0] if 'eod' in self._ltypes else True))
                
                ## if over self._max_sent, force a max-likelihood prediction over the current backlog and continue with the remains if necessary
                ## i.e., if the current token is the most likely sentence ender, just modify kwargs['sep']. Otherwise, recur.
                if (eot and not eos) and (len(self._tokens) >= self._max_sent) and self._max_sent and predict_tags:
                    sep_ps = []
                    for t in self._tokens:
                        sep_hat, sep_Ps = self.output(t._vec, 'eos')
                        sep_ps.append(sep_Ps[(True, 'eos')])
                    max_sep_ix = max(zip(sep_ps, range(len(sep_ps))))[1]
                    ## first rewind
                    temp_whatever = self._tokens[max_sep_ix]._whatevers[-1]; temp_ix = self._ix; self._ix = temp_whatever._ix
                    temp_whatevers = self._whatevers; self._whatevers = list(self._tokens[max_sep_ix]._whatevers[:-1])
                    ## make sure to ditch the max_sep_ix token, as rewind goes back to before it was formed
                    if max_sep_ix == len(self._tokens) - 1:
                        temp_tokens = []; self._tokens = []
                    else:
                        temp_tokens = list(self._tokens[max_sep_ix+1:]); self._tokens = self._tokens[:max_sep_ix]
                    
                    ## find the right control settings
                    sep_w = temp_whatever._form
                    sep_eot = True; sep_eos = True; sep_eod = False
                    sep_nov = temp_whatever._nov# ; sep_oov = temp_whatever._oov
                    sep_atn = temp_whatever._atn; sep_vec = temp_whatever._vec
                    sep_wi = (len([w._form for d in self._documents for s in d._sentences for t in s._tokens for w in t._whatevers]) +
                              len([w._form for s in self._sentences for t in s._tokens for w in t._whatevers]) +
                              len([w._form for t in self._tokens for w in t._whatevers]) + 
                              len([w._form for w in self._whatevers]))
                    ## grok the stream up through this 'best' sentence separator
                    self.grok(sep_wi, sep_w, sep_eot, sep_eos, sep_eod, sep_nov, 
                              sep_atn, sep_vec, seed, predict_tags = predict_tags)
                    ## now wind back forward to the current whatever before the regular old grokking
                    self._ix = temp_ix
                    self._tokens = temp_tokens
                    self._whatevers = temp_whatevers
                self.grok(wi, w, eot, eos, eod, w in self._w_set,
                          atn, vec, seed, predict_tags = predict_tags)
            if digest:
                self.digest()
        else:
            self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
            
    def predict_layer(self, s, ltype, i = 0, contexts = set(), v = np.array([])):
        NLLs = self.batch_NLLs([(t, ltype) for t in self._ltypes[ltype]], s, i, contexts = contexts, v = v)
        Ps = 10**-(NLLs + min(NLLs))
        return Ps/Ps.sum()
    
    def batch_NLLs(self, ts, s, i, contexts = set(), v = np.array([])): 
        # if no vector is provided, this is a sparse prediction, and a hot vector is built
        if not sum(v.shape):
            if not contexts:
                contexts = get_context(i, s, m = self._m, positional = self._positional)
            Csum = sum([self._C.get(c,0.) for c in contexts])
            if Csum:
                v = sum([self.hot_encode(c) for c in contexts if c in self._C])
            else:
                v = np.zeros(len(self._con_voc))
                Csum = self._missing_mass
        else:
            Csum = v.dot([self._C.get(c,0.) for c in self._con_voc])
        # when a vector is provided, it is just passed through
        ltype = ts[0][1]
        NLLs = self._Xs[ltype].dot(v)
        return np.array([NLLs[self._idxs[ltype][t]] - np.log10(self._T.get(t, 1)/Csum) for t in ts])
    
    def hot_encode(self, w):
        if len(w) == 2:
            vec = np.array(self._zeds[w[1]]); vec[self._idxs[w[1]][w]] = 1.
        else:
            vec = np.zeros(len(self._con_voc)); vec[self._con_idx.get(w, -1)] = 1. ## nulls are now included in contexts # +1
        return vec
    
    def output(self, vec, ltype):
        segsum = sum(vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
        layer_Ps = Counter({t: vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._idxs[ltype][t]]/segsum
                            for ix, t in enumerate(self._vocs[ltype])})
        layer_that = list(layer_Ps.most_common(1))[0][0]
        return(layer_that, layer_Ps)

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
    
    def predict_document(self, Td):
        if Td:
            Dsums = {d_i: sum(self._Tds[d_i].values()) + len(self._D)*self._noise for d_i in self._Tds} # this should be a marginalized pre-compute
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
    
    def blend_layer(self, Ps, transfer_p, s, ltype, i, contexts, convec, Td = Counter()): # , noise
        if ltype == 'doc':
            layer_Ps = self.predict_document(Td)
            LAY_Ps = Counter({t: sum([layer_Ps[d_i]*((self._Tds[d_i].get(t[0], 0) + self._noise)/
                                                    (self._T[d_i] + self._noise*len(Ps)))
                                      for d_i in layer_Ps]) 
                              for t in Ps})
        else:
            layer_Ps = Counter({self._vocs[ltype][pix]: p 
                                for pix, p in enumerate(self.predict_layer(s, ltype, i = i, 
                                                                           contexts = contexts, v = convec))})
            LAY_Ps = Counter({t: sum([layer_Ps[LAY]*((self._Tls[ltype][LAY[0]].get(t[0], 0) + self._noise)/
                                                    (self._T[LAY] + self._noise*len(Ps)))
                                      for LAY in layer_Ps]) 
                              for t in Ps})
        return self.blend_predictions(Ps, LAY_Ps, transfer_p)
    
    def blend_predictions(self, P1, P2, transfer_p):
        blended = {t: P1[t]*(1 - transfer_p) + P2[t]*transfer_p for t in P1}
        blended_sum = sum(blended.values())
        return Counter({t: blended[t]/blended_sum for t in blended})
    
    def novelty_clamp(self, P, transfer_p, Td): # transfer_p is the novel mass
        # gets the total probability covered by the current document
        TdP_tot = sum([P[t]  for t in P if (t[0] in Td)] + [0])
        # clamps towards words unseen in this document by the transfer probabilty
        P = Counter({t: ((P[t]*(1 - transfer_p)/TdP_tot if TdP_tot else 1/len(self._vocs['form'])) if (t[0] in Td) else # 
                        (P[t]*transfer_p)/(1 - TdP_tot) if (1 - TdP_tot) else 1/len(self._vocs['form'])) for t in P}) # 
        Ptot = sum(P.values())
        return Counter({ti: P[ti]/Ptot for ti in P})
    
    def generate(self, m = 1, prompt = "", docs = [], Td = Counter(), revise = [],
                 rhyme = 0., slang = 0., focus = 0., prose = 0., style = 0., punct = 0., chunk = 0.,
                 seed = None, top = 1., covering = [],  all_layers = defaultdict(lambda : defaultdict(list)),
                 digest = True, predict_tags = True, dense_predict = False, predict_contexts = False,
                 verbose = True, return_output = False):
        test_ps = [[]]; d_i, w_i = 0, 0
        if docs:
            print("Tokenizing documents..")
            docs = [[t for s in doc for t in self.tokenizer.tokenize(s)] for doc in tqdm(docs)]
            print("Evaluating language model..")
            pbar = tqdm(total=len(docs[d_i]))
        if seed is not None:
            np.random.seed(seed)
        self.grokdoc([prompt] if prompt else [], -1,
                     seed = seed, covering = covering, all_layers = all_layers, 
                     digest = digest, predict_tags = predict_tags, 
                     dense_predict = dense_predict, predict_contexts = predict_contexts)
        output = []; sampled = 0; talking = True; wis = []
        if revise and not docs:
            numchars = np.cumsum([len(w) for w in self._s])
            wis = [wi for wi in range(len(self._s)) 
                   if ((revise[0] <= numchars[wi] - 1 < revise[1]) or 
                       (revise[0] <= numchars[wi] - len(self._s[wi]) < revise[1]))]
        if self._s: 
            Td += Counter(self._s)
            if verbose: 
                if wis:
                    print("".join(self._s[:wis[0]]), end = '')
                else:        
                    print("".join(self._s), end = '')
        while talking:
            if revise and not docs:
                wi = wis.pop(0); self._s[wi] = ""
                if not wis: talking = False
            else:                
                self._s = list(self._s) + [""] 
                wi = len(self._s) - 1
            if dense_predict:
                vecs = self.get_vecs(self._s)
            if False not in self._attn_type:
                wav_idx, wav_f, wav_M, wav_T = wave_index(self._s)
                self._attn = sum([get_wave(w, wav_idx, self._f, self._M, wav_T, # wav_f, wav_M, 
                                           'accumulating' in self._attn_type, 'backward' in self._attn_type) 
                                  for w in wav_f if w])
            else:
                self._attn = np.ones(len(self._s))
            atn = abs(self._attn[wi])
            if dense_predict:
                contexts = set()
                convec = self.con_vec(vecs, wi)
            else:
                convec = np.array([])
                contexts = get_context(wi, self._s, m = self._m, positional = self._positional) # set()
                ## mask the contexts with ife
                if self._do_ife:
                    contexts = [self.if_encode_context(c) for c in contexts] # set()
            # get the base prediction distribution for the surface ('form') vocabulary
            P = Counter({self._vocs['form'][pix]: p 
                        for pix, p in enumerate(self.predict_layer(self._s, 'form', 
                                                                   i = wi, contexts = contexts, v = convec))})
            # clamps the vocabulary according to in-siteu generative statistics
            if rhyme:
                # measures the local probability of an novel token, i.e., one that's new to the document        
                alpha = Counter({self._vocs['nov'][pix]: p 
                                 for pix, p in enumerate(self.predict_layer(self._s, 'nov', i = wi, contexts = contexts, 
                                                                            v = convec))})[(True, 'nov')] # p new to doc # 
                # in theory, theta (the average replication rate) should stabilize/temper generation over longer documents
                if any([(t, 'form') in P for t in Td]):
                    theta = 1 - (np.mean(self._alphas.get(int(sum(Td.values())),
                                                          [min([x for x in map(np.mean, self._alphas.values()) if x])])) 
                                 if Td else 1-len(self._D)/self._total_tokens)
                    pnew = ((1 - theta)*alpha*(1 - rhyme))**(1/3)
                    P = self.novelty_clamp(P, pnew, Td)
            # slang is a noise-sampling rate, and transfers probability between locality and topic
            if slang:
                # coverage from Td operates a bernoulli-bayes document model, which is then blended as
                # dynamic model of document-frequency (topical) information, elevating document-specific slang
                P = self.blend_layer(P, slang, self._s, 'doc', wi, contexts, convec, Td)
            # blend various predictable layers, as desired
            if 'lem' in self._ltypes and focus: # lem blending helps with semantic stabilization
                P = self.blend_layer(P, focus, self._s, 'lem', wi, contexts, convec) 
            if 'pos' in self._ltypes and prose: # pos blending supports prose-like improvements
                P = self.blend_layer(P, prose, self._s, 'pos', wi, contexts, convec) 
            if 'sty' in self._ltypes and style: # sty blending crystalizes sentence types
                P = self.blend_layer(P, style, self._s, 'sty', wi, contexts, convec) 
            if 'eot' in self._ltypes and chunk: # eot blending builds end of chunk awareness
                P = self.blend_layer(P, chunk, self._s, 'eot', wi, contexts, convec) 
            if 'eos' in self._ltypes and punct: # eos blending builds end of sentence awareness
                P = self.blend_layer(P, punct, self._s, 'eos', wi, contexts, convec) 
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
                w = docs[d_i][w_i]; w_i += 1
                test_ps[-1].append(P.get((w,'form'), P[('','form')]))
                if w_i == len(docs[d_i]):
                    w_i = 0; d_i += 1
                    if d_i == len(docs):
                        talking = False
            # replace the last/empty element with the sampled (or stenciled) token
            self._s[wi] = w if docs else what
            # update the process with the sampled type
            sampled += 1; Td[self._s[wi]] += 1
            if return_output: output.append([what, P])
            if verbose:
                if docs:
                    pbar.update(1)
                    if not w_i: 
                        pbar.close()
                        print("document no., pct. complete, 1/<P>, M: ", d_i, 100*round((d_i)/len(docs), 2), 
                              round(1/(10**(np.log10([test_p for test_p in test_ps[-1]]).mean())), 2), len(test_ps[-1]))
                        if talking: pbar = tqdm(total=len(docs[d_i]))
                else:
                    print(what, end = '')
            if docs and not w_i: test_ps.append([])
            if sampled == m and not (docs or revise): talking = False
        if revise and not docs:
            if verbose and (wi+1 < len(self._s)): print("".join(self._s[wi+1:]), end = '')
        if verbose and not docs: print('\n', end = '')
        if return_output: return output
############################