import json, re, torch
import numpy as np
from math import ceil
from tqdm import tqdm, trange
from abc import ABC, abstractmethod
from itertools import groupby
from functools import reduce, partial
from collections import Counter, defaultdict
from .types import Dummy, Whatever, Token, Sentence, Document, Vocab, Cipher
from ..utils.fnlp import get_context, wave_index, get_wave
from ..utils.munge import process_document, aggregate_processed, build_eots, count, detach, unroll, to_gpu
from ..utils.stat import agg_vecs, blend_predictions
from ..utils.hr_bpe.src.bpe import HRBPE
from ..utils.hr_bpe.src.utils import tokenize, sentokenize
from pyspark import SparkContext
import pyspark

conf = pyspark.SparkConf()
conf.set('spark.local.dir', '/local-data/tmp/')

# purpose: manages and executes all aspects of being of class language model (LM)
# arguments: see __init__()
# prereqs: hr_bpe, numpy, torch, pyspark
# use methods: 
# - __init__: initialize an LM with user-defined hyperparameters
# - fit: trains a single-layer LM over a set of documents
# - post_train: re-trains a single-layer model by refining model.fit() parameters with additional documents
# - fine_tune: utilizes model.fit() parameters to train a second layer's model on a given set of documents
# - interpret: utilizes model parameters to predict labels for given documents
# - generate: utilizes model parameters to iteratively predict the next token
# - stencil: utilizes model parameters to iteratively attempt to reconstruct the next tokens of documents
# use attributes:
# - _documents: list, containing any documents passed through model.interpret() 
class LM(ABC):
    # purpose: initialize an LM with user-defined hyperparameters
    # arguments:
    # - m: int, no less than 1 for radius of context, building features over +-m tokens
    # - tokenizer: one of 'hr-bpe' or 'sentokenizer'; other values default to space-wise segmentation
    # - noise: float, greater than 0 used for sparsity smoothing
    # - positionally_encode: bool, with True implementing a sinusoidal positional encoding
    # - seed: initialization seed for model-level randomization
    # - positional: 'dependent' or 'independent', setting the positional information model
    # - space: bool, with True allowing space (' ') tokens to operate on their own. False right-justifies text
    # - attn_type: list, with none, one, or two of the strings 'accumulating' or 'backward' for vector aggregation
    # - do_ife: bool, with True mapping one-hots dimensions down to frequency-hot vectors
    # - runners: int, the number of map-reduce runners for data pre-processing
    # - hrbpe_kwargs: dict, of keyword arguments used to hyperparameterize LM-scoped hr-bpe tokenizers 
    # - gpu: bool, when True vectors and parameters move from numpy arrays to torch.cuda.current_device() tensors
    # - bits: int or None, for dimensionality reduction; None defaults to one-hot (standard basis) vectors for the model's vocabulary W. bits cannot be smaller than log2(|W|), which the Cipher defaults to if set too low
    # output: un-trained model class with user-specified attributes and related helper attributes
    def __init__(self, m = 10, tokenizer = 'hr-bpe', noise = 0.001, positionally_encode = True, seed = None, positional = 'dependent',
                 space = True, attn_type = [False], do_ife = True, runners = 0, hrbpe_kwargs = {}, gpu = False, bits = None): 
        # set basic user-defined parameters for data handling and aggregation
        self._runners = int(runners); self._gpu = bool(gpu)
        self._seed = int(seed); self._space = bool(space); self._attn_type = list(attn_type)
        # set tokenizer-related parameters
        self._tokenizer_name = str(tokenizer)
        if self._tokenizer_name == 'hr-bpe':
            self._hrbpe_kwargs = {'method': 'char', 'param_method': 'est_theta', 'reg_model': 'mixing', 
                                  'early_stop': True, 'num_batches': 100, 'batch_size': 10_000,
                                  'action_protect': ["\n","[*\(\{\[\)\}\]\.\?\!\,\;][ ]*\w", 
                                                     "\w[ ]*[*\(\{\[\)\}\]\.\?\!\,\;]"],
                                 } if not hrbpe_kwargs else dict(hrbpe_kwargs)
        # set context and dimensionality reduction parameters
        self._bits = int(bits) if bits is not None else None
        if do_ife and self._bits is None: # the user sets integer frequency encipherment
            self._cltype = 'frq'  
        elif self._bits is not None: # the user sets bitvector encipherment
            self._cltype = 'bits'
        else: # the user defaults to one-hot encoding
            self._cltype = 'form'
        self._cnull = 0 if ((self._cltype == 'frq') or (self._cltype == 'bits')) else ''
        # various flags that indicate the current model states and other model parameters 
        self._fine_tuned = False
        self._lorder = list(); self._ltypes = defaultdict(set) 
        if self._seed:
            np.random.seed(seed=self._seed)
        self._positional = str(positional); self._positionally_encode = bool(positionally_encode)
        self._noise = noise; self._m = m; self._As = {}; self._Ls = {}
        self._X = Counter(); self._F = Counter(); self._Xs = {}; self._Fs = defaultdict(Counter)
        self._ife = Counter(); self._tru_ife = Counter(); self._f0 = Counter()
        self._trXs = {}; self._trIs = {}
        self._Tds, self._Tls = defaultdict(Counter), defaultdict(lambda : defaultdict(Counter))
        self._D = defaultdict(set); self._L = defaultdict(lambda : defaultdict(set))
        self._alphas = defaultdict(list); self._total_sentences = 0; self._total_tokens = 0; self._max_sent = 0
        # containers for processed data, as well as object constructors and object handling methods
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        self.array = partial(torch.tensor, dtype = torch.double, device = torch.cuda.current_device()) if self._gpu else np.array
        self.intarray = partial(torch.tensor, dtype = torch.long, device = torch.cuda.current_device()) if self._gpu else np.array
        self.long = partial(torch.tensor, dtype = torch.long) if self._gpu else np.array
        self.double = partial(torch.tensor, dtype = torch.double) if self._gpu else np.array
        self.ones = partial(torch.ones, dtype = torch.double, device = torch.cuda.current_device()) if self._gpu else np.ones
        self.zeros = partial(torch.zeros, dtype = torch.double, device = torch.cuda.current_device()) if self._gpu else np.zeros
        self.cat = partial(torch.cat,  axis = -1) if self._gpu else np.concatenate
        self.log10 = torch.log10 if self._gpu else np.log10
        self.stack = torch.stack if self._gpu else np.array
        # self.view = lambda dims, vec: vec.view(*dims) if self._gpu else lambda dims, vec: vec.reshape(*dims)
    # purpose: trains an LM over a set of documents
    # arguments: 
    # - docs, a list (corpus) of lists documents of strings (sentences); 
    # - docs_name, a string name for the data (docs); 
    # - covering, a list (corpus) of lists (documents) of lists (sentences) of strings (whatevers), the innermost of which concatenate exactly to doc's strings (sentences); 
    # - all_layers, a dictionary (document-index:multi-tagged-documents) of dictionaries (tag-type:documents) of lists (tag-documents) of lists (tag-sentences) of strings (tags); 
    # - fine_tune, bool, with True indicating a 2-layer, attention-based model
    # prereqs: minimally, user provide a set of sentence-tokenized documents and a name for the training corpus
    # effect: trains a 1- or 2-layer model, powering other class methods (e.g., generate or interpret)
    def fit(self, docs, docs_name, covering = [], all_layers = defaultdict(lambda : defaultdict(list)), 
            fine_tune = False):
        # assure all covering segmentations match their documents
        if covering:
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
        # define the fit model's tokenizer method 
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
        all_data, streams = self.process_documents(docs, all_layers, covering, fine_tune = fine_tune) 
        # compute the marginals
        print('Computing marginal statistics...')
        self.compute_marginals()
        # build dense models
        print('Building dense output heads...')
        self.build_dense_output(fine_tune = fine_tune) 
        # build transition matrices for tag decoding
        self.build_trXs(all_data)
        # fine-tune a model for/using attention over predicted contexts
        if fine_tune:
            self.fine_tune(docs, covering = covering, all_layers = all_layers, streams = streams)
        # report model statistics
        print('Done.')
        print('Model params, types, encoding size, contexts, vec dim, max sent, and % capacity used:', 
              len(self._Fs[self._cltype]), len(self._ltypes['form']), len(self._ltypes[self._cltype]), 
              len(self._con_vocs[self._cltype]), self._vecdim, self._max_sent,
              round(100*len(self._Fs[self._cltype])/(len(self._con_vocs[self._cltype])*len(self._ltypes['form'])), 3))
        
        return streams
        
    ## now re-structured to train on layers by document
    # purpose: tokenize documents and absorb co-occurrences
    # arguments: 
    # - docs: see .fit;
    # - all_layers: see .fit
    # - covering: see .fit 
    # - update_ife: bool, with True indicating that the model will update its integer frequency encipherment protocol with docs' new data
    # - update_bow: bool, with True indicating that the model will update its bag of words model for positional encoding, using docs' new data
    # - fine_tune: see .fit
    # - post_train: bool, with True indicating that processing is for re-training the base-layer's model over additional documents
    # prereqs: not to be run by user, operated by system during .fit, .fine_tune, and .post_train methods
    # output: fully encoded data streams
    def process_documents(self, docs, all_layers = defaultdict(lambda : defaultdict(list)), 
                          covering = [], update_ife = False, update_bow = False, 
                          fine_tune = False, post_train = False):
        if (covering and self._tokenizer_name == 'hr-bpe') or (not covering): print('Tokenizing documents...')
        docs = ([json.dumps([self.tokenize(s) for s in doc]) for doc in tqdm(docs)]
                if (covering and self._tokenizer_name == 'hr-bpe') or (not covering) else [json.dumps(d) for d in covering])
        d_is = range(len(docs))
        all_data = list(zip(docs, [covering[d_i] if covering else [] for d_i in d_is], # docs and covering
                            [list(all_layers[d_i].keys()) for d_i in d_is], # ltypes
                            [list(all_layers[d_i].values()) for d_i in d_is])) # layers
        Fs = []; old_ife = Counter(self._ife)
        bc = {'m': int(self._m), 'old_ife': Counter(self._ife), 'post_train': post_train} # 'tokenizer': self.tokenize,
        if self._runners:
            print('Counting documents and aggregating counts...')
            sc = SparkContext(f"local[{self._runners}]", "IaMaN", self._runners, conf=conf)
#             SparkContext.setSystemProperty('spark.executor.memory', '1g')
            SparkContext.setSystemProperty('spark.driver.memory', '2g')
            bc = sc.broadcast(bc)
            Fs = Counter({ky: ct for (ky, ct) in tqdm(sc.parallelize(all_data)\
                                                        .flatMap(partial(count, bc = bc), preservesPartitioning=False)\
                                                        .reduceByKey(lambda a, b: a+b, numPartitions = int(len(docs)/10) + 1)\
                                                        .toLocalIterator())})
            print('Collecting pre-processed data...') # this should be mapped and collected with spark, as required
            docs, dcons, layers, ltypes, metas = [], [], [], [], []
            for i in range(0,ceil(len(all_data)/self._runners) + 1):
                batch = list(all_data[i*self._runners:(i+1)*self._runners])
                if not batch: continue
                bdocs, bdcons, blayers, bltypes, bmetas = zip(*list(tqdm(sc.parallelize(batch)\
                                                                           .map(partial(process_document, bc = bc), 
                                                                                preservesPartitioning=False).toLocalIterator())))
                docs.extend(bdocs); dcons.extend(bdcons); layers.extend(blayers); ltypes.extend(bltypes); metas.extend(bmetas)
            sc.stop()
        else:
            print('Counting documents...')
            all_Fs = [Counter({ky: ct for (ky, ct) in count([doc, cover, ltypes, layers], bc = bc)})
                      for d_i, (doc, cover, ltypes, layers) in tqdm(list(enumerate(all_data)))]
            print("Aggregating counts...")
            Fs = reduce(lambda a, b: a + b, tqdm(all_Fs))
            print('Collecting pre-processed data...')
            docs, dcons, layers, ltypes, metas = zip(*[process_document([doc, cover, ltypes, layers], bc = bc)
                                                       for d_i, (doc, cover, ltypes, layers) in tqdm(list(enumerate(all_data)))])
        print("Aggregating metadata...")
        meta = reduce(aggregate_processed, tqdm(metas))
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
        ## start forming the vocabularies, aggregations, and layers thereof
        self._ltypes = defaultdict(set); self._ctypes = set()
        self._ltypes['form'].add(''); self._ltypes['frq'].add(0)
        if self._cltype == 'bits': self._ltypes['bits'].add(0)
        if fine_tune and self._cltype == 'form':
            self._ltypes['form-attn'].add('')
        elif fine_tune and self._cltype == 'frq':
            self._ltypes['frq-attn'].add(0)
        elif fine_tune and self._cltype == 'bits':
            self._ltypes['bits-attn'].add(0)
        ## store the raw data
        self._F += Fs
        ## determines the integer frequencies for encipherment
        if not self._ife or update_ife:
            for t, c in Fs:
                if c[-1] == 'form': 
                    self._ife[c[0]] += Fs[(t,c)]
            self._ife_tot = sum(self._ife.values())
            self._ife[''] = 0
        if self._cltype == 'bits':
            print('Building cipher... ', end = '')
            ## creates an n-bit cipher for the context vocabulary
            self._cipher = Cipher([t for t, _ in self._ife.most_common()], self._bits)
            print(' done.')
        ## encode
        self._Fs = defaultdict(Counter)        
        print("Encoding parameters...")
        for t, c in tqdm(self._F):
            ents, encs, intensity = self.encode(t, c)
            impact = intensity*self._F[(t,c)]
            for enc in encs:
                ltype = t[1]+'-'+enc[-1] if enc[-1] == 'attn' else t[1]
                if (self._positional == 'independent' and c[-1] == 'attn' and fine_tune) or (c[-1] != 'attn'):
                    self._Fs[ltype][((t[0], ltype),enc)] += impact
                    self._ltypes[ltype].add(t[0])
                    self._ctypes.add(enc[-1])
                    if enc[-1] == 'attn':
                        self._ltypes['attn'].add(enc[0])
                if t[1] == 'form':
                    if (self._positional == 'independent' and enc[-1] == 'attn' and fine_tune) or (enc[-1] != 'attn'):
                        for ent in ents:
                            ltype = ent[1]+'-'+enc[-1] if enc[-1] == 'attn' else ent[1]; ent = (ent[0], ltype)
                            self._Fs[ltype][(ent,enc)] += impact
                            self._ltypes[ltype].add(ent[0])
                            self._ctypes.add(enc[-1])
                            if enc[-1] == 'attn':
                                self._ltypes['attn'].add(enc[0])
        print('Building target vocabularies...')
        # sets the vocabularies according to the current accumulated collection of data in a unified way
        # these are the individual layers' vocabularies
        self._zeds, self._vocs = {}, {}
        for ltype in tqdm(self._ltypes):
            if ltype == 'attn': 
                self._vocs[ltype] = Vocab([(t, ltype) for t in sorted(self._ltypes[ltype], key = lambda t: int(t))])
            else:
                self._vocs[ltype] = Vocab(sorted([(t, ltype) for t in self._ltypes[ltype]]))
            self._zeds[ltype] = np.zeros(len(self._vocs[ltype]))
        print('Pre-computing BOW probabilities...', end = '')
        # set dense bow vector-probabilities for the language model to fall back to when contextless
        self._P0 = {}
        self._P0['form'] = np.array([self._f0.get(t[0], 0) for t in self._vocs['form']])
        self._P0['form'][self._vocs['form']._type_index[('', 'form')]] = 1
        self._P0['form'] = self._P0['form']/self._P0['form'].sum()
        f_ife = Counter()
        for t in self._f0:
            f_ife[self._ife[t]] += self._f0[t]
        self._P0['frq'] = np.array([f_ife.get(t[0], 0) for t in self._vocs['frq']])
        self._P0['frq'][self._vocs['frq']._type_index[(0, 'frq')]] = 1
        self._P0['frq'] = self._P0['frq']/self._P0['frq'].sum()
        if self._cltype == 'bits':
            self._P0['bits'] = self.array(self._zeds['bits'])
            for t in self._vocs['form']:
                bits = self._cipher.sparse_encipher(t[0])
                for bit in bits:
                    self._P0['bits'][self._vocs['bits']._type_index[(int(bit), 'bits')]] += self._f0.get(t[0], 0)/len(bits)
            self._P0['bits'] = self._P0['bits']/self._P0['bits'].sum()
        print(' done.')
        print('Building context vocabularies...')
        # these are the combined context vocabularies, which spread the vocabs across the entire (2m+1) positional range
        self._con_vocs = defaultdict(list)
        for ctype in tqdm(self._ctypes):
            if self._positional == 'dependent':
                self._con_vocs[ctype] = Vocab([(t,r,ltype) for r in range(-self._m,self._m+1) for t, ltype in self._vocs[ctype]], 
                                              null = self._cnull)
            else:
                self._con_vocs[ctype] = Vocab([(t,0,ltype) for t, ltype in self._vocs[ctype]], 
                                              null = self._cnull)
        print('Pre-computing wave amplitudes...', end = '')
        ## these are pre-computed for positional encoding
        if self._positional == 'dependent': #  and (self._cltype != 'bits')
            if self._cltype == 'bits':
                self._positional_intensities = self.array(np.zeros(len(self._con_vocs[self._cltype])))
                for rad in range(-self._m,self._m+1):
                    for t, _ in self._vocs['form']:
                        impact = self.intensity(t, abs(rad))
                        _, encs, impact = self.encode(('', 'form'), (t, rad, 'form'))
                        self._positional_intensities[self.long([self._con_vocs[self._cltype].encode(enc) 
                                                                for enc in encs])] += impact/len(encs)
                self._positional_intensities /= self._positional_intensities.max()
            else:
                self._positional_intensities = self.array([self.intensity(t, abs(rad)) for rad in range(-self._m,self._m+1) 
                                                           for t, _ in self._vocs[self._cltype]])
        elif self._positional == 'independent' and (self._cltype != 'bits'):
            if 'bits':
                self._positional_intensities = [self.array(np.zeros(len(self._vocs[self._cltype])))]*(2*self._m + 1)
                for rad in range(-self._m,self._m+1):
                    for t, _ in self._vocs['form']:
                        impact = self.intensity(t, abs(rad))
                        _, encs, impact = self.encode(('', 'form'), (t, rad, 'form'))
                        self._positional_intensities[rad + self._m][self.long([self._vocs[self._cltype].encode(enc) 
                                                                               for enc in encs])] += impact/len(encs)
                for ix in range(len(self._positional_intensities)):
                    self._positional_intensities[ix] /= self._positional_intensities[ix].max()
            else:
                self._positional_intensities = [self.array([self.intensity(t, abs(rad)) for t, _ in self._vocs[self._cltype]])
                                                for rad in range(-self._m,self._m+1)]
        else:
            self._positional_intensities = self.array(np.zeros(len(self._con_vocs[self._cltype]))) + 1 
        print(' done.')
        print('Stacking output vocabularies for decoders...')
        # these are the combined output vocabulary (indexing the combined output vectors)
        # note: by not using con_voc for c-dist's output vocabulary, fine-tuning will mis-align
        self._vecsegs = {}; self._vecdim = 0; self._allvocs = []
        # this is only for fine tuning, and somehow should/could be witheld to speed calculations
        if self._cltype not in self._lorder:
            self._lorder += [self._cltype]
        for ltype in tqdm(self._lorder):
            self._vecsegs[ltype] = (self._vecdim, self._vecdim + len(self._vocs[ltype]))
            self._vecdim += len(self._vocs[ltype])
            self._allvocs += list(self._vocs[ltype])
        if post_train:
            streams = []
        else:
            print('Encoding data streams for torch processing...')
            streams = self.encode_data(docs, dcons, layers, ltypes)
            print(' done.')    
        return all_data, streams

    # purpose: determine encodings and their intensity for a type and context
    # arguments: 
    # - t, a tuple of two: (str, str), with the first being the string's type and the second being it's tag-type
    # - c, a tuple of three: (str, int, str), with the first being the string's type, the second being it's radius (distance) from t, and the third being its tag-type (currently always 'form').
    # prereqs: not run by user, operated by system during .process_documents and .build_dense_output
    # output: lists of the encoded types, contexts, and their intensities of interaction
    def encode(self, t, c):
        intensity = 1; ents = [None]
        if self._positionally_encode:
            intensity *= self.intensity(c[0], abs(c[1]))
        encs = [c]
        if c[-1] == 'form':
            if self._cltype == 'frq':
                encs = [self.if_encode_context(c)]
            elif self._cltype == 'bits':
                encs = [(int(i), c[1], 'bits') for i in self._cipher.sparse_encipher(c[0])]
            else:
                encs = [c]
        if ('attn' in c[-1]) or (self._positional == 'independent'):
            encs = [(enc[0], 0, enc[-1]) for enc in encs] # int(enc[1]/abs(enc[1]) if enc[1] else enc[1])
        if t and t[1] == 'form': # [0]
            ents = [(self._ife[t[0]], 'frq')]
            if self._cltype == 'bits':
                ents = [(int(i), 'bits') for i in self._cipher.sparse_encipher(t[0])]
        return ents, encs, intensity
    # purpose: determine the amplitude of the wavefunction for co-occurrence weighting
    # arguments:
    # - c: int (integer frequency) or string (keying an integer frequency) in a bag of whatevers model (._p0)
    # - dm: int, radius of co-occurrance (distance between c and the other type)
    # prereqs: not run by user, operated by system during .encode
    # output: float, intensity of measurement
    def intensity(self, c, dm):
        if type(c) == int:
            theta = c/self._M0
        else:
            theta = self._p0.get(c, 0)
        return (np.cos((2*np.pi*dm*theta) if (c or type(c) == bool) else np.pi) + 1)/2
    
    # purpose: map string-types to integer frequencies as a dimensionality reduction
    # arguments:
    # - c, a tuple of three: (str, int, str) (see .encode)
    # prereqs: not run by user, operated by system during .encode
    # output: a tuple of three: (str, int, str) (see type c from .encode)
    def if_encode_context(self, c):
        return(tuple([self._ife.get(c[0], 0), c[1], 'frq']))
    
    # purpose: get a document-length vector of superimposed amplitudes based on its internal bag of whatevers distribution
    # arguments:
    # - doc: a list of strings (whatevers) representing an unrolled document
    # prereqs: not run by user, operated by system during .encode_data
    # output: a vector of weights for document-level whatever-vector aggregation
    def get_amps(self, doc):
        if False not in self._attn_type:
            wav_idx, wav_f, wav_M, wav_T = wave_index(doc)
            return sum([get_wave(w, wav_idx, self._f, self._M, wav_T, # wav_f, wav_M, 
                                 'accumulating' in self._attn_type, 'backward' in self._attn_type) 
                        for w in wav_f if w])
        else:
            return np.ones(len(doc))
        
    # purpose: encode the corpus for rapid data loading (required for practical use of gpu-based processing)
    # arguments:
    # - docs: a list (corpus) of lists (documents) of lists (sentences) of strings (whatevers), the innermost of which are now tokenized sentences 
    # - dcons: a list (corpus) of lists (documents) of lists (whatevers) of tuples (contexts), the innermost of which are typed positional contexts (see LM.encode, as well as get_context and process_document from src/utils/munge.py)
    # - layers: list (corpus) of lists (documents) of lists (tag layers) of lists (sentences) of strings (tags)
    # - ltypes: list (corpus) of lists (documents) of lists (tag types)
    # prereqs: not run by user, operated by system during .process_documents
    # output: a list of data streams (see .encode_stream) for each input and output
    def encode_data(self, docs, dcons, layers, ltypes):
        encoded = []
        for doc, dcon, doc_layers, doc_ltypes in zip(docs, dcons, layers, ltypes):
            encoded.append({'cons': self.encode_stream(dcon, 'cons'), # encode the context
                            'amps': self.encode_stream(dcon, 'amps'), # encode the context's wave amplitude
                            'm0ps': self.encode_stream(dcon, 'm0ps')}) # encode the context's sign of radius
            encoded[-1]['form'] = self.encode_stream(unroll(doc), 'form') # encode the whatever types
            for layer, ltype in zip(doc_layers, doc_ltypes): # iterate through the document's tag layers
                encoded[-1][ltype] = self.encode_stream(unroll(layer), ltype) # encode the tag layer's types
            encoded[-1]['amp'] = self.double(self.get_amps(unroll(doc))) # encode document aggregation weights
            if self._cltype == 'frq': # check if the frequency-context type requires encoding (for ife only)
                encoded[-1]['frq'] = self.encode_stream(unroll(doc), 'frq') # encode the integer frequency
        return encoded
    
    # purpose: encode a stream of given layer type (ltype)
    # arguments:
    # - stream: a list of ground truth data. numerical types are system controlled, categorical include user-defined tag layers
    # - ltype: str, uniquely encoding the stream's type of information. currently used ltypes for data streams are: 
    # prereqs: not run by user, operated by system during .encode_data
    # output: a system scoped (cpu or gpu) and possibly multidimensional array of encoded stream data
    def encode_stream(self, stream, ltype):
        if ltype == 'cons':
            # return [self.long([self._con_vocs[self._cltype].encode(self.encode(('', 'form'),c)[1]) 
            #                    for c in cs]) for cs in stream]
            return [self.long([self._con_vocs[self._cltype].encode(enc) 
                               for c in cs for enc in self.encode(('', 'form'),c)[1]]) for cs in stream]
        elif ltype == 'amps':
            # return [self.double([self.encode(('', 'form'),c)[-1] for c in cs]) for cs in stream]
            return [self.double([amps/len(encs) # if enc[0] else 0.
                                 for c in cs for _, encs, amps in [self.encode(('', 'form'),c)] 
                                 for enc in encs]) for cs in stream]
        elif ltype == 'm0ps':
            return [self.long([int(c[1]/abs(c[1]) if c[1] else c[1])
                               for c in cs for enc in self.encode(('', 'form'),c)[1]]) for cs in stream]
        elif ltype == 'frq':
            return self.long([self._vocs[ltype].encode((self._ife[t[0]], ltype)) for t in stream])
        else:
            return self.long([self._vocs[ltype].encode((t, ltype)) for t in stream])
    
    # purpose: manage the encoding of streams of all layer types for a corpus
    # arguments:
    # - docs: a corpus of sentence-tokenized documents (see .fit)
    # - covering: a gold standard tokenization for the corpus (see .fit)
    # - all_layers: multiple layers of gold standard tags (see .fit)
    # prereqs: not run by user, operated by system during secondary/predictive (post .fit) exposures to data
    # output: system scoped (cpu or gpu) and possibly multidimensional arrays of encoded streams of data
    def encode_streams(self, docs, covering = [], all_layers = defaultdict(lambda : defaultdict(list))):
        # tokenize documents
        docs = ([json.dumps([self.tokenize(s) for s in doc]) for doc in docs]
                if (covering and self._tokenizer_name == 'hr-bpe') or (not covering) else [json.dumps(d) for d in covering])
        d_is = range(len(docs))
        all_data = list(zip(docs, [covering[d_i] if covering else [] for d_i in d_is], # docs and covering
                            [list(all_layers[d_i].keys()) for d_i in d_is], # ltypes
                            [list(all_layers[d_i].values()) for d_i in d_is])) # layers
        bc = {'m': int(self._m), 'old_ife': Counter(self._ife), 'post_train': False}
        if self._runners:
            sc = SparkContext(f"local[{self._runners}]", "IaMaN", self._runners, conf=conf)
            # SparkContext.setSystemProperty('spark.executor.memory', '1g')
            SparkContext.setSystemProperty('spark.driver.memory', '2g')
            bc = sc.broadcast(bc)
            docs, dcons, layers, ltypes, metas = [], [], [], [], []
            for i in range(0,ceil(len(all_data)/self._runners) + 1):
                batch = list(all_data[i*self._runners:(i+1)*self._runners])
                if not batch: continue
                # collecting pre-processed data
                bdocs, bdcons, blayers, bltypes, bmetas = zip(*list(tqdm(sc.parallelize(batch)\
                                                                           .map(partial(process_document, bc = bc), 
                                                                                preservesPartitioning=False).toLocalIterator())))
                docs.extend(bdocs); dcons.extend(bdcons); layers.extend(blayers); ltypes.extend(bltypes); metas.extend(bmetas)
            sc.stop()
        else:            
            # collect pre-processed data
            docs, dcons, layers, ltypes, metas = zip(*[process_document([doc, cover, ltypes, layers], bc = bc)
                                                       for d_i, (doc, cover, ltypes, layers) in list(enumerate(all_data))])
        streams = self.encode_data(docs, dcons, layers, ltypes)
        return streams, docs
        
    # purpose: compute marginal co-occurrence statistics used in the analytic noise model
    # arguments: NA, calculations are based on the existence of cached co-occurrences within self._Fs
    # prereqs: not run by user, operated by system during process_documents
    # output: NA, function builds system parameters (statistics) used in balancing noise for sparsity handling
    def compute_marginals(self):
        numcon = len(self._con_vocs[self._cltype])
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
            self._beta[ltype] = defaultdict(lambda : 0., {t: Cp[t]/C_tot for t in T})
            self._zeta[ltype] = defaultdict(lambda : 0., {c: Tp[c]/C_tot for c in C})
            self._Cn[ltype] = defaultdict(lambda : numcon, {t: numcon - len(TCs[t]) for t in T})
            self._Tn[ltype] = defaultdict(lambda : numtok, {c: len(T) - len(CTs[c]) for c in C})
            if ltype == 'form':
                self._f = Counter({t[0]: T[t] for t in T if t[1] == 'form'}); self._f[''] = 1
                self._M = sum(self._f.values())
                
    # purpose: wrap the construction of first-order parameter matrices for the system's current store of co-occurrences (in self._Fs)
    # arguments: fine_tune: bool, see .process_documents, used to restrict the construction of second order parameter matrices
    # prereqs: not run by user, operated by system during process_documents
    # output: NA, function builds parameter matrices used in forward prediction of whatevers and tags
    def build_dense_output(self, fine_tune = False):
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
                X[self._vocs[ltype]._type_index[t], self._con_vocs[ctype]._type_index[c]] = self._beta[ltype][t]*self._Fs[ltype][(t,c)]/fl[t]
            X[X==0] = self._noise; X /= X.sum(axis = 1)[:,None]; X = np.nan_to_num(-np.log10(X))
            self._Xs[ltype] = self.array(X); del(X)
            if fine_tune and (ltype == self._cltype or ctype == 'attn'):
                # now build the transition matrices
                A = np.zeros((len(self._vocs[ltype]), len(self._con_vocs[ctype])))
                for j in range(A.shape[1]): ## add the negative information
                    if self._Tn[ltype][self._con_vocs[ctype][j]]:
                        A[:,j] = (1 - self._zeta[ltype][self._con_vocs[ctype][j]])/self._Tn[ltype][self._con_vocs[ctype][j]]
                for t, c in self._Fs[ltype]: ## add the positive information
                    A[self._vocs[ltype]._type_index[t], self._con_vocs[ctype]._type_index[c]] = self._zeta[ltype][c]*self._Fs[ltype][(t,c)]/cl[c]
                A[A==0] = self._noise; A /= A.sum(axis = 0); A = np.nan_to_num(-np.log10(A))
                self._As[ltype] = self.array(A); del(A)
                
    # purpose: re-trains a previously-trained single-layer model by refining model.fit() parameters with additional documents
    # arguments: 
    # - ptdocs: a list (corpus) of lists (documents) of lists (sentences). (see .fit for further details)
    # - update_ife: bool, with True indicating that the updated model's integer frequency encipherment should be updated. (only has affect when ife is used)
    # - update_bow: bool, with True indicating that the updated model's bag of whatevers positional encoding model should be updated (only has affect when positional encoding is used)
    # - fine_tune: bool, with True indicating that sparse poritions of the 2-layer, attention-based model should be updated as well, i.e., which means that the .fine_tune method will likely be used afterwards to re-train the second order model away from ptdocs and towards a small target set.
    # prereqs: requires a previously-trained model to exist and be updated. intended for use in densification of first order models, and thus refined second order models. for low-footprint models, use should train an initial model using .fit on the target (gold standard) training set. following this, .post_train should be used to densify the statistics of the target inputs and outputs using external data  (ptdocs).
    # output: a re-trained model, potentially in need of being .fine_tune()'ed on target data.
    def post_train(self, ptdocs, update_ife = False, update_bow = False, fine_tune = False):
        if ptdocs:
            ptdocs = [["".join(s) for s in d] for d in ptdocs]
            print("Processing post-training documents...")
            ptdata, ptstreams = self.process_documents(ptdocs, update_ife = update_ife, update_bow = update_bow, post_train = True)
            print("Re-computing marginal statistics...")
            self.compute_marginals()
            print("Re-building dense output heads...")
            self.build_dense_output(fine_tune = fine_tune)
            # report model statistics
            print('Done.')
            print('Model params, types, encoding size, contexts, vec dim, max sent, and % capacity used:', 
                  len(self._Fs[self._cltype]), len(self._ltypes['form']), len(self._ltypes[self._cltype]), 
                  len(self._con_vocs[self._cltype]), self._vecdim, self._max_sent,
                  round(100*len(self._Fs[self._cltype])/(len(self._con_vocs[self._cltype])*len(self._ltypes['form'])), 3))

    # purpose: builds transition matrices for token-level viterbi decoding of tags
    # arguments: all_data: list (corpus) of lists (documents) of lists (layers) of lists (sentences) of strings (whatevers or tags)
    # prereqs: not a user-run function, requires a processed (including tokenized) corpus of data
    # output: parameter matrix attributes are stored within the object for later use in decoding within the .interpret method
    def build_trXs(self, all_data): # viterbi decoding will work best with token-level transitions
        print("Counting for transition matrices...")
        self._trFs = defaultdict(Counter)
        for d_i, (doc, cover, ltypes, layers) in tqdm(list(enumerate(all_data))):          
            for ltype, layer in zip(ltypes, layers):
                lstream = [''] + [lt for ls in layer for lt in ls]
                self._trFs[ltype] += Counter(list(zip(lstream[1:], lstream[:-1])))
            d, _, layers, ltypes, _ = process_document((doc, cover, ltypes, layers), bc = {'m': int(self._m), 
                                                                                           'old_ife': Counter(self._ife), 
                                                                                           'post_train': False})
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
                    self._trIs[ltype][self._vocs[ltype]._type_index[(t, ltype)]] += self._trFs[ltype][(t,c)]
            self._trIs[ltype] /= self._trIs[ltype].sum() # record the initial state probabilities
            for t, c in self._trFs[ltype]: ## add the positive information
                if not c: continue
                self._trXs[ltype][self._vocs[ltype]._type_index[(t, ltype)], 
                                  self._vocs[ltype]._type_index[(c, ltype)]] = self._beta[ltype][(t, ltype)]*self._trFs[ltype][(t,c)]/fl[t]
            for j in range(self._trXs[ltype].shape[1]): ## add the negative information
                zees = self._trXs[ltype][:,j] == 0.; Cn = sum(zees)
                nonz = self._trXs[ltype][:,j] != 0.; Cp = sum(nonz)
                bet = Cp/(Cn + Cp)
                if Cn: 
                    self._trXs[ltype][zees,j] = (1 - bet)/Cn
                    self._trXs[ltype][nonz,j] *= bet
                self._trXs[ltype][:,j] /= self._trXs[ltype][:,j].sum()

    # purpose: abstracts inner products, in-line with the used computational framework
    # arguments: 
    # - x: array (left) in inner product
    # - y: array (right) in inner product
    # prereqs: an initialized model class
    # output: a tensor of type (on gpu or cpu) given by the computational framework
    def dot(self, x, y):
        return x.matmul(y) if self._gpu else x.dot(y)

    # purpose: abstracts array re-shaping, in-line with the used computational framework
    # arguments: 
    # - dims: tuple of ints, indicating the target dimensionality (shape), with -1 leaving the size of any dimension free
    # - vec: array to be re-shaped, must have the same product dimension as the target 
    # prereqs: an initialized model class
    # output: an array of target (dims) dimension
    def view(self, dims, vec):
        return vec.view(*dims) if self._gpu else vec.reshape(*dims)

    # purpose: abstracts outer products, in-line with the used computational framework
    # arguments:
    # - x: array (left) of one dimension in outer product
    # - y: array (right) of one dimension in outer product
    # prereqs: an initialized model class
    # output: a two-dimensional array of outer dimension
    def outer(self, x, y):
        if self._gpu:
            return x.view(-1,1) * y
        else:
            return np.outer(x,y)
    
    # purpose: abstracts finding an array's min value, in-line with the used computational framework
    # arguments: 
    # - x: array (left) of arbitrary dimension
    # - axis: int, defining the axis of minimization
    # prereqs: an initialized model class
    # output: the scalar minimizer of the input array      
    def min(self, x, axis = 1):
        if self._gpu:
            return torch.min(x, axis = axis).values
        else:
            return np.min(x, axis = axis)

    # purpose: make a first order prediction from the given contexts and their amplitudes for the given layer (tag) types
    # arguments: 
    # - amps: array (contexts) of floats (amplitudes) 
    # - cons: array (contexts) of ints (types)
    # - ltypes: list of strings (tag types) for which to make predictions
    # prereqs: a trained first-order model (also used in second-order models), obtained after running .fit
    # output: a concatenation of prediction vectors in the order of layers specified by ltypes
    def P1(self, amps, cons, ltypes):
        Ps = []; Csum = (self._Cvs[self._cltype][cons]*amps).sum()
        for ltype in ltypes:
            Ps.append((amps*self._Xs[ltype][:,cons]).sum(axis = 1))
            Ps[-1] -= self.log10(self._Tvs[ltype]/Csum)
            Ps[-1] = 10**-(Ps[-1] - Ps[-1].min()); Ps[-1] /= Ps[-1].sum()
        return self.cat(Ps)

    # purpose: manage the computation of first order predictions (a representation) for a stream of data
    # arguments: 
    # - stream: a possibly multidimensional array of encoded stream data (see output from: .encode_stream)
    # - predict_contexts: bool, if True predicts vectors of context encoding, otherwise predicts according to ltypes
    # - forward: bool, if True decodes vectors in sequential (left to right) directional (LM'ing) order
    # - ltypes: list (layers) of strings (layer-types) to be decoded within the representation vectors
    # - to_cpu: bool, with True indicating that vectors in the stream should be detached from the gpu (if one's being used)
    # - previous_vecs: list of 1-d arrays being passed through from a previous sequential decoding step (only needed if forward == True)
    # prereqs: a model trained using .fit
    # output: a list of prediction vectors for the stream of data
    def R1(self, stream, predict_contexts = False, forward = True, 
           ltypes = [], to_cpu = True, previous_vecs = []):
        vecs = [] + previous_vecs
        if predict_contexts:
            ltypes = [self._cltype]
        elif not ltypes:
            ltypes = list(self._lorder) 
        for wi in range(len(vecs),len(stream['amps'])):
            amps, cons, m0ps = (self.double(detach(stream['amps'][wi])), 
                                self.long(detach(stream['cons'][wi])),
                                self.long(detach(stream['m0ps'][wi])))
            if self._gpu: amps, cons = to_gpu(amps), to_gpu(cons)
            if not forward:
                for ci, m0p in enumerate(m0ps):
                    if m0p >= 0:
                        amps[ci] = 0
            if not amps.sum():
                if predict_contexts:
                    vecs.append(self.array(detach(self._P0[self._cltype])))
                else:
                    vecs.append(self.array(detach(self._P0['form'])))
            else:   
                vecs.append(self.P1(amps, cons, ltypes))
            if to_cpu:
                vecs[-1] = detach(vecs[-1])
        return vecs

    # purpose: utilizes model.fit() parameters to train a second layer's model on a given set of documents
    # arguments: 
    # - docs: a list (corpus) of lists (documents) lists (sentences) (see .fit)
    # - covering: see .fit
    # - all_layers: see .fit
    # - streams: a list (corpus) of possibly multidimensional arrays of encoded streams of data (see output from: .encode_stream). note: if data streams are provided, they will be utilized (avoiding pre-processing) instead of the docs.
    # prereqs: a trained single-layer model, responsible for producing an initial representation over the data
    # output: learned attributes, principally including a second layer of matrix parameters encoding the response to attended features
    def fine_tune(self, docs, covering = [], all_layers = defaultdict(lambda : defaultdict(list)), streams = []):
        if not streams:
            streams, _ = self.encode_streams(docs, covering = covering, all_layers = all_layers)
        self._Ls = {ltype: self.array(np.zeros((len(self._vocs[ltype]), len(self._con_vocs[self._cltype])))) 
                    for ltype in self._ltypes}
        print("Fine-tuning dense output heads...")
        for stream in tqdm(streams):
            # get the vectors and attention weights for the document
            cvecs = self.R1(stream, predict_contexts = True, to_cpu = False, forward = True); 
            avecs = self.attend(cvecs)
            ## these should be stacked and applied to a contiguous portion of the dense distribution
            for wi, (amps, cons) in enumerate(zip(stream['amps'], stream['cons'])):
                if self._gpu: amps, cons = to_gpu(amps), to_gpu(cons)
                # the likelihood context vector from all positional predictions
                dcv = self.dense_context(amps, cons, avecs, wi)
                for ltype in self._Ls:
                    if ltype not in stream: #  or ltype == 'form' # this implies uniform params in Ls['form']!
                        continue
                    self._Ls[ltype][stream[ltype][wi],:] += dcv
        for ltype in self._Ls:
            self._Ls[ltype][self._Ls[ltype]==0] = self._noise
            self._Ls[ltype] /= self._Ls[ltype].sum(axis=1)[:,None]
            self._Ls[ltype] = self.array(np.nan_to_num(-np.log10(detach(self._Ls[ltype]))))
        self._fine_tuned = True

    # purpose: manage the computation of second order predictions from a list of first order predictions
    # arguments: 
    # - cvecs: list of arrays (context) being attended with each other for an underlying data stream
    # - previous_avecs: list of arrays (prediction vectors) directionally produced (left to right) from cvecs
    # - forward: bool, controling directionality of featurization->prediction (see .R1)
    # prereqs: a trained second-order model, leant from using the class' .fine_tune method
    # output: a list of arrays blending the predictions from .R1 via a user-defined attention model
    def attend(self, cvecs, previous_avecs = [], forward = True):
        avecs = [] + previous_avecs
        for wi in range(len(avecs), len(cvecs)):
            cmax = len(cvecs) if forward else wi + 1
            a = self.array(np.zeros(len(self._con_vocs[self._cltype]))) + 1
            dcv = self.array(np.zeros(len(self._con_vocs[self._cltype]))) # note this sets and keeps zero
            if self._positional == 'independent':
                # gather the (2*m+1)-location positional attention distribution
                pa = self.dot(self._As[self._cltype+'-attn'].T, cvecs[wi])
                pa = 10**-(pa - pa.max())
                if pa.sum(): pa = pa/pa.sum()
                # gather the |V|-context semantic distribution
                sa = self.dot(self._As[self._cltype].T, cvecs[wi])
                sa = 10**-(sa - sa.max())
                if sa.sum(): sa = sa/sa.sum()
                avec = sum([cvecs[ci]*pa[ci-wi+self._m]*sa*self._positional_intensities[ci-wi+self._m] 
                            for ci in range(max([wi-self._m, 0]), min([wi+self._m+1, len(cvecs[:cmax])]))])
            elif self._positional == 'dependent':
                window = np.array(range(max([wi-self._m, 0]),min([wi+self._m+1, len(cvecs[:cmax])])))
                radii = window - wi
                mindex = self._m + radii[0]
                mincol = mindex*len(self._vocs[self._cltype])
                maxcol = mincol + len(window)*len(self._vocs[self._cltype])
                # concatenate the likelihood context vector from all positional predictions
                dcv[mincol:maxcol] += self.cat(cvecs[window[0]:window[-1]+1])
                dcv[mincol:maxcol] *= self.array(detach(self._positional_intensities[mincol:maxcol]))
                a = self.dot(self._As[self._cltype].T, cvecs[wi])
                a = 10**-(a - a.max()); a /= a.sum(); dcv *= a
                avec = self.view((2*self._m + 1,len(self._vocs[self._cltype])), dcv).sum(axis = 0)
            avecs.append(avec/avec.sum())
        return avecs

    # purpose: superimpose a second order context vector from a list of vectors and the original data stream
    # arguments: 
    # - amps: an array of floats (amplitudes) 
    # - cons: an array of ints (context types)
    # - avecs: an array of first-order prediction vectors (output from .R1)
    # - wi: int, the position within the list of vectors (and original data stream) around which the context is centered
    # prereqs: not user-driven, see .fine_tune for implicit use
    # output: an array of comtext-dimensionality
    def dense_context(self, amps, cons, avecs, wi): 
        dcv = self.array(np.zeros(len(self._con_vocs[self._cltype])))
        window = self.long(range(max([wi-self._m, 0]),min([wi+self._m+1, len(avecs)])))
        radii = window - wi 
        mindex = self._m + radii[0]
        mincol = mindex*len(self._vocs[self._cltype])
        maxcol = mincol + len(window)*len(self._vocs[self._cltype])
        if self._positional == 'independent':
            dcv += sum(avecs[window[0]:window[-1]+1])
            # dcv[mincol:maxcol] += sum(avecs[window[0]:window[-1]+1])
        elif self._positional == 'dependent':
            dcv[mincol:maxcol] += self.cat(avecs[window[0]:window[-1]+1])
        # dcv[cons] += amps
        dcvsum = dcv.sum(); ampssum = amps.sum() # maybe only needed to stabilize non-generative tasks?
        if dcvsum:
            if ampssum:
                dcv = dcv*ampssum/dcvsum
                dcv[cons] += amps; dcv /= 2
                # dcv[cons] += amps*dcvsum/amps.sum(); dcv /= 2
        else:
            dcv[cons] = amps
        return dcv

    # purpose: produce second-order prediction for a stream of data at a point (wi)
    # arguments: 
    # - amps: an array of floats (amplitudes)
    # - cons: an array of ints (context indices)
    # - avecs: an array of first-order prediction vectors (output from .R1)
    # - wi: int, the index-location within the stream for which the prediction is being made
    # - ltypes: list (layers) of strings (layer types) indicating which layers for which to provide a second order prediction
    # - dense_predict: bool, if True indicates that the model will flippantly use a second-order context on a first-order model's parameters (experimental).
    # prereqs: not user driven (see .R2), requires a trained second-order model using .fit(..., fine_tune = True)
    # output: a prediction vectors for those layer types indicated in ltypes
    def P2(self, amps, cons, avecs, wi, ltypes, dense_predict):
        dcv = self.dense_context(amps, cons, avecs, wi)
        Ps = []; Csum = self.dot(dcv, self._Cvs[self._cltype])
        for ltype in ltypes:
            if dense_predict:
                Ps.append(self.dot(self._Xs[ltype], dcv))
            else:
                Ps.append(self.dot(self._Ls[ltype], dcv))
            Ps[-1] -= self.log10(self._Tvs[ltype]/Csum)
            Ps[-1] = 10**-(Ps[-1] - Ps[-1].min()); Ps[-1] /= Ps[-1].sum()
            # note the next step applies the 'residual' connection from self.P1)
            Ps[-1] = Ps[-1] + self.P1(amps, cons, [ltype]); Ps[-1] /= Ps[-1].sum() 
        return self.cat(Ps)

    # purpose: manage the computation of second order predictions (a representation) for a stream of data
    # arguments: 
    # prereqs: 
    # output:     
    def R2(self, stream, predict_contexts = False, forward = True, ltypes = [], to_cpu = True, 
           previous_vecs = [], previous_avecs = [], previous_cvecs = [], dense_predict = True):
        vecs = [] + previous_vecs; cvecs = [] + previous_cvecs; avecs = [] + previous_avecs
        if self._fine_tuned: dense_predict = False
        if predict_contexts:
            ltypes = [self._cltype]
        elif not ltypes:
            ltypes = list(self._lorder) 
        ##
        cvecs = self.R1(stream, predict_contexts = True, to_cpu = False, 
                        forward = forward, previous_vecs = cvecs)
        avecs = self.attend(cvecs, previous_avecs = avecs, forward = forward)
        ##
        for wi in range(len(vecs), len(stream['amps'])):
            amps, cons, m0ps = (self.double(detach(stream['amps'][wi])), 
                                self.long(detach(stream['cons'][wi])),
                                self.long(detach(stream['m0ps'][wi])))
            if self._gpu:
                amps, cons = to_gpu(amps), to_gpu(cons)
            if not forward:
                for ci, m0p in enumerate(m0ps):
                    if m0p >= 0:
                        amps[ci] = 0
            if not amps.sum():
                vecs.append(self.array(detach(self._P0['form'])))
            else:
                vecs.append(self.P2(amps, cons, avecs if forward else avecs[:wi + 1], wi, ltypes, dense_predict))
            if to_cpu: 
                vecs[-1] = detach(vecs[-1])
        return vecs, avecs, cvecs

    # purpose: establish a rapport with whatever (and whatever they construct), i.e., absorb predictions about whatever and record implications
    # arguments: 
    # - wi: int, representing the character-position of whatever within its pre-tokenized superstring
    # - w: str, representing the fundamental type of whatever 
    # - eot: bool, with True representing whatever's status as the end of a token
    # - eos: bool, with True representing whatever's status as the end of a sentence
    # - eod: bool, with True representing whatever's status as the end of a document
    # - nov: bool, with True representing whatever's status as a type not before seen in its document
    # - atn: float, constructed as the superposition of wave amplitudes from document whatevers and their siusoidal bag (BOW) model
    # - vec: array, constructed via prediction(s) based on sliding-window contexts of either first or second order and learnt model parameters
    # - seed: int, randomizing sampling conducted by the function for absorption of predicted semantics
    # - all_vecs: list of arrays, representing all vectors having been constructed for the overall data stream, used in aggregation
    # - all_atns: list of floats, representing all aggregations weights used for aggregation within the data stream
    # - all_nrms: list of norms positively-derived from weights, and also used in aggregations within the data stream
    # - all_ixs: list of character indices for all whatevers within the data stream
    # - tags: dict of typed tags to be applied to this whatever
    # prereqs: not a user-driven function, numerous arguments are provided for application to the system's internal document model in situ of prediction
    # output: NA, .grok only absorbs information (and takes actions, where appropriate) given the provided predictions and situational context
    def grok(self, wi, w, eot, eos, eod, nov, atn, vec, seed, 
             all_vecs, all_atns, all_nrms, all_ixs, tags = {}):
        self._whatevers.append(Whatever(w, ix = self._ix, sep = eot, nov = nov, 
                                        atn = atn, vec = all_vecs[0][wi])) # vec
        self._w_set.add(w); self._ix += len(w)
        # apply token-level information to build the next level sequence
        seps = [eot, eos, eod]
        if eot:
            wis = [wi - wix for wix in range(len(self._whatevers))]
            kwargs = {'sep': seps[1], 'nrm': all_nrms[1][all_ixs[1][wi]], 
                      'atn': all_atns[1][all_ixs[1][wi]], 'vec': all_vecs[1][all_ixs[1][wi]]}
            for ltype in ['lem', 'sen', 'pos', 'ent', 'dep', 'sup', 'infs']:
                kwargs[ltype] = tags.get(ltype, None)
            self._tokens.append(Token(self._whatevers, 
                                      ix = self._ix - len("".join([w._form for w in self._whatevers])), 
                                      **kwargs)); self._whatevers = []
            # apply sentence-level information to build the next level sequence
            if kwargs['sep']: 
                kwargs = {'sep': seps[2], 'nrm': all_nrms[2][all_ixs[2][wi]],
                          'atn': all_atns[2][all_ixs[2][wi]], 'vec': all_vecs[2][all_ixs[2][wi]]}
                # 'sty' prediction is a sentence-level mult-class classification tag (sentence type)
                for ltype in ['sty']:
                    kwargs[ltype] = tags.get(ltype, None)
                self._sentences.append(Sentence(self._tokens, 
                                                ix = self._ix - len("".join([w._form for t in self._tokens 
                                                                             for w in t._whatevers])),
                                                **kwargs)); self._tokens = []
                # apply document-level information to build the next level sequence
                if kwargs['sep']:
                    kwargs = {'nrm': all_nrms[3][all_ixs[3][wi]], 'atn': all_atns[3][all_ixs[3][wi]], 
                              'vec': all_vecs[3][all_ixs[3][wi]]}
                    self._documents.append(Document(self._sentences, 
                                                    ix = self._ix - len("".join([w._form for s in self._sentences 
                                                                                 for t in s._tokens 
                                                                                 for w in t._whatevers])),
                                                    **kwargs)); 
                    self._w_set = set(); self._ix = 0; self._sentences = []

    # purpose: perform a viterbi-algorithm decoding of the given set of probabilistic vectors
    # arguments: 
    # - vecs: a list of arrays (probability distributions)
    # - ltype: str, indicating the layer-type (including tags) for predictions to be decoded
    # prereqs: a sequence of prediction arrays with sections tagged for the different types (tags) of predictions to be made
    # output: a list of decoded tags (strings)
    def viterbi(self, vecs, ltype):
        # start off the chain of probabilities 
        V = [{}]; segsum = sum(vecs[0][self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
        for t in self._vocs[ltype]:
            V[0][t] = {"P": (self._trIs[ltype][self._vocs[ltype]._type_index[t]] * # initial state
                             (vecs[0][self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._vocs[ltype]._type_index[t]]/segsum)), # emission potential
                       "pt": None}
        # continue with chains for each branching point
        for vec in vecs[1:]:
            V.append({}); i = len(V) - 1
            maxP = 0.
            segsum = sum(vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
            for t in self._vocs[ltype]:
                P, pt = max([(V[i - 1][pt]["P"] * # probability of chain to this point
                              (vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._vocs[ltype]._type_index[t]]/segsum) * # emission potential
                              self._trXs[ltype][self._vocs[ltype]._type_index[t],self._vocs[ltype]._type_index[pt]], pt) # transmission potential
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

    # purpose: provide both the most-likely prediction (arg-max) and prediction Counter for a given point
    # arguments: 
    # - vec: an array, representing the probabilistic vector over which the prediction is sampled
    # - ltype: str, representing the type of tag to be sampled (pointing to its vocabulary)
    # prereqs: not user driven, implicitly operated by one of .generate, .interpret, or .stencit, and requiring a trained model and prediction vector
    # output:  a tuple, consiting of a string (the predicted type) and its prediction Counter
    def output(self, vec, ltype):
        segsum = sum(vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]])
        layer_Ps = Counter({t: vec[self._vecsegs[ltype][0]:self._vecsegs[ltype][1]][self._vocs[ltype]._type_index[t]]/segsum
                            for ix, t in enumerate(self._vocs[ltype])})
        layer_that = list(layer_Ps.most_common(1))[0][0]
        return(layer_that, layer_Ps)

    # purpose: require a token within a sentence to relent an iterator containing its dependencies
    # arguments: 
    # - sentence: a list of Token objects
    # - t_i: a int representing the position within the sentence whose Token is being traversed
    # prereqs: a constructed docuemtn object, consisting of a list of sentences
    # output: an iterator of indices for other Tokens within the sentence that are linguistically dependent (up to tags) on t_i'th token
    def yield_branch(self, sentence, t_i):
        yield t_i
        for i_t_i in sentence[t_i]._infs:
            yield from self.yield_branch(sentence, i_t_i)

    # purpose: tag a sentence of tokens with dependency tags, using an ad hoc likelihood ascention process
    # arguments: 
    # - tvecs: a list (sentence) of dependency-predicting arrays to be sampled for tags
    # prereqs: a list of token-aggregated probabilistic vectors
    # output: a tuple of sups, deps, infs containing a lists of ints, strs, and lists of ints, encoding suprema, dependence-types, and subesquent dependences
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

    # purpose: decode a sequence of probabilistic vectors for their representation of end-of-token status
    # arguments: 
    # - vecs: a list of probabilistic arrays of floats (for whatevers)
    # - decode_method: either 'viterbi' or other, defaulting the arg-max prediction
    # prereqs: not user-driven, operated by .interpret over a stream of prediction vectors (vecs) and an intended decode method
    # output: a list of boolean values indicating the True/False sampling of the prediction
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

    # purpose: decode a sequence of probabilistic vectors for their representation of end-of-sentence status
    # arguments: 
    # - tvecs: a list of probabilistic array of floats (for tokens)
    # - twis: a list (for each token) of character indices of subsumed whatevers
    # prereqs: not user-driven, operated by .interpret over a stream of prediction vectors (vecs) and an intended decode method
    # output: a list of boolean values indicating the sampling of the prediction   
    def decode_eoss(self, tvecs, twis):
        # decoding the sentence segmentation
        tokens = []; teoss = []
        for ti in range(len(tvecs)):
            tokens.append(ti)
            teoss.append(self.output(tvecs[ti], 'eos')[0][0] if 'eos' in self._ltypes else True)
        if (not teoss[-1]) and (len(tokens) >= self._max_sent) and self._max_sent: #  and predict_tags
                teoss[max([(self.output(tvecs[tix], 'eos')[1][(True, 'eos')], tix) for tix in tokens])[1]] = True
        eoss = []
        for ti, teos in enumerate(teoss):
            eoss.extend([teos]*len(twis[ti]))
        if not eoss[-1]: eoss[-1] = True
        return eoss
    
    # purpose: decode a sequence of probabilistic vectors for their representations by unbound token tags
    # arguments: 
    # - tvecs: a list of probabilistic array of floats (for tokens)
    # - twis: a list (for each token) of character indices of subsumed whatevers
    # - decode_method: either 'viterbi' or other, defaulting the arg-max prediction
    # prereqs: not user-driven, operated by .interpret over a stream of prediction vectors (vecs) and an intended decode method
    # output: a list of boolean values indicating the sampling of the prediction      
    def decode_tags(self, tvecs, twis, ltype, decode_method = 'viterbi'):
        # viterbi decode pos tags
        if decode_method == 'viterbi':
            ttags = self.viterbi(tvecs, ltype)
        # argmax decode pos tags
        if decode_method == 'argmax':
            ttags = [list(self.output(vec, ltype)[1].most_common(1))[0][0][0] for vec in tvecs]
        tags = []
        for ti in range(len(ttags)):
            tags.extend([ttags[ti]]*len(twis[ti]))
        return tags

    # purpose: decode a sequence of probabilistic vectors for their representation of parse-tagging information
    # arguments: 
    # - tvecs: a list of probabilistic array of floats (for tokens)
    # - twis: a list (for each token) of character indices of subsumed whatevers
    # - eoss: a list of boolean values indicating whether the position marks the end of a sentence
    # prereqs: not user-driven, operated by .interpret over a stream of prediction vectors (vecs) and an intended decode method
    # output: a list of boolean values indicating the sampling of the prediction      
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

    # purpose: decode a sequence of probabilistic vectors for their representation of sentence type information
    # arguments: 
    # - svecs: a list of probabilistic array of floats (for sentences)
    # - stis: a list (for each sentences) of character indices of subsumed tokens
    # - twis: a list (for each token) of character indices of subsumed whatevers
    # prereqs: not user-driven, operated by .interpret over a stream of prediction vectors (vecs) and an intended decode method
    # output: a list of boolean values indicating the sampling of the prediction      
    def decode_stys(self, svecs, stis, twis):
        sstys = []; stys = []
        for svec in svecs:
            sstys.append(list(self.output(svec, 'sty')[1].most_common(1))[0][0][0])
        for si in range(len(svecs)):
            stys.extend([sstys[si]]*sum([len(twis[ti]) for ti in stis[si]]))
        return stys

    # purpose: determine predictions for user-specified layers over a corpus of documents
    # arguments: 
    # - docs: see .fit
    # - covering: see .fit, enoforces token-level predictions over a gold covering
    # - all_layers: see .fit
    # - seed: controls randomization for all numpy processes
    # - predict_tags: bool, if False system will not sample tags from prediction vectors
    # - predict_contexts: bool, if True system will predict context vectors, instead of LM's (vocabulary-prediction vectors)
    # - dense_predict: bool, if True system will use dense contexts (integrating first-order predictions) for prediction of all tags (experimental)
    # prereqs: a trained model, provided by the .fit method
    # output: NA, system will store tags and other predictables within the ._documents object as per the .grok method
    def interpret(self, docs, covering = [], all_layers = defaultdict(lambda : defaultdict(list)),
                  seed = None, predict_tags = True, predict_contexts = False, dense_predict = False):
        streams, docs = self.encode_streams(docs, covering = covering, all_layers = all_layers)
        self._documents = []; self._sentences = []; self._tokens = []; self._whatevers = []
        if seed is not None:
            np.random.seed(seed)
        print("Interpreting documents...")
        for d_i, (stream, doc) in tqdm(list(enumerate(zip(streams, docs)))):
            docstream  = unroll(doc); self._w_set = set(); self._ix = 0; vecs = [] # ; cvecs = []
            eots = list(map(lambda x: x[0], map(self._vocs['eot'].decode, map(int, stream['eot'])))) if 'eot' in stream else []
            eoss = list(map(lambda x: x[0], map(self._vocs['eos'].decode, map(int, stream['eos'])))) if 'eos' in stream else []
            eods = list(map(lambda x: x[0], map(self._vocs['eod'].decode, map(int, stream['eod'])))) if 'eod' in stream else []
            # if a fine-tuned layer exists, use it to attend the prediction vectors
            if self._fine_tuned or dense_predict: 
                vecs, _, _ = self.R2(stream, dense_predict = dense_predict)
            else:
                vecs = self.R1(stream)
            atns = detach(stream['amp'])
            # atns = detach(atns)
            # determine the token segmentation, if necessary
            if not eots: # eot == True means the whatever is a singleton itself; others are compound
                eots = self.decode_eots(vecs) if predict_tags else [True] * len(vecs)
            tvecs, tnrms, tatns, twis = agg_vecs(vecs, atns, eots)
            # determine the sentence segmentation, if necessary
            if not eoss:
                eoss = self.decode_eoss(tvecs, twis) if predict_tags else [([False] * len(s))[-1] + [True] for s in doc]
            teoss = [eos for eot, eos in zip(eots, eoss) if eot]
            svecs, snrms, satns, stis = agg_vecs(tvecs, tatns, teoss, tnrms)
            # determine the document segmentation, if necessary
            if not eods:
                eods = [False for _ in range(len(docstream))]; eods[-1] = True
            seods = [eod for eot, eos, eod in zip(eots, eoss, eods) if (eos and eot)]
            dvecs, dnrms, datns, dsis = agg_vecs(svecs, satns, seods, snrms)
            # determine the part of speech tags, if necessary
            poss = [None for _ in range(len(vecs))]
            if ('pos' in self._lorder) and ('pos' not in all_layers) and predict_tags:
                poss = self.decode_tags(tvecs, twis, 'pos')
            # determine the lemma tags, if necessary
            lems = [None for _ in range(len(vecs))]
            if ('lem' in self._lorder) and ('lem' not in all_layers) and predict_tags:
                lems = self.decode_tags(tvecs, twis, 'lem')
            # determine the parse tags, if necessary
            sups, deps, infss = [None for _ in range(len(vecs))], [None for _ in range(len(vecs))], [[] for _ in range(len(vecs))]
            if ('sup' in self._lorder) and ('sup' not in all_layers[d_i]) and predict_tags:
                sups, deps, infss = self.decode_parse(tvecs, twis, eoss)
            # determine the sentence type tags, if necessary
            stys = [None for _ in range(len(vecs))]
            if ('sty' in self._lorder) and ('sty' not in all_layers) and predict_tags:
                stys = self.decode_stys(svecs, stis, twis)
            # feed the tags and data into interaction the framework 
            wi = 0
            all_ixs = [list(range(len(docstream))),
                       [ti for ti, wis in enumerate(twis) for _ in wis],
                       [si for si, tis in enumerate(stis) for ti in tis for _ in twis[ti]],
                       [di for di, sis in enumerate(dsis) for si in sis for ti in stis[si] for _ in twis[ti]]]
            all_vecs = (vecs, tvecs, svecs, dvecs); all_atns = (atns, tatns, satns, datns)
            all_nrms = ([1 for _ in range(len(vecs))], tnrms, snrms, dnrms)
            for w, eot, eos, eod, pos, lem, sup, dep, infs, sty in zip(docstream, eots, eoss, eods, poss, lems, sups, deps, infss, stys):
                tags = {'pos': pos, 'lem': lem, 'sup': sup, 'dep': dep, 'infs': infs, 'sty': sty}
                self.grok(wi, w, eot, eos, eod, w in self._w_set, atns[wi], vecs[wi], seed, 
                          all_vecs, all_atns, all_nrms, all_ixs, tags = tags)
                wi += 1

    # purpose: determine predictions for the forms of a corpus of documents
    # arguments: 
    # - docs: see .fit
    # - covering: see .fit, enoforces token-level predictions over a gold covering
    # - verbose: bool, with True indicating that information about the predictions will be printed to STDOUT
    # - return_output: bool, whereupon the function will return a list (corpus) of lists (documents) of prediction probabilities for the corpus's target whatevers
    # - dense_predict: bool, whereupon the function with utilize dense context vectors on the first-order model's parameters for prediction (experimental)
    # prereqs: a model trained by .fit
    # output: if return_output == True: then a list of predictions of prediction is returned, otherwise NA
    def stencil(self, docs, covering = [], verbose = True, return_output = False, dense_predict = False):
        streams, docs = self.encode_streams(docs, covering = covering); ps = []
        for d_i in tqdm(range(len(streams))):
            stream = streams[d_i]; doc = unroll(docs[d_i])
            if self._fine_tuned or dense_predict:
                vecs, _, _ = self.R2(stream, ltypes = ['form'], forward = False, dense_predict = dense_predict)
            else:
                # get the vectors and attention weights for the sentence
                vecs = self.R1(stream, ltypes = ['form'], forward = False)
            ps.append([]) 
            for wi, vec in enumerate(vecs):
                p = vec[stream['form'][wi]]
                ps[-1].append([p, p if self._vocs['form'].decode(int(stream['form'][wi]))[0] != ' ' else None])
            print("document no., pct. complete, 1/<P>, 1/<P> (nsp), M, M (nsp): ", d_i+1, 100*round((d_i+1)/len(docs), 2), 
                  round(1/(10**(np.log10([p[0] for p in ps[-1]]).mean())), 2), 
                  round(1/(10**(np.log10([p[1] for p in ps[-1] if p[1] is not None]).mean())), 2),
                  len(ps[-1]), len([p[1] for p in ps[-1] if p[1] is not None]))
        if return_output: return ps

    # purpose: determine predictions for the forms of a system-derived corpus to-be-determined
    # arguments: 
    # - m: int, at least one equal to the number of whatevers to generate
    # - prompt: str, to be tokenized indicating the context horizion from whose contexts generation is based
    # - docs: see .fit
    # - Td: Counter, of document's by-whatever integer frequenices
    # - revise: list of character indices, representing a span of intersecting whatevers to re-predict (experimental)
    # - top: float, or int, determining sampling method for generation. if int, sample selects from the integer list up to that point, and if float, samples up to that (number's) cumulative probability
    # - covering: see .fit
    # - seed: int, see stencil
    # - verbose: bool, see stencil
    # - return_output: bool, see .stencil
    # - dense_predict: bool, see .interpret
    # prereqs: a trained model (see .fit)
    # output: if return_output == True: then a list of predictions of prediction probabilities is returned, otherwise NA
    def generate(self, m = 1, prompt = "", docs = [], Td = Counter(), revise = [], top = 1., covering = [], 
                 seed = None, verbose = True, return_output = False, dense_predict = False):
        test_ps = [[]]; d_i, w_i = 0, 0
        ws = []; streams = []; vecs = []; avecs = []; cvecs = [];
        if docs:
            streams, docs = self.encode_streams(docs, covering = covering)
            eots = [list(map(lambda x: x[0],map(self._vocs['eot'].decode, map(int, stream['eot'])))) 
                    for stream in streams] if covering else []
            doc = unroll(docs[d_i]); tok_ps = []
            print("Evaluating language model..")
            pbar = tqdm(total=len(unroll(docs[d_i])))
        if seed is not None:
            np.random.seed(seed)
        if prompt and not docs:
            streams, prompt_docs = self.encode_streams([list(self.tokenize(prompt))])
            ws = prompt_docs[0]
        stream = streams[0] if streams else {'amps': [], 'cons': [], 'm0ps': [], 'amp': self.double([])}
        output = []; sampled = 0; talking = True; wis = []
        if revise and not docs:
            numchars = np.cumsum([len(w) for w in ws])
            wis = [wi for wi in range(len(ws)) 
                   if ((revise[0] <= numchars[wi] - 1 < revise[1]) or 
                       (revise[0] <= numchars[wi] - len(ws[wi]) < revise[1]))]
        if ws: 
            Td += Counter(ws)
            if verbose: 
                if wis:
                    print("".join(ws[:wis[0]]), end = '')
                else:        
                    print("".join(ws), end = '')
        while talking:
            if revise and not docs:
                wi = wis.pop(0); ws[wi] = ""
                if not wis: talking = False
            else:                
                ws = list(ws) + [""] 
                wi = len(ws) - 1
            if not docs:
                stream['amps'].append(self.double([0.])); stream['cons'].append(self.long([0])); stream['m0ps'].append(self.long([0]))
                stream['amp'] = self.double(list(self.get_amps(ws[:wi])) + [0.])
                for ci in range(max([0,wi - self._m]), wi):
                    r = ci - wi; c = (ws[ci], r, 'form')
                    _, encs, impact = self.encode(('', 'form'), c)
                    cons, amps, m0ps = [], [], []
                    for enc in encs:
                        cons.append(self._con_vocs[self._cltype].encode(enc))
                        amps.append(impact/len(encs))
                        m0ps.append(int(r/abs(r) if r else r))
                    stream['cons'][wi] = self.long(cons)
                    stream['amps'][wi] = self.double(amps)
                    stream['m0ps'][wi] = self.long(m0ps)
            if self._fine_tuned or dense_predict: 
                vecs, avecs, cvecs = self.R2(stream, forward = False, ltypes = ['form'],
                                             previous_avecs = avecs, previous_cvecs = cvecs, 
                                             previous_vecs = vecs, dense_predict = dense_predict)
            else:
                # get the vectors and attention weights for the sentence
                vecs = self.R1(stream, forward = False, ltypes = ['form'], previous_vecs = avecs)
            P = Counter(dict(zip(list(self._vocs['form']), detach(vecs[wi]))))
            
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
            # replace the last/empty element with the sampled (or stenciled) token
            ws[wi] = doc[wi] if docs else what
            # this is where we build up the stream if not stenciling
            if not docs:
                for ci in range(max([0,wi - self._m]), wi + 1):
                    r = wi - ci; c = (ws[wi], r, 'form')
                    _, encs, impact = self.encode(('', 'form'), c)
                    cons, amps, m0ps = [], [], []
                    for enc in encs:
                        cons.append(self._con_vocs[self._cltype].encode(enc))
                        amps.append(impact/len(encs))
                        m0ps.append(int(r/abs(r) if r else r))
                    stream['cons'][wi] = self.long(cons)
                    stream['amps'][wi] = self.double(amps)
                    stream['m0ps'][wi] = self.long(m0ps)
            # update the process with the sampled type
            sampled += 1; Td[ws[wi]] += 1
            if return_output: output.append([what, P])
            # gather stenciling information
            if docs:
                w = doc[w_i]
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
                w_i += 1; pbar.update(1)
                if w_i == len(doc): # update the document number and clear the model data
                    pbar.close()
                    print("document no., pct. complete, 1/<P>, 1/<P> (nsp), M, M (nsp): ", d_i+1, 100*round((d_i+1)/len(docs), 2), 
                          round(1/(10**(np.log10([test_p[0] for test_p in test_ps[-1]]).mean())), 2), 
                          round(1/(10**(np.log10([test_p[1] for test_p in test_ps[-1] if test_p[1] is not None]).mean())), 2),
                          len(test_ps[-1]), len([test_p[1] for test_p in test_ps[-1] if test_p[1] is not None]))
                    w_i = 0; d_i += 1
                    ws = []; vecs = []; avecs = []; cvecs = []
                    if d_i == len(docs):
                        talking = False
                    else:
                        doc = unroll(docs[d_i]); stream = streams[d_i]
                        pbar = tqdm(total=len(doc))
            if verbose and not docs:
                print(what, end = '')
            if docs and not w_i: test_ps.append([])
            if sampled == m and not (docs or revise): talking = False
        if revise and not docs:
            if verbose and (wi+1 < len(ws)): print("".join(ws[wi+1:]), end = '')
        if verbose and not docs: print('\n', end = '')
        if return_output and docs: return test_ps