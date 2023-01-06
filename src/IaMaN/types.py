import numpy as np
from collections import Counter
from ..utils.stat import noise

# purpose: for when you can really just go for an object, even if you don't know why
# arguments: none
# prereqs: none
# output: an object, nothing more
class Dummy(object):
    pass

# purpose: principal object type for any string, i.e., inherited by all others
# arguments: see __init__()
# prereqs:
# use methods: 
# - __init__: initialize whatever with specified attributes
# properties:
# - position and extent: whatevers and inheritors can be compared for position in a document, i.e., numerical values assigned to the position of the whatever's first character, and likewise to the number of characters that they contain, allowing containment-based comparisons, too
# use attributes:
# - _form: string value of the whatever
# - _ix: int, indicating the first character of the whatever within the document
# - _sep: bool, indicating the status of the whatever as a separator for the next scale
# - _nov: bool, indicating the status of the whatever as a novel type/form in its sequence (document)
# - _nrm: float, indicating a positive value weighting the intensity of the whatever, for use in aggregation
# - _atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
# - _vec: array of floats, representing the (possibly predicted) identity (including possibly for tags) of the whatever
# output: an 'atomic' object within a document
class Whatever:
    # purpose: initialize whatever with specified attributes
    # arguments:
    # - form: string value of the whatever
    # - ix: int, indicating the first character of the whatever within the document
    # - sep: bool, indicating the status of the whatever as a separator for the next scale (token)
    # - nov: bool, indicating the status of the whatever as a novel type/form in its sequence (document)
    # - nrm: float, indicating a positive value weighting the intensity of the whatever, for use in aggregation
    # - atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
    # - vec: array of floats, representing the (possibly predicted) identity (including possibly for tags) of the whatever
    # prereqs: minimally, whatevers have immutble form (string value)
    # output: an initialized whatever
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
        
# purpose: object type for whatever any token is supposed to be
# arguments: see __init__()
# use attributes: for lower-level corporeal objects see class: Whatever
# - _form: string value of the token
# - _ix: int, indicating the first character of the token within the document
# - _sep: bool, indicating the status of the token as a separator for the next scale (sentence)
# - _nrm: float, indicating a positive value weighting the intensity of the whatevers within the token, for use in aggregation
# - _atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
# - _vec: array of floats, representing the (possibly predicted) aggregate identity (including possibly for tags) of token constituents (whatevers)
# - _lem: str, indicating the defined lemma tag for the token
# - _sen: str, indicating the defined sense tag for the token
# - _pos: str, indicating the defined part of speech tag for the token
# - _ent: str, indicating the defined named entitiy tag for the token
# - _dep: str, indicating the type of dependence that the token has on its parent
# - _sup: int, indicating by signed radius the parent token of the instance
# - _infs: list (descendents) of ints (signed radii) indicating those tokens dependent on the instnace
# - _whatevers: list of Whatever class objects subsumed by the Token
# prereqs: a list of atomic whatever class objects
# output: a smallest-class compound object within a document, likewise that available for user control
class Token(Whatever):
    # purpose: initialize token with specified attributes
    # arguments:
    # - whatevers: list of Whatever class objects, subsumed by the Token to be cast
    # - ix: int, indicating the first character of the token within the document
    # - sep: bool, indicating the status of the token as a separator for the next scale (sentence)
    # - nrm: float, indicating a positive value weighting the intensity of the whatevers within the token, for use in aggregation
    # - atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
    # - vec: array of floats, representing the (possibly predicted) aggregate identity (including possibly for tags) of token constituents (whatevers)
    # - lem: str, indicating the defined lemma tag for the token
    # - sen: str, indicating the defined sense tag for the token
    # - pos: str, indicating the defined part of speech tag for the token
    # - ent: str, indicating the defined named entitiy tag for the token
    # - dep: str, indicating the type of dependence that the token has on its parent
    # - sup: int, indicating by signed radius the parent token of the instance
    # - infs: list (descendents) of ints (signed radii) indicating those tokens dependent on the instnace
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

# purpose: object type for compounds and hierarchies of whatever tokens are
# arguments: see __init__()
# use attributes: for lower-level corporeal objects see each other class: Whatever and Token
# - _form: string value of the sentence
# - _ix: int, indicating the first character of the sentence within the document
# - _sep: bool, indicating the status of the sentence as a separator for the next scale (document)
# - _nrm: float, indicating a positive value weighting the intensity of the tokens within the sentence, for use in aggregation
# - _atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
# - _vec: array of floats, representing the (possibly predicted) aggregate identity (including possibly for tags) of sentence constituents (tokens)
# - _sty: str, indicating the type of the sentence
# - _tokens: list of Token class objects subsumed by the Sentence
# use methods: 
# - yield_branch: iterate a generator of dependent tokens for any given token within the sentence
# prereqs: a list of (possibly) compound token class objects
# output: an object, consisting as a compound of tokens, possibly containing internal structure
class Sentence(Token):
    # purpose: initialize sentence with specified attributes
    # arguments:
    # - tokens: list of Token class objects, subsumed by the Sentence to be cast
    # - ix: int, indicating the first character of the sentence within the document
    # - sep: bool, indicating the status of the sentence as a separator for the next scale (document)
    # - nrm: float, indicating a positive value weighting the intensity of the tokens within the sentence, for use in aggregation
    # - atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
    # - vec: array of floats, representing the (possibly predicted) aggregate identity (including possibly for tags) of token constituents
    # - sty: str, indicating the type of the sentence
    def __init__(self, tokens, ix = None, sep = None, nrm = None, atn = None, vec = None, sty = None):
        super(Sentence, self).__init__([w for t in tokens for w in t._whatevers], ix = ix, sep = sep, 
                                       nrm = nrm, atn = atn, vec = vec) 
        self._tokens = list(tokens)
        self._sty = str(sty) if sty is not None else None
    # purpose: iterate a generator of dependent tokens for any given token within the sentence
    # arguments: t_i: token location who's dependent tokens should be yielded
    # prereqs: a heirarchy of tokens
    # output: recursive generation of dependent tokens from a point within the sentence---the 'sub-branch'
    def yield_branch(self, t_i):
        yield t_i
        for i_t_i in self._tokens[t_i]._infs:
            yield from self.yield_branch(i_t_i)

# purpose: object type for compounds of sentences of tokens
# arguments: see __init__()
# use attributes: for lower-level corporeal objects see each other class: Whatever, Token, and Sentence
# - _form: string value of the document
# - _ix: int, indicating the first character of the document
# - _nrm: float, indicating a positive value weighting the intensity of the sentences within the document, for use in aggregation
# - _atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
# - _vec: array of floats, representing the (possibly predicted) aggregate identity (including possibly for tags) of document constituents (sentences)
# - _sentences: list of Sentence class objects subsumed by the Document
# prereqs: a list of (possibly) compound sentence class objects
# output: an object, consisting as a compound of sentences
class Document(Sentence):
    # purpose: initialize document with specified attributes
    # arguments:
    # - sentences: list of Sentence class objects, subsumed by the Document to be cast
    # - ix: int, indicating the first character of the document
    # - nrm: float, indicating a positive value weighting the intensity of the sentences within the document, for use in aggregation
    # - atn: float, indicating a numerical value from which the positive (_nrm) weighting is derived and used for aggregation
    # - vec: array of floats, representing the (possibly predicted) aggregate identity (including possibly for tags) of sentence constituents
    def __init__(self, sentences, ix = None, nrm = None, atn = None, vec = None):
        super(Document, self).__init__([t for s in sentences for t in s._tokens], ix = ix, 
                                       nrm = nrm, atn = atn, vec = vec)
        self._sentences = list(sentences)
        
# purpose: handles mapping to and from whatever to vocab indices. as the encoding function, it could be more descriptive to call this `class Code:`
# arguments: see __init__()
# use attributes: none
# use methods:
# - encode: map string to index
# - decode: map index to string
# properties:
# - size: vocabularies have enumerated size of representation as evinced by the len() function
# - iteration: vocabularies have consitituents, which can be enumerated with syntax iteration
# - order: vocabularies are aligned to a non-negative integer ordering, startign from zero
# prereqs: a list of whatevers, those of which are unique will form the vocabulary
# output: an object, mapping of an indexed set of strings
class Vocab: 
    # purpose: initialize vocabulary
    # arguments:
    # - whatevers: list of immutables (whatevers)
    # - null: immutable representing non-data (nothing)
    def __init__(self, whatevers, null = None):
        self._type_index = {}; self._index_type = {}
        self.train(whatevers)
        self._null = null
    # purpose: operate the initialization of the vocabulary (not user driven)
    # arguments:
    # - whatevers: list of immutables (whatevers)
    def train(self, whatevers):
        # generate vocab list
        for whatever in whatevers:
            if whatever not in self._type_index:
                self._type_index[whatever] = len(self._type_index)
        # build reverse lookup
        self._index_type = {i: whatever for whatever, i in self._type_index.items()}
    # purpose: map whatever to its index
    # arguments:
    # - whatever: immutables (whatever) to be mapped to an index
    # output: integer index representing the unique type
    def encode(self, whatever):
        # return self._type_index.get(whatever, 0)
        return self._type_index.get(whatever, 0 if self._null is None else
                                    self._type_index[(self._null, whatever[1], whatever[2])])
    # purpose: map index to its whatever
    # arguments:
    # - i: int (index) to be mapped to a whatever
    # output: string representing the index's str of representation
    def decode(self, i):
        return self._index_type.get(i, self._index_type[0])
    def __iter__(self):
        return (self._index_type[i] for i in self._index_type)
    def __len__(self):
        return len(self._type_index)
    def __getitem__(self, i):
        return self._index_type[i]

# purpose: generalize 1-hot encoding with low-dimensional, invertible bit-vector assignment
# arguments: see __init__()
# use attributes: none
# use methods:
# - encipher: map str to a 1-normed bit vector
# - decipher: map 1-normalized bit vector to str
# - sparse_encipher: map str to a list of indicies representing active bits in a 1-normed bit vector
# prereqs: a vocabulary (set of strings) to which vectors will be assigned, ideally in order of need for distinguishability
# output: an object of class: Cipher, which entails capability for dimensionality reduction and indexing
class Cipher:
    # purpose: initialize a cipher
    # arguments:
    # - vocab: list of immutables (whatevers), ideally in order of need for distinguishability
    # - bits: number of bits to use in (dimensionality of) representation. note: cannot be lower than roughly log2 of the vocabulary's size
    def __init__(self, vocab, bits = None):
        self._vocab = vocab
        self._bits = (int(np.log2(len(self._vocab)) + 1) if bits is None else 
                      (int(bits) if 2**int(bits) > len(self._vocab) else 
                       int(np.log2(len(self._vocab)) + 1)))
        self._basis = list(np.eye(self._bits))
        self._type_index = {}; self._index_type = {}; self._vs = [[np.zeros(self._bits)], []]
        self._sparse_index = {}; self._sparse_types = {}
        self._U = np.zeros((len(self._vocab), self._bits))
        self._V = np.zeros((len(self._vocab), self._bits))
        self._i = 0; self._j = 0; self._k = len(self._vs) - 1
        self.train()
    # purpose: operate the initialization of the cipher (not user driven)    
    # arguments: none
    def train(self):
        for w in self._vocab:
            v, idx = self.update(w)
            self._sparse_types[idx] = w
            self._sparse_index[w] = idx
            self._V[self._type_index[w],:] = v/v.sum()
            self._U[self._type_index[w],:] = v
    # purpose: update the cipher's maps between whatevers and bit vectors (not user driven)
    # arguments: w: str value of the whatever
    # output: tuple of array (bit vector) and list (active bits) of ints (bit locations) being assigned to a given string
    def update(self, w):
        enciphered = False
        while not enciphered:
            v = np.abs(self._vs[self._k-1][self._j] - self._basis[self._i])
            idx = tuple(np.arange(v.shape[0])[v != 0])
            if len(idx) == self._k and idx not in self._sparse_types:
                self._type_index[w] = len(self._type_index)
                self._index_type[self._type_index[w]] = w
                self._vs[self._k].append(v)
                enciphered = True
            self._j += 1
            if self._j == len(self._vs[self._k-1]):
                self._j = 0; self._i += 1
                if self._i == len(self._basis):
                    if self._k == 1: self._basis.reverse()
                    self._i = 0; self._k += 1
                    self._vs[-1].reverse(); self._vs.append([])
        return v, idx
    # purpose: map str to a 1-normed bit vector
    # arguments: w: str value of the whatever
    # output: normalized bit vector representing the string
    def encipher(self, w):
        return np.array(self._V[self._type_index[w],:]) if w in self._type_index else np.array(self._vs[0][0])
    # purpose: map 1-normalized bit vector to str
    # arguments: v: 1-normalized vector to be mapped to a str
    # output: string represented by bit-vector most-closely approximated by the reference
    def decipher(self, v):
        return self._index_type.get(self._U.dot(v).argmax(), '')
    # purpose: map str to a list of indicies representing active bits in a 1-normed bit vector
    # arguments: w: str value of the whatever
    # output: array of indices that indices representing active bits in a 1-normed bit vector
    def sparse_encipher(self, w):
        return np.array(self._sparse_index[w]) if w in self._type_index else np.array([])
    # determines if whatever has a representation in the cipher
    def __contains__(self, whatever):
        return whatever in self._type_index
    
class Edgevec:
    def __init__(self, representation = lambda x: np.array([{'': 0.}.get(x, 1.)])):
        self.rep = representation
        self._bits = 1 if self.rep is None else (1 + self.rep._bits if hasattr(self.rep, '_bits') else 2)
    def __call__(self, x):
        return (np.array([{'': 1}.get(x, 0)]) if self.rep is None else 
                np.concatenate([np.array([{'': 1}.get(x, 0)]), self.rep(x)]))
    
class Basis:
    def __init__(self, vocabulary):
        self._bits = len(vocabulary)
        self.voc = vocabulary
    def __call__(self, x):
        vec = np.zeros(self._bits); vec[self.voc.encode(x)] = 1
        return vec
    
class Numvec:
    def __init__(self, max = 1, representation = None):
        self.rep = representation; self._max = max
        self._bits = 1 if self.rep is None else (1 + self.rep._bits if hasattr(self.rep, '_bits') else 2)
    def __call__(self, x):
        return (np.array([1/self._max if x == '' else x/self._max]) if self.rep is None else 
                np.concatenate([np.array([1/self._max if x == '' else x/self._max]), self.rep(x)]))
    
class Represent:
    def __init__(self, data = None, bits = 0, numerical = 0, noisy = True, btype = 'nf'):
        self._bits_set = int(bits); self._numerical = numerical; self._noisy = noisy
        if data is None:
            self._bits = 2 # + (1 if numerical else 0)
            self.rep = None if numerical else lambda x: np.array([{'': 0.}.get(x, 1.)])
            self.rep = Edgevec(Numvec(numerical, self.rep)) if numerical else Edgevec(self.rep)
        else:
            self._f =  Counter([t for d in data for t in d])
            if ('', 'oov') in self._f: 
                self._f[('', 'oov')] = len(self._f) - 1
            self._M = sum(self._f.values()); self._k = len(data)
            self._df = Counter([ti[0] for ti in Counter([(t,i) for i, d in enumerate(data) for t in d])])
            self._n = Counter([self._f[t] for t in self._f])
            self.build_cipher(self._bits_set, btype)
            
    def set_btype(self, btype):
        self._btype = btype
        if self._btype == 'df':
            self._beta = Counter({t: self._df[t]/self._k for t in self._df}) # switch to?: self._df[t]/(self._k + 1)
        elif self._btype == 'nf':
            self._beta = Counter({t: 1/self._n[self._f[t]] for t in self._f}) # switch to?: 1/(self._n[self._f[t]] + 1)
        elif self._btype == 'f': 
            self._beta = Counter({t: self._f[t]/(self._f[t] + 1) for t in self._f})
        elif self._btype == 'dfnf':
            self._beta = Counter({t: ((self._df[t]/self._k) * (1/self._n[self._f[t]])) ** (1/2) for t in self._f})
        elif self._btype == 'all':
            self._beta = Counter({t: ((self._df[t]/self._k) * (1/self._n[self._f[t]]) * (self._f[t]/(self._f[t] + 1))) ** (1/3)
                                  for t in self._f})
        else:
            self._beta = Counter()
        
    def update_cipher(self, data, btype = ''):
        self._f += Counter([t for d in data for t in d])
        if ('', 'oov') in self._f: 
            self._f[('', 'oov')] = len(self._f) - 1
        self._M = sum(self._f.values()); self._k += len(data)
        self._df += Counter([ti[0] for ti in Counter([(t,i) for i, d in enumerate(data) for t in d])])
        self._n = Counter([self._f[t] for t in self._f])
        self.set_btype(btype if btype else self._btype)
        self.build_cipher(self._bits_set, btype if btype else self._btype)
        
    def build_cipher(self, bits, btype):
        if bits > 0:
            self._bits = bits + 1 + (1 if self._numerical else 0)
            self.cipher = Cipher([t for t, _ in self._f.most_common()], bits = bits)
            self.vec = self.cipher.encipher
        else:
            self.vec = Basis(Vocab([t for t in set(self._f.values() if bits < 0 else self._f.keys())]))
            self._bits = self.vec._bits + 1 + (1 if self._numerical else 0)
        self._frep = sum([self.vec(self._f[t] if bits < 0 else t)*self._f[t] for t in self._f])
        self.set_btype(btype)
        if self._noisy:
            self.rep = lambda t: noise(self.vec(self._f[t] if bits < 0 else t), self._beta.get(t, 1), self._frep)
        else:
            self.rep = lambda t: self.vec(self._f[t] if bits < 0 else t)
        self.rep = Edgevec(Numvec(self._numerical, self.rep)) if self._numerical else Edgevec(self.rep)
        
    def __call__(self, t):
        return self.rep(t)