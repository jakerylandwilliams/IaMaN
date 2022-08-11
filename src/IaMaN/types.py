import numpy as np

class Dummy(object):
    pass

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
        
# I know these things are usually called vocabularies, i.e. we define `class Vocab:`,
class Vocab: # but since what they do is encode, wouldn't it be more descriptive to define them as `class Code:`?
    # Class that handles mapping to and from whatever to vocab indices
    def __init__(self, whatevers, null = None):
        self._type_index = {}; self._index_type = {}
        self.train(whatevers)
        self._null = null
        
    def train(self, whatevers):
        # generate vocab list
        for whatever in whatevers:
            if whatever not in self._type_index:
                self._type_index[whatever] = len(self._type_index)
        # build reverse lookup
        self._index_type = {i: whatever for whatever, i in self._type_index.items()}
        
    def encode(self, whatever):
        # return self._type_index.get(whatever, 0)
        return self._type_index.get(whatever, 0 if self._null is None else
                                    self._type_index[(self._null, whatever[1], whatever[2])])
    
    def decode(self, i):
        return self._index_type.get(i, self._index_type[0])
    
    def __iter__(self):
        return (self._index_type[i] for i in self._index_type)
    
    def __len__(self):
        return len(self._type_index)
    
    def __getitem__(self, i):
        return self._index_type[i]
    
# this differentes the concept of a Code (vocabs and 1-hot encodings), which are used for indexing types; 
# with a Cipher, which entails dimensionality reduction and compression (SVD; IFEncipherment, nee Encoding).
# the cool thing (technically), is that this Cipher generalizes to 1-hot encoding when bits = len(self._vocab) ;)
# does this mean that every Cipher is a Code? as conceived, a Cipher is 'fuzzy', while a Code is exact.
# note: we're probably stomping on disciplinary terminology, and better terms may exist.
class Cipher:
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
        
    def train(self):
        for w in self._vocab:
            v, idx = self.update(w)
            self._sparse_types[idx] = w
            self._sparse_index[w] = idx
            self._V[self._type_index[w],:] = v/v.sum()
            self._U[self._type_index[w],:] = v
                    
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

    def encipher(self, w):
        return np.array(self._V[self._type_index[w],:]) if w in self._type_index else np.array(self._vs[0][0])
    
    def decipher(self, v):
        return self._index_type.get(self._U.dot(v).argmax(), '')
    
    def sparse_encipher(self, w):
        return np.array(self._sparse_index[w]) if w in self._type_index else np.array([])
        