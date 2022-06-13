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