from sklearn.decomposition import TruncatedSVD
import numpy as np

def svdsub(X, d=50, state = 691):
    return TruncatedSVD(n_components=d, random_state = state).fit_transform(X)

def cbow(TDM, CoM):
    return TDM.T.dot(CoM).T

def softmax(z):
    expz = np.exp(z - np.max(z))
    return expz / sum(expz)

def rankguess(sizeranks, f):
    if f in sizeranks:
        return sizeranks[f]
    else:
        sizes = np.array(list(sizeranks.keys()))
        errs = np.abs(f - sizes)
        return sizeranks[sizes[errs == min(errs)][0]]