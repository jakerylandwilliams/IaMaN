import re
import scipy as sp
import numpy as np
import scipy.sparse
from collections import Counter

def tokenize(text, space = True, wordchars = "a-zA-Z0-9-'"):
    tokens = []
    for token in re.split("(["+wordchars+"]+)", text):
        if not space:
            token = re.sub("[ ]+", "", token)
        if not token:
            continue
        if re.search("["+wordchars+"]", token):
            tokens.append(token)
        else: 
            tokens.extend(token)
    return tokens

def sentokenize(text, space = True, delims = ".?!\n|\t:", sentchars = "a-zA-Z0-9-'"): # harmonize with hr-bpe
    sentences = []
    
    for chunk in re.split("(\s*(?<=["+delims+"][^"+sentchars+"])\s*)", text):
        if (len(chunk)==1 and not re.search("["+sentchars+"]", chunk[0])):
            if space or (chunk[0] != " "):
                if len(sentences):
                    sentences[-1] = sentences[-1] + [chunk]  
                else:
                    sentences.append([chunk])
        elif not re.search("["+sentchars+"]", chunk):
            tokens = tokenize(chunk, space = space)
            if len(sentences):
                sentences[-1] = sentences[-1] + tokens  
            else:
                sentences.append(tokens)
        else:
            sentences.append(tokenize(chunk, space = space))
    
    return sentences

def make_TDM(documents, do_tfidf = True, space = True, normalize = True):
    document_frequency = Counter()
    for j, document in enumerate(documents):
        frequency = Counter([t for s in sentokenize(document, space = space) 
                         for t in s])
        document_frequency += Counter(frequency.keys())
    type_index = {t:i for i, t in enumerate(sorted(list(document_frequency.keys())))}
    document_frequency = np.array(list(document_frequency.values()))
    # performs the counting again, and stores with standardized indexing`
    counts, row_ind, col_ind = map(np.array, zip(*[(count, type_index[t],j) 
                                                   for j, document in enumerate(documents) 
                                                   for t, count in Counter(tokenize(document, space = space)).items()]))
    # constructs a sparse TDM from the indexed counts
    TDM = sp.sparse.csr_matrix((counts, (row_ind, col_ind)),
                             shape = (len(document_frequency),len(documents)))
    if normalize:
        # normalize frequency to be probabilistic
        TDM = TDM.multiply(1/TDM.sum(axis = 0))
    # apply tf-idf
    if do_tfidf:
        num_docs = TDM.shape[1]
        IDF = -np.log2(document_frequency/num_docs)
        TDM = (TDM.T.multiply(IDF)).T
    return(TDM, type_index)

def get_context(i, sentence, m = 20, positional = True, normed = False):
    context = np.array(sentence)
    locs = np.array(range(len(sentence))) - i
    dists = (lambda x: x-x[i])(np.cumsum(list(map(len, sentence))))
    mask = (locs != 0) & (np.abs(locs) <= m) if m else (locs != 0)
    # context = context[mask]
    if positional:
        if normed:
            return list(zip(context[mask], dists[mask]))
        else:
            return list(zip(context[mask], locs[mask]))
    else:
        return context[mask]

def make_CoM(documents, k = 20, gamma = 0, space = True, do_tficf = True, normalize = True):
    document_frequency = Counter()
    for j, document in enumerate(documents):
        sentences = sentokenize(document, space = space)
        documents[j] = sentences
        frequency = Counter([t for s in documents[j] for t in s])
        document_frequency += Counter(frequency.keys())
    type_index = {t:i for i, t in enumerate(sorted(list(document_frequency.keys())))}

    co_counts = Counter()  
    for document in documents:
        for sentence in document:
            for i, ti in enumerate(sentence):
                context, weights = get_context(i, sentence, k = k, gamma = gamma)        
                for j, tj in enumerate(context):
                    co_counts[(type_index[ti], type_index[tj])] += weights[j]

    type_ijs, counts = zip(*co_counts.items())
    row_ind, col_ind = zip(*type_ijs)

    # constructs a sparse CoM from the indexed counts
    CoM = sp.sparse.csr_matrix((counts, (row_ind, col_ind)),
                              shape = (len(type_index),len(type_index)))
    if normalize:
        # normalize frequency to be probabilistic
        CoM = CoM.multiply(1/CoM.sum(axis = 0))
    
    # apply tf-icf
    if do_tficf:
        context_frequency = np.count_nonzero(CoM.toarray(), axis=1)
        num_cons = CoM.shape[1]
        ICF = -np.log2(context_frequency/num_cons)
        CoM = (CoM.T.multiply(ICF)).T
        
    return CoM, type_index