import re
import scipy as sp
import numpy as np
import scipy.sparse
from collections import Counter, defaultdict

def get_context(i, sentence, m = 20):
    radii = range(-m,m+1); locs = range(i-m,i+m+1)
    components = [(ra, lo) for ra, lo in zip(radii, locs) 
                  if lo >= 0 and lo < len(sentence)] # and sentence[lo]
    if components:
        radii, locs = zip(*components)
        components = [np.concatenate([[sentence[lo] for lo in locs]])]
        weights = np.array(list(radii))
        components.append(weights)
        components.append(['form']*len(locs))
    return(list(zip(*components)))

def wave_index(stream, m = 0, seed = 0):
    s = list(stream)
    if seed:
        ra.seed(seed)
        ra.shuffle(s)
    M = len(s) + m
    indices = defaultdict(list)
    T = np.arange(M)
    f = Counter()
    Nts = []
    for i, w in enumerate(s):
        f[w] += 1
        indices[w].append(i + m)
        Nts.append(len(f))
    for w in indices:
        indices[w] = np.array(indices[w])
    indices[-1] = np.array(Nts)
    f[''] = 1; M += 1
    return indices, f, M, T

def get_wave(w, idxs, f, M, T, accumulating = True, backward = False):
    locs = np.zeros(len(T))
    locs[idxs[w]] = 1
    fts = np.cumsum(locs)
    Mts = T+1
    Nts = idxs[-1]
    if accumulating:
        wave = np.zeros(len(T))
        prev_loc = -1
        for Mt, ft, loc in zip(Mts[idxs[w]], fts[idxs[w]], idxs[w]):
            if backward: loc = 0
            delta = loc - prev_loc
            alps = 2*np.pi*(T[loc:]*ft/Mt)
            bet = 2*np.pi*((loc + 1)/(Mt/ft) - np.floor((loc + 1)/(Mt/ft)))
            wave[loc:] += np.cos(alps + bet)*delta
            prev_loc = loc
            
        if backward:
            wave /= (sum(Nts)/len(f)) # maybe len(T), and not len(f)? nvm
        else:
            wave /= np.cumsum(Nts)
    else: 
        alpha = 2*np.pi*(T*f.get(w, f[''])/M)
        betas = 2*np.pi*((idxs[w]+1)/(M/f.get(w, f[''])) - np.floor((idxs[w]+1)/(M/f.get(w, f['']))))
        if backward:
            wave = np.cos(alpha)*sum(np.cos(betas)) - np.sin(alpha)*sum(np.sin(betas))
            wave /= len(f) # maybe len(T), and not len(f)? nvm
        else:
            deltas = np.array([idxs[w][0] + 1]+list(idxs[w][1:] - idxs[w][:-1]))
            wave = np.zeros(len(T)) 
            prev_loc = -1
            for idxi, loc in enumerate(idxs[w]):
                if prev_loc >= 0:
                    bets = betas[:idxi]
                    delts = deltas[-idxi:]
                    wave[prev_loc:loc] = (np.cos(alpha[prev_loc:loc])*sum(np.cos(bets)*delts) - 
                                          np.sin(alpha[prev_loc:loc])*sum(np.sin(bets)*delts))
                prev_loc = loc
            wave[prev_loc:] = (np.cos(alpha[prev_loc:])*sum(np.cos(betas)*deltas) - 
                               np.sin(alpha[prev_loc:])*sum(np.sin(betas)*deltas))
            wave /= (len(f)*Mts) # maybe len(T), and not len(f)? nvm

    return wave/(f.get(w, f[''])*M)

