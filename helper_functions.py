import random
from collections import defaultdict
 
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer



# ----------------------------------------------------------------------------
# Stage 0: reply graph
# ----------------------------------------------------------------------------
def build_reply_graph(df, min_degree=3):
    """Weighted undirected user graph as dict: node -> {neighbor: weight}.
    Edge weight = number of replies between two users (either direction).
    Replies are taken regardless of subreddit, to avoid rediscovering subreddits."""
    author_of = dict(zip(df["id"], df["author"]))
    parent = df["parent_id"].str.replace("t1_", "", regex=False)
    pa = parent.map(author_of)
    edges = pd.DataFrame({"a": df["author"].values, "b": pa.values}).dropna()
    edges = edges[edges.a != edges.b]
    lo = np.where(edges.a < edges.b, edges.a, edges.b)
    hi = np.where(edges.a < edges.b, edges.b, edges.a)
    w = pd.DataFrame({"a": lo, "b": hi}).value_counts()
 
    adj = defaultdict(dict)
    for (a, b), c in w.items():
        adj[a][b] = float(c)
        adj[b][a] = float(c)
    adj = dict(adj)
    # iterative degree filter (a simple k-core)
    while True:
        drop = [n for n, nb in adj.items() if len(nb) < min_degree]
        if not drop:
            break
        for n in drop:
            for m in adj.pop(n, {}):
                if m in adj:
                    adj[m].pop(n, None)
    return adj


# ----------------------------------------------------------------------------
# Stage 1: Louvain (own implementation)
# Convention: A is symmetric; A[i][i] of an aggregated node = sum of internal A_ij
# (both directions), k_i = sum_j A_ij, m2 = sum_ij A_ij = 2m.
# ----------------------------------------------------------------------------
def modularity(adj, comm):
    m2 = sum(sum(nb.values()) for nb in adj.values())
    inner, tot = defaultdict(float), defaultdict(float)
    for i, nb in adj.items():
        for j, a in nb.items():
            if comm[i] == comm[j]:
                inner[comm[i]] += a
            tot[comm[i]] += a
    return sum(inner[c] / m2 - (tot[c] / m2) ** 2 for c in tot)
 
 
def _local_moving(adj, rng, eps=1e-9):
    k = {i: sum(nb.values()) for i, nb in adj.items()}
    m2 = sum(k.values())
    comm = {i: i for i in adj}
    tot = dict(k)
    order = list(adj)
    improved, moved = False, True
    while moved:
        moved = False
        rng.shuffle(order)
        for i in order:
            ci = comm[i]
            w = defaultdict(float)             # weight from i to each neighbour community
            for j, a in adj[i].items():
                if j != i:
                    w[comm[j]] += a
            tot[ci] -= k[i]                    # remove i from its community
            best, best_gain = ci, w.get(ci, 0.0) - tot[ci] * k[i] / m2
            for c, wc in w.items():            # gain ∝ k_i,in - tot_c * k_i / 2m
                g = wc - tot[c] * k[i] / m2
                if g > best_gain + eps:
                    best, best_gain = c, g
            tot[best] += k[i]
            if best != ci:
                comm[i] = best
                moved = improved = True
    return comm, improved
 
 
def _aggregate(adj, comm):
    new = defaultdict(lambda: defaultdict(float))
    for i, nb in adj.items():
        for j, a in nb.items():
            new[comm[i]][comm[j]] += a
    return {c: dict(nb) for c, nb in new.items()}
 
 
def louvain(adj, seed=0):
    """Returns a list of levels: (membership dict original_node -> community, modularity)."""
    rng = random.Random(seed)
    member = {n: n for n in adj}
    levels, cur = [], adj
    while True:
        comm, improved = _local_moving(cur, rng)
        if not improved:
            break
        member = {n: comm[member[n]] for n in member}
        levels.append((dict(member), modularity(adj, member)))
        cur = _aggregate(cur, comm)
    return levels
 

 # ----------------------------------------------------------------------------
# Stage 2: MinHash + LSH  (communities and subreddits share one signature matrix)
# ----------------------------------------------------------------------------
P = np.uint64(4294967311)  # prime just above 2^32; a, x < 2^32 so a*x+b fits in uint64
 
 
def minhash(set_idx, user_idx, n_sets, k=128, seed=0):
    """set_idx[i], user_idx[i] = one (set, member) pair (deduplicated).
    Returns signature matrix (k, n_sets); empty sets stay at P."""
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 2**32, size=k, dtype=np.uint64)
    B = rng.integers(0, 2**32, size=k, dtype=np.uint64)
    order = np.argsort(set_idx, kind="stable")
    s, u = set_idx[order], user_idx[order].astype(np.uint64)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    present = s[starts]
    sig = np.full((k, n_sets), P, dtype=np.uint64)
    for h in range(k):
        hv = (A[h] * u + B[h]) % P
        sig[h, present] = np.minimum.reduceat(hv, starts)
    return sig
 
 
def lsh_candidates(sig, n_comm, b, r, max_bucket=500):
    """Candidate (community, subreddit) pairs only; sets 0..n_comm-1 are communities."""
    assert b * r <= sig.shape[0]
    cands = set()
    for band in range(b):
        block = np.ascontiguousarray(sig[band * r:(band + 1) * r].T)
        buckets = defaultdict(lambda: ([], []))
        for idx in range(block.shape[0]):
            if block[idx, 0] == P:             # empty set
                continue
            side = 0 if idx < n_comm else 1
            buckets[block[idx].tobytes()][side].append(idx)
        for cs, ss in buckets.values():
            if cs and ss and len(cs) + len(ss) <= max_bucket:
                cands.update((c, s - n_comm) for c in cs for s in ss)
    return cands
 
 
def rank_subreddits(cands, comm_users, sub_users, n_users_total, min_cov=0.05):
    """Exact verification. comm_users/sub_users: lists of sorted unique int arrays.
    coverage = |C∩S|/|C| ; lift = coverage / (|S|/N)."""
    out = defaultdict(list)
    for c, s in cands:
        inter = np.intersect1d(comm_users[c], sub_users[s], assume_unique=True).size
        cov = inter / len(comm_users[c])
        if cov >= min_cov:
            jac = inter / (len(comm_users[c]) + len(sub_users[s]) - inter)
            lift = cov / (len(sub_users[s]) / n_users_total)
            out[c].append((s, cov, lift, jac))
    return {c: sorted(v, key=lambda t: -t[2]) for c, v in out.items()}