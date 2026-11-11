
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


# ----------------------------------------------------------------------------
# Stage 0: reply graph
# ----------------------------------------------------------------------------
def build_reply_graph(df, posts=None, min_degree=3, drop_authors=("[deleted]", "AutoModerator")):
    """Weighted undirected user graph as dict: node -> {neighbor: weight}.
    Edge weight = number of replies between two users (either direction)."""
    author_of = dict(zip(df["id"], df["author"]))
    if posts is not None:
        post_ids = posts["id"].astype(str).str.replace(r"^t3_", "", regex=True)
        author_of.update(zip(post_ids, posts["author"]))
    parent = df["parent_id"].astype(str).str.replace(r"^t[13]_", "", regex=True)
    pa = parent.map(author_of)
    edges = pd.DataFrame({"a": df["author"].values, "b": pa.values}).dropna()
    edges = edges[(edges.a != edges.b) & ~edges.a.isin(drop_authors) & ~edges.b.isin(drop_authors)]
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
# Stage 1: Louvain 
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
# Stage 2: MinHash + LSH  
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
 
 
# ----------------------------------------------------------------------------
# Stage 3: log-odds ratio with informative Dirichlet prior (Monroe et al. 2008)
# ----------------------------------------------------------------------------
def community_term_matrix(docs, min_count=20):
    """Counts per community; drops terms with fewer than min_count total occurrences
    (a corpus-wide count cutoff, NOT a document-frequency cutoff: there are few documents)."""
    cv = CountVectorizer(stop_words="english", token_pattern=r"[a-z]{3,}")
    Y = cv.fit_transform(docs).tocsr()
    words = np.array(cv.get_feature_names_out())
    keep = np.asarray(Y.sum(axis=0)).ravel() >= min_count
    return Y[:, np.flatnonzero(keep)], words[keep]
 
 
def log_odds_z(Y, alpha0=1000.0):
    """Returns dense (n_comm, n_terms) z-scores: community i vs. all other communities."""
    y_tot = np.asarray(Y.sum(axis=0)).ravel().astype(float)
    alpha = alpha0 * y_tot / y_tot.sum()                 # prior ∝ global frequency
    Z = np.zeros(Y.shape)
    for i in range(Y.shape[0]):
        y_i = Y.getrow(i).toarray().ravel().astype(float)
        y_r = y_tot - y_i
        n_i, n_r = y_i.sum(), y_r.sum()
        l_i = np.log((y_i + alpha) / (n_i + alpha0 - y_i - alpha))
        l_r = np.log((y_r + alpha) / (n_r + alpha0 - y_r - alpha))
        var = 1.0 / (y_i + alpha) + 1.0 / (y_r + alpha)
        Z[i] = (l_i - l_r) / np.sqrt(var)
    return Z
 
 
def top_terms(Z, words, i, n=10):
    return list(words[np.argsort(Z[i])[::-1][:n]])
 
 
def tfidf_top_terms(Y, words, i, n=10):
    """Plain TF.IDF baseline (Week 1 definition) for comparison."""
    Yd = Y.getrow(i).toarray().ravel().astype(float)
    tf = Yd / Yd.max()
    n_t = np.asarray((Y > 0).sum(axis=0)).ravel()
    idf = np.log2(Y.shape[0] / n_t)
    score = tf * idf
    top = np.argsort(score)[::-1][:n]
    return list(words[top[score[top] > 0]])      # drop zero-score filler
 
 
def npmi_coherence(top_words, B, vocab_index):
    """NPMI coherence of a word list. B: binary (docs x terms) csc matrix over a
    reference corpus (e.g. individual comments); vocab_index: word -> column."""
    N = B.shape[0]
    cols = [vocab_index[w] for w in top_words if w in vocab_index]
    df = np.asarray(B[:, cols].sum(axis=0)).ravel()
    co = (B[:, cols].T @ B[:, cols]).toarray()
    scores = []
    for a in range(len(cols)):
        for c in range(a + 1, len(cols)):
            p1, p2, p12 = df[a] / N, df[c] / N, co[a, c] / N
            if p12 == 0:
                scores.append(-1.0)
            else:
                scores.append(np.log(p12 / (p1 * p2)) / -np.log(p12))
    return float(np.mean(scores)) if scores else float("nan")
 
 
# ----------------------------------------------------------------------------
# Stage functions: each takes inputs and returns outputs, so people can own one
# stage each and re-run one stage without redoing the others.
# ----------------------------------------------------------------------------
 
 
def save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
 
 
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
def select_communities(levels, level=0, min_comm_size=30):
    """Pick one Louvain level and drop small communities.
    Returns user_comm: dict user -> community index 0..n_comm-1."""
    membership, _ = levels[min(level, len(levels) - 1)]
    sizes = pd.Series(membership).value_counts()
    keep = sizes[sizes >= min_comm_size].index.tolist()
    comm_id = {c: i for i, c in enumerate(keep)}
    return {u: comm_id[c] for u, c in membership.items() if c in comm_id}
 
 
def match_subreddits(df, user_comm, min_comments_per_sub=2, k=128, b=64, r=2, seed=0):
    """Stage 2: MinHash + LSH + exact verification.
    Returns dict(ranked, sub_names, comm_sizes, n_candidates, sig)."""
    n_comm = max(user_comm.values()) + 1
    graph_users = sorted(user_comm)
    uid = {u: i for i, u in enumerate(graph_users)}
    sub_names = sorted(df["subreddit"].unique())
    sid = {s: i for i, s in enumerate(sub_names)}
 
    d = df[df["author"].isin(uid)]
    cnt = d.groupby(["author", "subreddit"]).size().reset_index(name="n")
    cnt = cnt[cnt.n >= min_comments_per_sub]
 
    # one list of (set id, user index) pairs: communities first, then subreddits
    pairs_set = np.r_[[user_comm[u] for u in graph_users],
                      n_comm + cnt["subreddit"].map(sid).values]
    pairs_user = np.r_[np.arange(len(graph_users)), cnt["author"].map(uid).values]
    sig = minhash(pairs_set, pairs_user, n_comm + len(sub_names), k=k, seed=seed)
 
    comm_users = [np.array(sorted(uid[u] for u, c in user_comm.items() if c == i))
                  for i in range(n_comm)]
    sub_users = [np.sort(cnt.loc[cnt.subreddit == s, "author"].map(uid).values)
                 for s in sub_names]
    cands = lsh_candidates(sig, n_comm, b, r)
    ranked = rank_subreddits(cands, comm_users, sub_users, len(graph_users))
    return {"ranked": ranked, "sub_names": sub_names, "n_candidates": len(cands),
                "n_possible": n_comm * len(sub_names),
                "comm_sizes": [len(c) for c in comm_users], "sig": sig}
 
 
def label_words(df, user_comm, max_comments_per_user=50, min_count=20,
                alpha0=1000.0, seed=0):
    """Stage 3: one document per community, then log-odds z-scores.
    Returns dict(Z, Y, words). Comments per user are capped by a random sample."""
    n_comm = max(user_comm.values()) + 1
    d = df[df["author"].isin(user_comm)].sample(frac=1, random_state=seed)
    d = d.groupby("author", group_keys=False).head(max_comments_per_user)
    d = d.assign(comm=d["author"].map(user_comm))
    docs = d.groupby("comm")["body"].apply(" ".join).reindex(range(n_comm)).fillna("")
    Y, words = community_term_matrix(docs.tolist(), min_count=min_count)
    return {"Z": log_odds_z(Y, alpha0=alpha0), "Y": Y, "words": words}


def community_names(labels, n_words=3):
    """Name each community by its top log-odds words, e.g. 'python / error / pandas'."""
    return {c: " / ".join(top_terms(labels["Z"], labels["words"], c, n_words))
            for c in range(labels["Z"].shape[0])}
 
 
def report(matches, labels, top=5):
    """Print the combined description of every community."""
    print(f"LSH: {matches['n_candidates']} candidate pairs out of {matches['n_possible']} possible")
    for c, size in enumerate(matches["comm_sizes"]):
        print(f"\nCommunity {c}  ({size} users)")
        print("  subreddits (name, coverage, lift):",
              [(matches["sub_names"][s], round(cov, 2), round(lift, 1))
               for s, cov, lift, _ in matches["ranked"].get(c, [])[:top]])
        print("  log-odds words:", top_terms(labels["Z"], labels["words"], c))
        print("  tf-idf words  :", [w for w in tfidf_top_terms(labels["Y"], labels["words"], c)])
 
 
def run(df, posts=None, level=0, min_comm_size=30, cache_dir=None, seed=0,
        match_args=None, label_args=None):
    """Thin wrapper that chains the stages. With cache_dir, each stage's output is
    saved to disk and reloaded on the next call (delete the file to redo a stage)."""
    def cached(name, fn):
        path = os.path.join(cache_dir, name + ".pkl") if cache_dir else None
        if path and os.path.exists(path):
            return load(path)
        out = fn()
        if path:
            save(out, path)
        return out
 
    levels = cached("louvain_levels", lambda: louvain(build_reply_graph(df, posts), seed=seed))
    for i, (m, q) in enumerate(levels):
        print(f"Louvain level {i}: {len(set(m.values()))} communities, modularity {q:.3f}")
    user_comm = select_communities(levels, level, min_comm_size)
    matches = cached("matches", lambda: match_subreddits(df, user_comm, seed=seed, **(match_args or {})))
    labels = cached("labels", lambda: label_words(df, user_comm, seed=seed, **(label_args or {})))
    report(matches, labels)
    return {"levels": levels, "user_comm": user_comm, "matches": matches, "labels": labels}


# ----------------------------------------------------------------------------
# Evaluation: null model and seed stability 
# ----------------------------------------------------------------------------
def to_unweighted(adj):
    """Same graph with every edge weight set to 1 (and self-loops removed)."""
    return {n: {m: 1.0 for m in nb if m != n} for n, nb in adj.items()}
 
 
def rewire(adj, swaps_per_edge=10, seed=0):
    """Degree-preserving random rewiring (double edge swap): pick edges (a,b) and (c,d),
    replace them by (a,d) and (c,b) unless that makes a self-loop or a duplicate edge.
    Every node keeps its number of neighbours; the structure is randomized."""
    rng = random.Random(seed)
    nbrs = {n: set(nb) for n, nb in adj.items()}
    edges = [(a, b) for a, nb in adj.items() for b in nb if str(a) < str(b)]
    n_edges = len(edges)
    for _ in range(swaps_per_edge * n_edges):
        i, j = rng.randrange(n_edges), rng.randrange(n_edges)
        (a, b), (c, d) = edges[i], edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or d in nbrs[a] or b in nbrs[c]:
            continue
        nbrs[a].remove(b); nbrs[b].remove(a); nbrs[c].remove(d); nbrs[d].remove(c)
        nbrs[a].add(d); nbrs[d].add(a); nbrs[c].add(b); nbrs[b].add(c)
        edges[i], edges[j] = (a, d), (c, b)
    return {n: {m: 1.0 for m in s} for n, s in nbrs.items()}
 
 
def _level_q(levels, level):
    """Modularity at a level; level=-1 means the coarsest (last) level."""
    idx = len(levels) - 1 if level == -1 else min(level, len(levels) - 1)
    return levels[idx][1]
 
 
def null_model_test(adj, n_null=20, level=-1, seed=0):
    """Is the modularity of the real graph larger than in degree-preserving random graphs?
    Uses the unweighted graph for both, so the comparison is fair.
    Returns dict(real, null_mean, null_std, z, null_scores)."""
    G = to_unweighted(adj)
    real = _level_q(louvain(G, seed=seed), level)
    nulls = [_level_q(louvain(rewire(G, seed=seed + i), seed=seed + i), level)
             for i in range(1, n_null + 1)]
    mean, std = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
    return {"real": real, "null_mean": mean, "null_std": std,
                "z": (real - mean) / std if std > 0 else float("inf"), "null_scores": nulls}
 
 
def seed_stability(adj, n_seeds=10, level=-1):
    """Run Louvain with different seeds and compare the partitions pairwise with NMI
    (1 = identical partitions, 0 = unrelated). Returns dict(nmi_mean, nmi_std, q_mean, q_std)."""
    from sklearn.metrics import normalized_mutual_info_score as nmi
    nodes = sorted(adj, key=str)
    parts, qs = [], []
    for sd in range(n_seeds):
        levels = louvain(adj, seed=sd)
        idx = len(levels) - 1 if level == -1 else min(level, len(levels) - 1)
        member = levels[idx][0]
        parts.append([member[n] for n in nodes])
        qs.append(levels[idx][1])
    scores = [nmi(parts[i], parts[j]) for i in range(n_seeds) for j in range(i + 1, n_seeds)]
    return {"nmi_mean": float(np.mean(scores)), "nmi_std": float(np.std(scores)),
            "q_mean": float(np.mean(qs)), "q_std": float(np.std(qs))}