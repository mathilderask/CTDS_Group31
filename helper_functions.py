import random
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

BOTS = ("[deleted]", "AutoModerator")


# ----------------------------------------------------------------------------
# Stage 0: one set of users per subreddit
# ----------------------------------------------------------------------------
def is_bot(user):
    """Deleted accounts, AutoModerator, mod-team accounts and names ending in 'bot'."""
    return user in BOTS or user.endswith("-ModTeam") or user.lower().endswith("bot")


def subreddit_sets(activity, min_count=1, min_users=10, exclude=()):
    """activity: one row per (user, subreddit, count). A user belongs to a subreddit's set with at
    least min_count posts/comments there; subreddits need at least min_users members. Subreddits in
    exclude (the seed subreddits, which contain every user) are dropped.
    Returns dict(names, set_idx, user_idx, sizes, n_users): one (set, user) pair per membership."""
    members = activity[(activity["count"] >= min_count) & ~activity["user"].map(is_bot)
                       & ~activity["subreddit"].isin(exclude)]
    sizes = members.groupby("subreddit").size()
    members = members[members["subreddit"].isin(sizes[sizes >= min_users].index)]
    names = sorted(members["subreddit"].unique())
    set_id = {s: i for i, s in enumerate(names)}
    user_codes, users = pd.factorize(members["user"])
    return {"names": names,
            "set_idx": members["subreddit"].map(set_id).to_numpy(),
            "user_idx": user_codes.astype(np.int64),
            "sizes": members.groupby("subreddit").size().reindex(names).to_numpy(),
            "n_users": len(users)}


# ----------------------------------------------------------------------------
# Stage 1: MinHash + LSH (Week 4), checked against exact Jaccard
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


def lsh_pairs(sig, b, r):
    """Banding technique: split the signature into b bands of r rows and put every set in one
    bucket per band (identical band values = same bucket). Candidate pairs share a bucket in at
    least one band. Returns a set of (i, j) with i < j."""
    assert b * r <= sig.shape[0]
    candidates = set()
    for band in range(b):
        block = np.ascontiguousarray(sig[band * r:(band + 1) * r].T)
        buckets = defaultdict(list)
        for i in range(block.shape[0]):
            buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            candidates.update(combinations(members, 2))
    return candidates


def s_curve(s, b, r):
    """Probability that a pair with Jaccard similarity s becomes a candidate."""
    return 1 - (1 - np.asarray(s, dtype=float) ** r) ** b


def jaccard_matrix(set_idx, user_idx, n_sets):
    """Exact Jaccard similarity of all pairs of sets, via a sparse set x user matrix."""
    M = sp.csr_matrix((np.ones(len(set_idx)), (set_idx, user_idx)), shape=(n_sets, user_idx.max() + 1))
    inter = (M @ M.T).toarray()
    size = np.diag(inter)
    return inter / (size[:, None] + size[None, :] - inter)


def all_pairs_jaccard(user_sets):
    """The approach LSH replaces: compare every pair of sets directly (Python sets)."""
    J = {}
    for i, j in combinations(range(len(user_sets)), 2):
        inter = len(user_sets[i] & user_sets[j])
        J[i, j] = inter / (len(user_sets[i]) + len(user_sets[j]) - inter)
    return J


def verify(candidates, user_sets, threshold):
    """Exact Jaccard for the candidate pairs only; keeps the pairs with J >= threshold."""
    similar = {}
    for i, j in candidates:
        inter = len(user_sets[i] & user_sets[j])
        jac = inter / (len(user_sets[i]) + len(user_sets[j]) - inter)
        if jac >= threshold:
            similar[i, j] = jac
    return similar


def lsh_quality(candidates, J, threshold):
    """Compare LSH candidates with the exact similar pairs (J >= threshold).
    False negatives: similar pairs LSH missed. False positives: candidates below the threshold."""
    n = J.shape[0]
    i, j = np.triu_indices(n, k=1)
    similar = {(a, b) for a, b in zip(i, j) if J[a, b] >= threshold}
    found = candidates & similar
    return {"pairs": len(i), "candidates": len(candidates), "similar": len(similar),
            "false_neg": len(similar - found), "false_pos": len(candidates - similar),
            "recall": len(found) / len(similar) if similar else float("nan")}


# ----------------------------------------------------------------------------
# Stage 2: similarity graph + Louvain (Week 7)
# ----------------------------------------------------------------------------
def similarity_graph(similar, names):
    """Weighted undirected graph as dict: node -> {neighbor: weight}, weight = Jaccard."""
    adj = defaultdict(dict)
    for (i, j), jac in similar.items():
        adj[names[i]][names[j]] = jac
        adj[names[j]][names[i]] = jac
    return dict(adj)


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
# Stage 3 (outside method): log-odds ratio with informative Dirichlet prior (Monroe et al. 2008)
# ----------------------------------------------------------------------------
def community_documents(comments, sub_comm, n_comm):
    """One document per community: all sampled comments of its subreddits."""
    text = comments.dropna(subset=["body"])
    text = text[~text["author"].astype(str).map(is_bot) & ~text["body"].isin(["[deleted]", "[removed]"])]
    text = text.assign(comm=text["subreddit"].map(sub_comm)).dropna(subset=["comm"])
    body = text["body"].str.lower().str.replace(r"http\S+", " ", regex=True)
    return body.groupby(text["comm"].astype(int)).apply(" ".join).reindex(range(n_comm)).fillna("").tolist()


def label_spread(label_words, subreddits, comments):
    """Share of a community's subreddits whose comment sample contains each label word, averaged
    over the words. A good label describes the whole community, not just one of its subreddits."""
    text = comments[comments["subreddit"].isin(subreddits)].groupby("subreddit")["body"].apply(
        lambda bodies: set(" ".join(bodies.dropna().astype(str)).lower().split()))
    if not label_words or text.empty:
        return float("nan")
    return float(np.mean([np.mean([word in words for words in text]) for word in label_words]))


def generic_share(label_words, Y, words, n_common=50):
    """Share of the label words that are among the n_common most frequent words of all
    communities together. Such words say little about one community."""
    common = set(words[np.argsort(np.asarray(Y.sum(axis=0)).ravel())[::-1][:n_common]])
    return float(np.mean([w in common for w in label_words])) if label_words else float("nan")


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
    if Yd.max() == 0:                            # community without comments
        return []
    tf = Yd / Yd.max()
    n_t = np.asarray((Y > 0).sum(axis=0)).ravel()
    idf = np.log2(Y.shape[0] / n_t)
    score = tf * idf
    top = np.argsort(score)[::-1][:n]
    return list(words[top[score[top] > 0]])      # drop zero-score filler


# ----------------------------------------------------------------------------
# Evaluation: null model (Week 7 + Bonferroni's principle, Week 1) and vertex order
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
    return {n: {m: 1.0 for m in sorted(s, key=str)} for n, s in nbrs.items()}  # sorted: same result every run


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


def order_stability(adj, n_orders=10, level=-1):
    """Louvain depends on the (random) vertex order. Run it with different orders and report
    how much the modularity and the number of communities change."""
    qs, counts = [], []
    for seed in range(n_orders):
        levels = louvain(adj, seed=seed)
        idx = len(levels) - 1 if level == -1 else min(level, len(levels) - 1)
        qs.append(levels[idx][1])
        counts.append(len(set(levels[idx][0].values())))
    return {"q_mean": float(np.mean(qs)), "q_min": float(np.min(qs)), "q_max": float(np.max(qs)),
            "n_comm_min": int(min(counts)), "n_comm_max": int(max(counts))}


# ----------------------------------------------------------------------------
# Evaluation: does the similarity graph help to recommend subreddits? (hold-out test)
# ----------------------------------------------------------------------------
def hold_out(sets, min_memberships=3, seed=0):
    """Hide one random membership of every user with at least min_memberships subreddits.
    Returns the training (set_idx, user_idx) pairs and the hidden (user, subreddit) pairs."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sets["set_idx"]))
    s, u = sets["set_idx"][order], sets["user_idx"][order]
    counts = np.bincount(u, minlength=sets["n_users"])
    first = np.unique(u, return_index=True)[1]                  # one random membership per user
    hidden = first[counts[u[first]] >= min_memberships]
    keep = np.ones(len(s), bool)
    keep[hidden] = False
    return {"set_idx": s[keep], "user_idx": u[keep], "test_user": u[hidden], "test_set": s[hidden]}


def similarity_matrix(similar, n_sets):
    """Dense symmetric matrix W[i, j] = Jaccard of the verified pairs, 0 otherwise."""
    W = np.zeros((n_sets, n_sets))
    for (i, j), jac in similar.items():
        W[i, j] = W[j, i] = jac
    return W


def recommendation_ranks(split, W, n_sets, n_users):
    """Score every subreddit for a user by its summed similarity to the user's other subreddits
    (ties broken by popularity) and return the rank of the hidden subreddit (1 = top)."""
    U = sp.csr_matrix((np.ones(len(split["set_idx"])), (split["user_idx"], split["set_idx"])),
                      shape=(n_users, n_sets))
    popularity = np.asarray(U.sum(axis=0)).ravel()
    users = split["test_user"]
    scores = np.asarray(U[users] @ W) + 1e-9 * popularity / popularity.max()
    scores[U[users].toarray() > 0] = -np.inf                      # do not recommend known subreddits
    hidden_score = scores[np.arange(len(users)), split["test_set"]]
    return 1 + (scores > hidden_score[:, None]).sum(axis=1)


def hit_rate(ranks, ks):
    return np.array([(ranks <= k).mean() for k in ks])


# ----------------------------------------------------------------------------
# Evaluation: runtime of LSH vs. comparing all pairs as the number of subreddits grows
# ----------------------------------------------------------------------------
def scaling_experiment(sets, sizes, n_hash, b, r, threshold, seed=0):
    rng = np.random.default_rng(seed)
    n_sets = len(sets["names"])
    rows = []
    for m in sizes:
        chosen = np.sort(rng.choice(n_sets, size=min(m, n_sets), replace=False))
        new_id = -np.ones(n_sets, int)
        new_id[chosen] = np.arange(len(chosen))
        mask = new_id[sets["set_idx"]] >= 0
        s_idx, u_idx = new_id[sets["set_idx"][mask]], sets["user_idx"][mask]
        user_sets = [set(u_idx[s_idx == k]) for k in range(len(chosen))]

        start = time.perf_counter()
        exact = all_pairs_jaccard(user_sets)
        time_exact = time.perf_counter() - start
        start = time.perf_counter()
        sig = minhash(s_idx, u_idx, len(chosen), k=n_hash)
        found = verify(lsh_pairs(sig, b, r), user_sets, threshold)
        time_lsh = time.perf_counter() - start

        similar = {p for p, jac in exact.items() if jac >= threshold}
        rows.append({"subreddits": len(chosen), "pairs": len(exact), "time exact [s]": time_exact,
                     "time LSH [s]": time_lsh, "recall": len(similar & set(found)) / max(len(similar), 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Who goes where: lift (Week 5) of each seed subreddit's users per community
# ----------------------------------------------------------------------------
def seed_lift(activity, sub_comm, seeds, communities, min_count=2, n_boot=500, seed=0):
    """Users who come from exactly one seed subreddit form a group per seed. For every group and
    community: share of the group's users who are members of the community, divided by the share
    among all users (lift; 1 = as often as everybody else). A bootstrap over the group's users
    gives a 95% interval. Returns dict(lift, low, high, sizes) with communities as rows."""
    rng = np.random.default_rng(seed)
    seed_sets = activity[activity["subreddit"].isin(seeds)].groupby("user")["subreddit"].apply(frozenset)
    members = activity[(activity["count"] >= min_count) & activity["subreddit"].isin(sub_comm)]
    users = sorted(members["user"].unique())                      # users with at least one membership
    row = {u: i for i, u in enumerate(users)}
    A = np.zeros((len(users), len(communities)), bool)              # user x community membership
    col = {c: j for j, c in enumerate(communities)}
    for u, s in zip(members["user"], members["subreddit"]):
        if sub_comm[s] in col:
            A[row[u], col[sub_comm[s]]] = True
    overall = A.mean(axis=0)
    lift, low, high, sizes = {}, {}, {}, {}
    for s in seeds:
        group = np.array([row[u] for u, ss in seed_sets.items() if ss == frozenset([s]) and u in row])
        sizes[s] = len(group)
        lift[s] = A[group].mean(axis=0) / overall
        boot = np.array([A[rng.choice(group, len(group))].mean(axis=0) / overall for _ in range(n_boot)])
        low[s], high[s] = np.percentile(boot, [2.5, 97.5], axis=0)
    frame = lambda d: pd.DataFrame(d, index=communities)
    return {"lift": frame(lift), "low": frame(low), "high": frame(high), "sizes": sizes}
