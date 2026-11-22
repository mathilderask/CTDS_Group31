"""Fetch the data for the project from the Arctic Shift API (https://github.com/ArthurHeitmann/arctic_shift).

Steps (each API answer is cached as JSON in data/cache, so a rerun skips finished requests):
1. users:      everyone with at least MIN_SEED_COMMENTS comments in one of the seed subreddits
               (June-September 2026), asked month by month (week by week if a month is too heavy).
               We use all of them (SAMPLE_SIZE = None); a number instead takes a random sample,
               e.g. for a quick test, as the API limits how fast we can ask.
2. activity:   for each user, the subreddits they posted or commented in over the same period
               (top 100 per user). Each subreddit becomes the set of our users who are active in it.
3. comments:   COMMENTS_PER_SUBREDDIT comments for every subreddit with at least MIN_USERS members
               (users with at least MIN_COUNT posts/comments there), used to label the communities.

We first tried to fetch every commenter of each subreddit (comments/search/aggregate by subreddit),
but those queries hit the API's ~10 s limit for most active subreddits. The per-user endpoint is
fast and never failed, so the sets are built from user histories instead.

To get more data, add seed subreddits and rerun. Two requests run in parallel
(WORKERS) and we wait whenever the API asks us to slow down; expect about 30 users per minute.
Run in a notebook cell or as: python fetch_data.py
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://arctic-shift.photon-reddit.com/api"
CACHE = Path("data/cache")

SEED_SUBREDDITS = ["datascience", "MachineLearning", "dataengineering", "statistics"]
START, END = "2026-06-01", "2026-10-01"
MIN_SEED_COMMENTS = 2         # users need at least this many comments in a seed subreddit
SAMPLE_SIZE = None            # None = all seed users (as in the report); a number = random sample
MIN_COUNT = 2                 # a user is a member of a subreddit with at least this many posts/comments
MIN_USERS = 10                # comments are fetched for subreddits with at least this many members
COMMENTS_PER_SUBREDDIT = 100
COMMENT_WINDOW = ("2026-09-15", "2026-09-22")
PAUSE = 1.5                   # seconds between requests; the API is free, so we keep the load low
WORKERS = 2                   # parallel requests


def get_cached(name, endpoint, params, retries=4):
    """GET one API endpoint, or read the answer from the cache if it was fetched before.
    Returns the 'data' field, or None if the API kept failing."""
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())["data"]
    if path.with_suffix(".failed").exists():
        return None
    answer = {}
    for attempt in range(retries):
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=60)
        answer = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        if response.status_code == 200 and answer.get("data") is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"params": params, "data": answer["data"]}))
            time.sleep(PAUSE)
            return answer["data"]
        # 429 = rate limited, 422 = "Timeout. Maybe slow down a bit": wait as long as the API asks
        reset = response.headers.get("x-ratelimit-reset")
        time.sleep(min(float(reset) + 1, 90) if reset else 15 * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".failed").write_text(json.dumps({"params": params, "error": answer.get("error")}))
    return None


def windows(start, end, freq):
    edges = pd.date_range(start, end, freq=freq).strftime("%Y-%m-%d").tolist()
    edges = sorted(set([start] + edges + [end]))
    return list(zip(edges[:-1], edges[1:]))


def seed_counts(subreddit):
    """Comments per author in a seed subreddit, month by month (week by week if a month fails)."""
    counts = pd.Series(dtype=int)
    for start, end in windows(START, END, "MS"):
        params = {"aggregate": "author", "subreddit": subreddit, "after": start, "before": end, "limit": 100000}
        rows = get_cached(f"seeds/{subreddit}_{start}", "comments/search/aggregate", params)
        parts = [rows] if rows is not None else [
            get_cached(f"seeds/{subreddit}_{a}_{b}", "comments/search/aggregate", {**params, "after": a, "before": b})
            for a, b in windows(start, end, "7D")]
        for part in parts:
            if part:
                c = pd.DataFrame(part).astype({"count": int}).set_index("key")["count"]
                counts = counts.add(c, fill_value=0)
    return counts


def seed_users():
    users = set()
    for subreddit in SEED_SUBREDDITS:
        counts = seed_counts(subreddit)
        users |= set(counts[counts >= MIN_SEED_COMMENTS].index)
        print(f"seed r/{subreddit}: {int((counts >= MIN_SEED_COMMENTS).sum())} users")
    return sorted(users - {"[deleted]", "AutoModerator"})


def user_activity(users):
    """One row per (user, subreddit) with the number of posts and comments."""
    def fetch(user):
        data = get_cached(f"users/{user}", "users/interactions/subreddits",
                          {"author": user, "after": START, "before": END, "limit": 100})
        return [{"user": user, "subreddit": r["subreddit"], "count": r["count"]} for r in data or []]

    rows = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, user_rows in enumerate(pool.map(fetch, users)):
            rows += user_rows
            if i % 250 == 0:
                print(f"activity: {i}/{len(users)} users")
    return pd.DataFrame(rows)


def sample(users):
    """A random sample of SAMPLE_SIZE users (always the same one, seed 0)."""
    if SAMPLE_SIZE is None or SAMPLE_SIZE >= len(users):
        return users
    rng = random.Random(0)
    return sorted(rng.sample(users, SAMPLE_SIZE))


def frequent_subreddits(activity):
    members = activity[activity["count"] >= MIN_COUNT]
    users_per_subreddit = members.groupby("subreddit")["user"].nunique()
    return sorted(users_per_subreddit[users_per_subreddit >= MIN_USERS].index)


def comment_samples(subreddits):
    """COMMENTS_PER_SUBREDDIT comments per subreddit from COMMENT_WINDOW."""
    def fetch(subreddit):
        data = get_cached(f"comments/{subreddit}", "comments/search",
                          {"subreddit": subreddit, "after": COMMENT_WINDOW[0], "before": COMMENT_WINDOW[1],
                           "limit": COMMENTS_PER_SUBREDDIT, "fields": "id,author,body"})
        return [{"subreddit": subreddit, **r} for r in data or []]

    rows = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, sub_rows in enumerate(pool.map(fetch, subreddits)):
            rows += sub_rows
            if i % 250 == 0:
                print(f"comments: {i}/{len(subreddits)} subreddits")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_users = seed_users()
    users = sample(all_users)
    print(f"users: {len(all_users)}, sample: {len(users)}")
    pd.Series(users, name="user").to_csv("data/users_sample.csv", index=False)
    activity = user_activity(users)
    subreddits = frequent_subreddits(activity)
    print(f"subreddits with >= {MIN_USERS} users: {len(subreddits)}")
    comments = comment_samples(subreddits)

    activity.to_csv("data/user_activity_2026-06_09.csv.gz", index=False)
    comments.to_csv("data/comment_samples_2026-09.csv.gz", index=False)
    print(f"saved: {len(activity)} (user, subreddit) rows, {len(comments)} comments")
