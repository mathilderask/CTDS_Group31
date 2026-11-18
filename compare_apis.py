# Compare Reddit data sources on the same queries (r/datascience, September 2026).
# Run in a notebook cell or as: python compare_apis.py
import time
import pandas as pd
import requests

START = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
END = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
NEEDED = ["id", "author", "subreddit", "body", "parent_id", "link_id", "created_utc"]
HEADERS = {"User-Agent": "DTU-02807-group31-api-test"}
REPEATS = 2  # run each test twice to see if errors are just bad luck

SUB, USER = "datascience", "CJDrew"


def archive_rows(js):          # Arctic Shift and PullPush: {"data": [...]}
    return js.get("data") or []


def reddit_rows(js):           # Reddit's own endpoint: {"data": {"children": [{"data": {...}}]}}
    return [c["data"] for c in js.get("data", {}).get("children", [])]


# name: (url per query, params per query, how to read the response)
APIS = {
    "Arctic Shift": (
        {"subreddit": "https://arctic-shift.photon-reddit.com/api/comments/search",
         "author": "https://arctic-shift.photon-reddit.com/api/comments/search"},
        {"subreddit": {"subreddit": SUB, "after": START, "before": END, "limit": 100},
         "author": {"author": USER, "after": START, "before": END, "limit": 100}},
        archive_rows),
    "PullPush": (
        {"subreddit": "https://api.pullpush.io/reddit/search/comment/",
         "author": "https://api.pullpush.io/reddit/search/comment/"},
        {"subreddit": {"subreddit": SUB, "after": START, "before": END, "size": 100},
         "author": {"author": USER, "after": START, "before": END, "size": 100}},
        archive_rows),
    # No date filter here: Reddit only returns the newest items, so we filter afterwards.
    "Reddit (official, no key)": (
        {"subreddit": f"https://www.reddit.com/r/{SUB}/comments.json",
         "author": f"https://www.reddit.com/user/{USER}/comments.json"},
        {"subreddit": {"limit": 100}, "author": {"limit": 100}},
        reddit_rows),
}

rows = []
for api, (urls, params, read) in APIS.items():
    for query in ["subreddit", "author"]:
        for run in range(1, REPEATS + 1):
            t = time.time()
            try:
                r = requests.get(urls[query], params=params[query], headers=HEADERS, timeout=60)
                status = r.status_code
                data = read(r.json()) if status == 200 else []
            except Exception as e:
                status, data = f"error: {type(e).__name__}", []
            df = pd.DataFrame(data)
            if "created_utc" in df:  # keep September 2026 only, same for every source
                df = df[(df["created_utc"] >= START) & (df["created_utc"] < END)]
            rows.append({
                "api": api, "query": query, "run": run, "status": status,
                "seconds": round(time.time() - t, 1),
                "Sept records": len(df),
                "subreddits": df["subreddit"].nunique() if "subreddit" in df else 0,
                "missing fields": "-" if df.empty else [f for f in NEEDED if f not in df.columns],
            })
            time.sleep(2)

pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))