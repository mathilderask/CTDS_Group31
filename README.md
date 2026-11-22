# Computational Tools for Data Science
# Group 31
Topic: Discovering communities on Reddit

Where do data scientists go on Reddit, and which of these subreddits belong together?
Users: all 6,335 users with at least two comments in r/datascience,
r/MachineLearning, r/dataengineering or r/statistics (June-September 2026). Each other subreddit is
the set of these users active in it. Similar pairs are found with
MinHash + LSH (Week 4), connected in a similarity graph and clustered with Louvain (Week 7).
The communities are labelled with TF.IDF and with the log-odds ratio with an informative
Dirichlet prior (Monroe et al. 2008, our method from outside the course).

## How to run
1. `pip install numpy pandas scipy scikit-learn matplotlib networkx requests`
2. The collected data is in `data/`, so this step can be skipped. To fetch it again, run
   `python fetch_data.py` (several hours; every API answer is cached in `data/cache`, so an
   interrupted run continues where it stopped). The API slows down heavy users, so the script
   waits when asked to. `SAMPLE_SIZE` can limit the run to a random sample for a quick test.
3. Run `project.ipynb` top to bottom. It prints all numbers, saves the figures in
   `report/figures/` and writes `report/results.tex` for the report.

## Files
- `fetch_data.py`: data collection from the Arctic Shift API
- `helper_functions.py`: MinHash, LSH, Louvain, null model, labels
- `project.ipynb`: the analysis
- `data_fetch.ipynb`, `data/datascience_*_2026-09.csv`: first exploration (reply graph of
  r/datascience), not used in the final pipeline
