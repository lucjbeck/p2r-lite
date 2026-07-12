import csv, json, os
import numpy as np

DATA = os.path.expanduser("~/p2r-data")
TIER_OFFSET = {"ck020000": 0, "ck040000": 20, "ck060000": 40, "ck080000": 60, "ck100000": 80}
rng = np.random.default_rng(42)              # preregistered seed

rows = list(csv.DictReader(open(f"{DATA}/rewards.csv")))
for r in rows:
    tier, ep = r["clip"].split("_ep")
    r["idx"] = TIER_OFFSET[tier] + int(ep)   # index in the merged m3_pool dataset
    r["score_final"] = float(r["score_final"])

fails = [r for r in rows if r["success"] == "0"]
succs = [r for r in rows if r["success"] == "1"]
pool = fails + [succs[i] for i in rng.choice(len(succs), 18, replace=False)]  # 51 eps, 35%

k = 17                                        # top third, fixed a priori
top_k = sorted(pool, key=lambda r: -r["score_final"])[:k]
random_k = [pool[i] for i in rng.choice(len(pool), k, replace=False)]

arms = {"all": sorted(r["idx"] for r in pool),
        "top_k": sorted(r["idx"] for r in top_k),
        "random_k": sorted(r["idx"] for r in random_k)}
json.dump(arms, open(f"{DATA}/m3_arms.json", "w"), indent=2)
for name, idxs in arms.items():
    n_succ = sum(1 for r in pool if r["idx"] in set(idxs) and r["success"] == "1")
    print(f"{name}: {len(idxs)} eps, {n_succ} labeled successes")
    print(f"  --dataset.episodes='[{','.join(map(str, idxs))}]'")