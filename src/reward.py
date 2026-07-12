import csv, glob, json, os
import numpy as np
import matplotlib.pyplot as plt

DATA = os.path.expanduser("~/p2r-data")

# demo -> rollout coordinate transform, refit from the landmark files
def fit_axis(d, r):
    A = np.stack([d, np.ones_like(d)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, r, rcond=None)
    return s, t

ld = np.array(json.load(open("landmarks_demo.json"))["points"], float)
lr = np.array(json.load(open("landmarks_rollout.json"))["points"], float)
sx, tx = fit_axis(ld[:, 0], lr[:, 0])
sy, ty = fit_axis(ld[:, 1], lr[:, 1])

# the goal: demo final-frame points, mapped into rollout coords, averaged
demo = np.load(f"{DATA}/tracks/tracks_demo_take1.npz")
gd = demo["tracks"][-1]                                       # (15, 2)
goal = np.array([sx * gd[:, 0] + tx, sy * gd[:, 1] + ty]).T.mean(axis=0)

def reward_curve(npz_path):
    d = np.load(npz_path)
    tr, vis = d["tracks"], d["visibility"]                    # (T,15,2), (T,15)
    cent = np.empty((len(tr), 2))
    for t in range(len(tr)):
        v = vis[t] > 0.5
        cent[t] = tr[t][v].mean(axis=0) if v.sum() >= 3 else tr[t].mean(axis=0)
    return -np.linalg.norm(cent - goal, axis=1)               # reward(t), pixels

# labels
label = {}
with open(f"{DATA}/rollout_clips/labels_filled.csv") as f:
    for row in csv.DictReader(f):
        label[row["clip"][:-4]] = int(float(row["success(1/0)"]))

# score all 100
GRID = np.linspace(0, 1, 100)
rows, curves = [], {0: [], 1: []}
for p in sorted(glob.glob(f"{DATA}/tracks/ck??????_ep??.npz")):
    name = os.path.basename(p)[:-4]
    r = reward_curve(p)
    rows.append((name, name.split("_")[0], label[name], r.mean(), r[-1]))
    curves[label[name]].append(np.interp(GRID, np.linspace(0, 1, len(r)), r))

with open(f"{DATA}/rewards.csv", "w") as f:
    w = csv.writer(f)
    w.writerow(["clip", "tier", "success", "score_mean", "score_final"])
    w.writerows(rows)

# Fig A: reward curves over time, success vs failure
plt.figure(figsize=(8, 5))
for y, color, nm in [(1, "tab:green", "success"), (0, "tab:red", "failure")]:
    for c in curves[y]:
        plt.plot(GRID, c, color=color, alpha=0.15, lw=1)
    plt.plot(GRID, np.mean(curves[y], axis=0), color=color, lw=3,
             label=f"{nm} (n={len(curves[y])})")
plt.xlabel("normalized time"); plt.ylabel("reward = -distance to demo goal (px)")
plt.legend(); plt.title("P2R-lite reward: successes vs failures")
plt.tight_layout(); plt.savefig(f"{DATA}/reward_curves.png", dpi=150)

# Fig B: one score per episode, by label
scores = np.array([(s, y) for _, _, y, s, _ in rows])
plt.figure(figsize=(5, 5))
jit = np.random.default_rng(0)
for y, color in [(0, "tab:red"), (1, "tab:green")]:
    v = scores[scores[:, 1] == y, 0]
    plt.scatter(jit.uniform(y - .08, y + .08, len(v)), v, color=color, alpha=.6)
plt.xticks([0, 1], ["failure", "success"]); plt.ylabel("episode score (mean reward, px)")
plt.tight_layout(); plt.savefig(f"{DATA}/score_separation.png", dpi=150)

mu1 = scores[scores[:, 1] == 1, 0].mean()
mu0 = scores[scores[:, 1] == 0, 0].mean()
print(f"mean score: success {mu1:.1f}px  failure {mu0:.1f}px  gap {mu1 - mu0:.1f}px")