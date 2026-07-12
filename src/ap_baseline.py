import csv, glob, os
import cv2
import numpy as np

DATA = os.path.expanduser("~/p2r-data")

def average_precision(scores, labels):
    order = np.argsort(-np.asarray(scores, float))     # best score first
    y = np.asarray(labels)[order]
    precision_at_hit = np.cumsum(y) / (np.arange(len(y)) + 1)
    return precision_at_hit[y == 1].mean()

rows = list(csv.DictReader(open(f"{DATA}/rewards.csv")))
labels = [int(r["success"]) for r in rows]

# naive baseline: demo final frame, warped into rollout coords via the landmark fit
import json
def fit_axis(d, r):
    A = np.stack([d, np.ones_like(d)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, r, rcond=None)
    return s, t
ld = np.array(json.load(open("landmarks_demo.json"))["points"], float)
lr = np.array(json.load(open("landmarks_rollout.json"))["points"], float)
sx, tx = fit_axis(ld[:, 0], lr[:, 0])
sy, ty = fit_axis(ld[:, 1], lr[:, 1])

def last_frame(path):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ok, f = cap.read()
    cap.release()
    assert ok, path
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float)

demo_cap = cv2.VideoCapture(os.path.expanduser(
    "~/p2r-data/demo/demo_take1.mov"))
demo_cap.set(cv2.CAP_PROP_POS_FRAMES, demo_cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
_, demo_final = demo_cap.read(); demo_cap.release()
M = np.array([[sx, 0, tx], [0, sy, ty]], float)
demo_goal_img = cv2.cvtColor(cv2.warpAffine(demo_final, M, (640, 480)),
                             cv2.COLOR_BGR2GRAY).astype(float)

pixel_scores = [-np.abs(last_frame(f"{DATA}/rollout_clips/{r['clip']}.mp4") - demo_goal_img).mean()
                for r in rows]

print(f"AP  P2R-lite final reward: {average_precision([r['score_final'] for r in rows], labels):.3f}")
print(f"AP  P2R-lite mean reward:  {average_precision([r['score_mean'] for r in rows], labels):.3f}")
print(f"AP  pixel-diff baseline:   {average_precision(pixel_scores, labels):.3f}")
print(f"AP  chance (base rate):    {np.mean(labels):.3f}")