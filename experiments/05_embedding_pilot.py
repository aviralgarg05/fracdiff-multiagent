"""Pilot: does a turn-level sentence-embedding trajectory actually diffuse?

Measures, for a conversation embedded turn by turn:
  * headroom   = (diameter of the space) / IQR of displacement from turn 0.
                 Below ~10 the boundary alone drives alpha towards 2, so the
                 estimate would be an artifact (see 03_stress_tests.py).
  * growth     = log-log slope of displacement against turn index. Free
                 diffusion gives roughly 0.25-0.5; ~0 means confined.
  * anisotropy = participation ratio of the increment spectrum.

The control is a conversation that repeats one sentence: if a genuinely drifting
conversation does not separate from it, the observable is not diffusive.

Requires `transformers` and `torch`.

    python experiments/05_embedding_pilot.py
"""
import sys
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# some environments ship a broken pyarrow that transformers imports transitively
sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
sys.modules["pyarrow"].__version__ = "18.0.0"

import torch                                        # noqa: E402
from transformers import AutoModel, AutoTokenizer   # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).eval()


def embed(texts, normalize=True):
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
        hidden = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = ((hidden * mask).sum(1) / mask.sum(1)).numpy()
    if normalize:
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def report(tag, V, bound):
    X = V - V[0]
    disp = np.linalg.norm(X, axis=1)
    step = np.linalg.norm(np.diff(V, axis=0), axis=1)
    iqr = np.subtract(*np.percentile(disp[1:], [75, 25]))
    k = np.arange(1, len(disp))
    slope = np.polyfit(np.log(k), np.log(disp[1:] + 1e-12), 1)[0]
    sv = np.linalg.svd(np.diff(V, axis=0), compute_uv=False)
    sv = sv / sv.sum()
    half = len(disp) // 2
    print(f"--- {tag}")
    print(f"    median step {np.median(step):.4f}   median displacement {np.median(disp):.4f}")
    print(f"    headroom  = {bound:.2f} / {iqr:.4f} = {bound / iqr:.1f}   (want >= 10)")
    print(f"    growth exponent = {slope:.3f}   (free diffusion ~0.25-0.5; 0 = confined)")
    print(f"    2nd half / 1st half mean displacement = "
          f"{np.mean(disp[half:]) / np.mean(disp[1:half]):.2f}")
    print(f"    increment anisotropy: top-1 {sv[0]:.3f}, top-5 {sv[:5].sum():.3f}, "
          f"participation {1 / np.sum(sv ** 2):.1f} of {len(sv)}")


chain = ["database index", "query caching", "cloud cost", "memory limits",
         "horizontal sharding", "data migration", "schema versioning", "team process",
         "code review culture", "hiring", "onboarding docs", "company strategy",
         "market positioning", "pricing", "customer churn", "support tooling",
         "incident response", "observability", "security audits", "compliance"]
drifting = []
for i, c in enumerate(chain):
    drifting.append(f"Turn {2*i}: Let us discuss {c} in detail and what it means for us.")
    drifting.append(f"Turn {2*i+1}: On {c}, the main constraint is effort versus benefit.")
repeated = [f"Turn {i}: Regarding the database index, we should weigh effort against benefit."
            for i in range(40)]

V = embed(drifting, normalize=True)
report("40 distinct turns across 20 topics, L2-normalised", V, 2.0)
V2 = embed(drifting, normalize=False)
report("same, unnormalised", V2, 2 * np.linalg.norm(V2, axis=1).max())
V3 = embed(repeated, normalize=True)
report("CONTROL: 40 near-identical turns", V3, 2.0)

print()
print("If the drifting conversation does not separate from the control, turn-level")
print("embedding position is not a diffusing observable, and free diffusion is the")
print("wrong model for it regardless of whether the headroom check passes.")
