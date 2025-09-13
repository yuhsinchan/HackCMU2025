import json, os, argparse
from pathlib import Path
import numpy as np
import pdb


# BODY_34 indices (per your map)
PELVIS, U_BACK, NECK = 0, 2, 3
L_SHO, R_SHO = 11, 4
L_HIP, R_HIP = 22, 18
HEAD = 27
L_TOE, R_TOE, L_HEEL, R_HEEL = 25, 21, 33, 32

SCALE_BONES = [(L_SHO,R_SHO), (L_HIP,R_HIP), (PELVIS,L_HIP), (PELVIS,R_HIP), (NECK,PELVIS)]

def _n(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def canonicalize_body34(X):
    """
    X: (T,34,3) arbitrary coords.
    Returns:
      Xc: (T,34,3) in canonical frame
      meta: dict with R (T,3,3), t (T,1,3), scale (T,), ground (T,)
    Canonical:
      - origin initially at pelvis, then shifted so feet z=0
      - +z up (pelvis->neck)
      - +x left→right across shoulders
      - +y forward (right-handed)
      - scale = mean listed bone lengths
    """
    T = X.shape[0]

    # 1) translate to pelvis
    t = X[:, PELVIS:PELVIS+1, :]
    X0 = X - t

    # 2) scale
    bl = [np.linalg.norm(X0[:, a] - X0[:, b], axis=-1) for a,b in SCALE_BONES]
    scale = np.maximum(np.mean(np.stack(bl,0), axis=0), 1e-6)  # (T,)
    Xs = X0 / scale[:,None,None]

    # 3) build axes (z=up, x=left→right, y=forward)
    up = _n((Xs[:, NECK] + Xs[:, U_BACK]) * 0.5 - Xs[:, PELVIS])   # z-axis
    x_lat = _n(Xs[:, R_SHO] - Xs[:, L_SHO])                         # provisional x-axis (L→R)
    y_fwd = _n(np.cross(up, x_lat))                                 # y = up × x
    x_lat = _n(np.cross(y_fwd, up))                                 # re-orthogonalize

    # face forward using head dir
    head_dir = Xs[:, HEAD] - Xs[:, NECK]
    flip = (np.sum(y_fwd * head_dir, axis=-1) < 0).astype(np.float32)[:,None]
    s = 1.0 - 2.0*flip
    y_fwd = y_fwd * s
    x_lat = x_lat * s  # keep right-handed when flipping y

    # Rotation: world→canonical, columns are basis [x,y,z]
    R = np.stack([x_lat, y_fwd, up], axis=-1)            # (T,3,3)
    Xc = np.einsum('tij,tbj->tbi', np.transpose(R,(0,2,1)), Xs)

    # 4) set ground so feet are at z=0 (robust: use min z among heels/toes)
    feet_z = np.stack([Xc[:, L_TOE, 2], Xc[:, R_TOE, 2], Xc[:, L_HEEL, 2], Xc[:, R_HEEL, 2]], axis=-1)
    ground = np.min(feet_z, axis=-1)                     # (T,)
    Xc[:, :, 2] -= ground[:, None]                       # shift z so ground→0
    # meta = {"R": R, "t": t, "scale": scale, "ground": ground}
    return Xc

# ---------- IO ----------
def load_runs(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    runs = data["runs"] if isinstance(data, dict) and "runs" in data else data
    if not isinstance(runs, list):
        raise ValueError("Input JSON must be a list of runs or a dict with key 'runs'.")
    return runs

def save_run(run, out_dir: Path, idx: int, zpad: int = 5):
    out = out_dir / f"run_{str(idx).zfill(zpad)}.json"
    with open(out, "w") as f:
        json.dump(run, f, separators=(",", ":"))
    return out

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Split multi-run JSON files from a directory and canonicalize BODY_34 keypoints.")
    p.add_argument("--input_dir", required=True, help="Directory containing input .json files")
    p.add_argument("--outdir", required=True, help="Output directory for per-run .json files")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--min_frames", type=int, default=9)
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    idx = args.start_idx
    kept = 0

    for fpath in files:
        runs = load_runs(fpath)
        for r in runs:
            if "data" not in r:
                raise ValueError(f"{fpath} missing key 'data'.")
            seq = np.array([s['keypoints_3d'] for s in r["data"]])
            if seq.ndim != 3 or seq.shape[1] != 34:
                raise ValueError(f"Bad seq shape {seq.shape}, expected (T,34,3).")
            if seq.shape[0] < args.min_frames:
                continue

            seq_c = canonicalize_body34(seq)
            run_out = {}
            run_out["label"] = r["label"]

            run_out["seq"] = seq_c.tolist()
            save_run(run_out, out_dir, idx)
            idx += 1; kept += 1

    print(f"wrote {kept} runs to {out_dir.resolve()}")

if __name__ == "__main__":
    main()