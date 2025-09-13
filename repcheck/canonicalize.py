import numpy as np

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

    meta = {"R": R, "t": t, "scale": scale, "ground": ground}
    return Xc, meta



# # preprocess.py
# import numpy as np

# BONES = [(2,5),(9,12),(2,9),(5,12),(0,1),(1,2)]  # adapt to your skeleton

# def canonicalize(X):  # X: (T,J,3)
#     pelvis = X[:,8:9,:]
#     X = X - pelvis
#     # scale by mean bone length
#     bl = []
#     for a,b in BONES: bl.append(np.linalg.norm(X[:,a]-X[:,b], axis=-1))
#     scale = np.maximum(np.mean(bl), 1e-6)
#     X = X / scale
#     # yaw align: rotate around vertical so shoulders lie on x-axis
#     sh_l, sh_r = X[:,2], X[:,5]
#     v = sh_r - sh_l
#     yaw = -np.arctan2(v[:,1], v[:,0])  # around z
#     cz, sz = np.cos(yaw), np.sin(yaw)
#     Rz = np.stack([np.stack([cz,-sz,np.zeros_like(cz)],-1),
#                    np.stack([sz, cz,np.zeros_like(cz)],-1),
#                    np.stack([np.zeros_like(cz),np.zeros_like(cz),np.ones_like(cz)],-1)], -2)  # (T,3,3)
#     X = np.einsum('tij,tbj->tbi', Rz, X)
#     return X

# def resample_clip(X, T=64):
#     t_orig = np.linspace(0,1,len(X))
#     t_new  = np.linspace(0,1,T)
#     Xr = np.stack([np.interp(t_new, t_orig, X[...,k]) for k in range(3)], axis=-1)  # wrong shape if used directly
#     # do per-joint
#     out = []
#     for j in range(X.shape[1]):
#         out.append(np.column_stack([np.interp(t_new,t_orig,X[:,j,k]) for k in range(3)]))
#     return np.stack(out, axis=1)  # (T,J,3)

# def make_features(X):  # X: (T,J,3)
#     V = np.gradient(X, axis=0)          # velocities
#     # simple angles: knee L/R example
#     def angle(a,b,c):
#         v1=a-b; v2=c-b
#         cos=(v1*v2).sum(-1)/(np.linalg.norm(v1,axis=-1)*np.linalg.norm(v2,axis=-1)+1e-6)
#         return np.arccos(np.clip(cos,-1,1))
#     Lk = angle(X[:,12], X[:,13], X[:,14])  # adjust indices
#     Rk = angle(X[:,9],  X[:,10], X[:,11])
#     feats = np.concatenate([X.reshape(len(X),-1),
#                             V.reshape(len(V),-1),
#                             Lk[:,None], Rk[:,None],
#                             (Lk-Rk)[:,None]], axis=1)
#     return feats.astype(np.float32)  # (T,D)
