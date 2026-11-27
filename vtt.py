import os, time
import numpy as np
import pandas as pd
import win32com.client
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

PROJECT = r"C:\Users\computer\Downloads\Monopole.cst"
S11_PATH = r"1D Results\S-Parameters\S1,1"
OUT, FAILED = "dataset.csv", "failed.csv"
TMP = os.path.abspath("s11_tmp.txt")

LO = np.array([24.0, 20.0, 0.7])     # L, R, a (mm)
HI = np.array([36.0, 50.0, 2.0])
N_INIT = 8                            # số mẫu khởi tạo tối thiểu
N_ITER = 15                           # số vòng Bayesian Optimization
BASE = (30.0, 40.0, 1.0)              # thiết kế gốc

def load(path, cols):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)

ok = load(OUT, ["idx", "L", "R", "a", "f0", "S11_min", "S11_2p4", "BW"])
bad = load(FAILED, ["idx", "L", "R", "a", "err"])
if len(ok) and "idx" not in ok.columns:
    ok["idx"] = range(len(ok))
mx = max(ok["idx"].max() if len(ok) else -1, bad["idx"].max() if len(bad) else -1)
next_idx = int(mx) + 1
os.makedirs("s11_curves", exist_ok=True)

def features(f, s):
    k = np.argmin(s)
    f0, smin = f[k], s[k]
    s24 = np.interp(2.4, f, s)
    good = s < -10
    bw = 0.0
    if good[k]:
        i = j = k
        while i > 0 and good[i - 1]: i -= 1
        while j < len(f) - 1 and good[j + 1]: j += 1
        bw = f[j] - f[i]
    return f0, smin, s24, bw

def read_s11():
    if os.path.exists(TMP):
        os.remove(TMP)
    mws.SelectTreeItem(S11_PATH)
    ax = mws.ASCIIExport()
    ax.Reset()
    ax.FileName(TMP)
    ax.Execute()
    if not os.path.exists(TMP):
        raise FileNotFoundError("ASCIIExport khong tao ra file")
    data = np.loadtxt(TMP, skiprows=2)
    return data[:, 0], data[:, 1]

def evaluate(L, R, a):
    """Chạy CST 1 mẫu. Trả về S11@2.4 GHz, hoặc None nếu mẫu hỏng."""
    global ok, bad, next_idx
    i = next_idx
    next_idx += 1
    t0 = time.time()
    L, R, a = float(L), float(R), float(a)
    try:
        mws.StoreDoubleParameter("L", L)
        mws.StoreDoubleParameter("R", R)
        mws.StoreDoubleParameter("a", a)
        mws.Rebuild()
        mws.Solver().Start()
        f, s = read_s11()
        np.save(f"s11_curves/{i}.npy", np.column_stack([f, s]))
        f0, smin, s24, bw = features(f, s)
        row = dict(idx=i, L=L, R=R, a=a, f0=f0, S11_min=smin, S11_2p4=s24, BW=bw)
        ok = pd.concat([ok, pd.DataFrame([row])], ignore_index=True)
        ok.to_csv(OUT, index=False)
        print(f"   XONG L={L:.2f} R={R:.1f} a={a:.2f} -> f0={f0:.3f} GHz, "
              f"S11@2.4={s24:.1f} dB, S11_min={smin:.1f} dB ({time.time()-t0:.0f}s)")
        return s24
    except Exception as e:
        print(f"   HONG L={L:.2f} R={R:.1f} a={a:.2f} ({type(e).__name__})")
        row = dict(idx=i, L=L, R=R, a=a, err=str(e)[:80])
        bad = pd.concat([bad, pd.DataFrame([row])], ignore_index=True)
        bad.to_csv(FAILED, index=False)
        return None

def training_data():
    X = ok[["L", "R", "a"]].astype(float).values
    y = ok["S11_2p4"].astype(float).values
    if len(bad):
        X = np.vstack([X, bad[["L", "R", "a"]].astype(float).values])
        y = np.concatenate([y, np.zeros(len(bad))])   # mẫu hỏng coi như S11 = 0 dB (tệ nhất)
    return X, y

def fit_gp(X, y):
    Xn = (X - LO) / (HI - LO)
    kernel = C(1.0) * Matern(length_scale=[0.5, 0.5, 0.5], nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                  n_restarts_optimizer=5, random_state=0)
    gp.fit(Xn, y)
    return gp

def propose(gp, y_best, rng):
    cand = qmc.scale(qmc.Sobol(d=3, scramble=True,
                               seed=int(rng.integers(1_000_000))).random(4096), LO, HI)
    mu, sd = gp.predict((cand - LO) / (HI - LO), return_std=True)
    imp = y_best - mu                       # đang tìm S11 nhỏ nhất (âm nhất)
    z = imp / np.maximum(sd, 1e-9)
    ei = imp * norm.cdf(z) + sd * norm.pdf(z)   # Expected Improvement
    return cand[np.argmax(ei)]

# ---------------- Mở CST ----------------
cst = win32com.client.Dispatch("CSTStudio.Application")
cst.OpenFile(PROJECT)
time.sleep(5)
mws = cst.Active3D()
T0 = time.time()
runs0 = len(ok) + len(bad)

# 0. Thiết kế gốc (để so sánh trong báo cáo)
has_base = (np.abs(ok[["L", "R", "a"]].astype(float).values - np.array(BASE)).sum(axis=1) < 1e-6).any()
if not has_base:
    print("Chay thiet ke goc:", BASE)
    evaluate(*BASE)

# 1. Mẫu khởi tạo (Latin Hypercube) cho đến khi đủ N_INIT mẫu thành công
init = qmc.scale(qmc.LatinHypercube(d=3, seed=7).random(3 * N_INIT), LO, HI)
k = 0
while len(ok) < N_INIT and k < len(init):
    print(f"Khoi tao: co {len(ok)}/{N_INIT} mau")
    evaluate(*init[k])
    k += 1

# 2. Vòng lặp Bayesian Optimization
rng = np.random.default_rng(0)
for it in range(N_ITER):
    X, y = training_data()
    gp = fit_gp(X, y)
    best_now = ok["S11_2p4"].astype(float).min()
    p = propose(gp, best_now, rng)
    print(f"BO {it+1}/{N_ITER}: tot nhat hien tai S11@2.4={best_now:.1f} dB | de xuat "
          f"L={p[0]:.2f} R={p[1]:.1f} a={p[2]:.2f}")
    evaluate(*p)

# 3. Tổng kết
ok["S11_2p4"] = ok["S11_2p4"].astype(float)
best = ok.loc[ok["S11_2p4"].idxmin()]
print("\n===== KET QUA =====")
for _, r in ok[(np.abs(ok[["L", "R", "a"]].astype(float).values - np.array(BASE)).sum(axis=1) < 1e-6)].iterrows():
    print(f"Thiet ke goc : L={r.L} R={r.R} a={r.a} -> f0={r.f0:.3f} GHz, "
          f"S11@2.4={r.S11_2p4:.1f} dB, BW={r.BW:.3f} GHz")
print(f"Thiet ke BO  : L={best.L:.2f} R={best.R:.1f} a={best.a:.2f} -> f0={best.f0:.3f} GHz, "
      f"S11@2.4={best.S11_2p4:.1f} dB, BW={best.BW:.3f} GHz")
print(f"So lan chay CST: {len(ok)+len(bad)-runs0} lan trong {(time.time()-T0)/60:.1f} phut "
      f"(trong do {len(bad)} mau hong)")
mws.Save()