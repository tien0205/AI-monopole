import os, warnings
import numpy as np
import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

LO = np.array([24.0, 20.0, 0.7])        # khoảng thông số đã mô phỏng bằng CST: L, R, a (mm)
HI = np.array([36.0, 50.0, 2.0])
LO_W = np.array([15.0, 10.0, 0.3])      # khoảng tìm thiết kế khi "mở rộng" (ngoại suy)
HI_W = np.array([45.0, 120.0, 3.0])
FG = np.linspace(1.8, 3.2, 57)          # lưới tần số để huấn luyện
FO = np.linspace(1.8, 3.2, 71)          # lưới dùng khi tìm thiết kế (nhanh)
FD = np.linspace(1.8, 3.2, 281)         # lưới dùng khi hiển thị (mịn)
MODEL_FILE = "surrogate.joblib"

# ---------------- Mô hình ----------------
def feat(L, R, a, f):
    f = np.atleast_1d(f)
    n = len(f)
    return np.column_stack([np.full(n, L), np.full(n, R), np.full(n, a), f, f * L])

def train():
    d = pd.read_csv("dataset.csv").dropna(subset=["idx", "S11_2p4"])
    items = []
    for _, r in d.iterrows():
        p = f"s11_curves/{int(r['idx'])}.npy"
        if os.path.exists(p):
            c = np.load(p)
            items.append((r["L"], r["R"], r["a"], np.interp(FG, c[:, 0], c[:, 1]), r["f0"]))
    if len(items) < 6:
        raise RuntimeError("Can it nhat 6 duong cong S11 trong s11_curves/ (hay chay vtt.py).")
    X = np.vstack([feat(L, R, a, FG) for L, R, a, _, _ in items])
    y = np.concatenate([s for _, _, _, s, _ in items])
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    models = []
    for seed in range(3):               # trung bình 3 mạng để ổn định hơn
        m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                         alpha=1e-3, max_iter=5000, random_state=seed)
        m.fit(sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel())
        models.append(m)
    bad = pd.read_csv("failed.csv") if os.path.exists("failed.csv") else pd.DataFrame()
    if len(bad):
        bad = bad.dropna(subset=["L", "R", "a"])
    Xbad = bad[["L", "R", "a"]].astype(float).values if len(bad) else np.empty((0, 3))
    f0s = np.array([it[4] for it in items], float)
    M = dict(models=models, sx=sx, sy=sy,
             Xok=np.array([[it[0], it[1], it[2]] for it in items], float),
             Xbad=Xbad, f0_min=float(f0s.min()), f0_max=float(f0s.max()), n=len(items))
    joblib.dump(M, MODEL_FILE)
    return M

def load_or_train():
    fresh = (os.path.exists(MODEL_FILE) and
             os.path.getmtime(MODEL_FILE) >= os.path.getmtime("dataset.csv"))
    if fresh:
        return joblib.load(MODEL_FILE)
    print("Đang huấn luyện mô hình (chỉ lần đầu hoặc khi có dữ liệu mới)...")
    return train()

def curve(M, L, R, a, f):
    X = M["sx"].transform(feat(L, R, a, f))
    z = np.mean([m.predict(X) for m in M["models"]], axis=0)
    s = M["sy"].inverse_transform(z.reshape(-1, 1)).ravel()
    return np.clip(s, -50.0, 0.0)       # |S11| <= 1 (0 dB); chặn giá trị ngoại suy vô lý

def features(f, s):
    k = np.argmin(s)
    good = s < -10
    bw = 0.0
    if good[k]:
        i = j = k
        while i > 0 and good[i - 1]: i -= 1
        while j < len(f) - 1 and good[j + 1]: j += 1
        bw = f[j] - f[i]
    return f[k], s[k], float(np.interp(2.4, f, s)), bw

def nn_dist(p, X):
    if len(X) == 0:
        return np.inf
    return float(np.min(np.linalg.norm((X - np.asarray(p)) / (HI - LO), axis=1)))

def check(M, p):
    L, R, a = p
    out = []
    if not (LO[0] <= L <= HI[0] and LO[1] <= R <= HI[1] and LO[2] <= a <= HI[2]):
        out.append("   (!) Ngoại suy ( Ngoài L 24-36, R 20-50, a 0.7-2.0): dự đoán, chưa kiểm chứng")
    elif nn_dist(p, M["Xok"]) > 0.35:
        out.append("   (!) Xa các mẫu đã mô phỏng : Nên kiểm tra lại bằng CST")
    return out

def inverse(M, ftar, fixed, wide=True):
    """Tìm tối đa 3 bộ (L,R,a) cho f0 = ftar. fixed = [L|None, R|None, a|None].
    wide=True: tìm trong khoảng rộng (ngoại suy), không phạt đi xa dữ liệu."""
    lo, hi = (LO_W, HI_W) if wide else (LO, HI)
    bounds = [(lo[i], hi[i]) if fixed[i] is None else (fixed[i] - 1e-6, fixed[i] + 1e-6)
              for i in range(3)]

    def cost(p):
        s = np.clip(curve(M, *p, FO), -40.0, 0.0)               # chặn dự đoán S11 quá sâu vô lý
        f0 = FO[np.argmin(s)]
        c = np.interp(ftar, FO, s) + 100 * abs(f0 - ftar)       # S11 sâu tại ftar và f0 đúng
        if not wide:
            dok = nn_dist(p, M["Xok"])
            c += 30 * max(0.0, dok - 0.3)                       # phạt đi xa dữ liệu (tắt khi mở rộng)
        return c

    cands = []
    for seed in range(4):
        res = differential_evolution(cost, bounds, seed=seed, maxiter=60,
                                     popsize=12, tol=1e-6, polish=False)
        if all(np.linalg.norm((res.x - q) / (hi - lo)) > 0.12 for q, _ in cands):
            cands.append((res.x, res.fun))
    cands.sort(key=lambda t: t[1])
    return [tuple(p) for p, _ in cands[:3]]

# ---------------- Giao diện ----------------
class App:
    def __init__(self, root, M):
        self.root, self.M, self.last = root, M, None
        root.title("AI thiết kế anten đơn cực(monopole) - 2.4 GHz")
        left = ttk.Frame(root, padding=8); left.pack(side="left", fill="y")
        right = ttk.Frame(root); right.pack(side="right", fill="both", expand=True)

        self.info = ttk.Label(left, text="", foreground="gray")
        self.info.pack(anchor="w")
        self.refresh_info()

        nb = ttk.Notebook(left); nb.pack(fill="x", pady=4)
        t1 = ttk.Frame(nb, padding=8); t2 = ttk.Frame(nb, padding=8)
        nb.add(t1, text="L, R, a -> S11, f0")
        nb.add(t2, text="f0 -> L, R, a")

        self.eL = self.entry(t1, "L (mm)", "28.0", 0)
        self.eR = self.entry(t1, "R (mm)", "46.0", 1)
        self.ea = self.entry(t1, "a (mm)", "1.9", 2)
        ttk.Button(t1, text="Dự đoán S11 và f0", command=self.on_forward).grid(
            row=3, column=0, columnspan=2, pady=6, sticky="ew")

        self.ef = self.entry(t2, "f0 cần (GHz)", "2.4", 0)
        ttk.Label(t2, text="Cố định (để trống = AI tự chọn):").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.fL = self.entry(t2, "L cố định", "", 2)
        self.fR = self.entry(t2, "R cố định", "", 3)
        self.fa = self.entry(t2, "a cố định", "", 4)
        self.var_wide = tk.BooleanVar(value=True)       # mặc định: tìm trong khoảng rộng
        ttk.Checkbutton(t2, text="Mở rộng khoảng tìm (ngoài suy)",
                        variable=self.var_wide).grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(t2, text="Tìm thông số thiết kế", command=self.on_inverse).grid(
            row=6, column=0, columnspan=2, pady=6, sticky="ew")

        self.txt = tk.Text(left, width=52, height=13, wrap="word")
        self.txt.pack(fill="x", pady=4)

        self.tree = ttk.Treeview(left, columns=("f", "s"), show="headings", height=8)
        self.tree.heading("f", text="Tan so (GHz)"); self.tree.heading("s", text="S11 (dB)")
        self.tree.column("f", width=120); self.tree.column("s", width=120)
        self.tree.pack(fill="x")

        bar = ttk.Frame(left); bar.pack(fill="x", pady=6)
        ttk.Button(bar, text="Lưu CSV", command=self.on_save).pack(side="left", expand=True, fill="x")
        ttk.Button(bar, text="Huấn luyện lại", command=self.on_retrain).pack(side="left", expand=True, fill="x")

        self.fig = Figure(figsize=(6.5, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def entry(self, parent, label, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        e = ttk.Entry(parent, width=12); e.insert(0, default)
        e.grid(row=row, column=1, sticky="e", padx=6)
        return e

    def refresh_info(self):
        M = self.M
        self.info.config(text=f"Dữ liệu: {M['n']} đường cong | f0 trong dữ liệu: "
                              f"{M['f0_min']:.2f} - {M['f0_max']:.2f} GHz")

    @staticmethod
    def num(e):
        return float(e.get().replace(",", "."))

    def optional(self, e):
        t = e.get().strip()
        return float(t.replace(",", ".")) if t else None

    def set_text(self, lines):
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "\n".join(lines))

    def show(self, designs, header):
        M = self.M
        self.ax.clear()
        lines = header + [""]
        for n, p in enumerate(designs):
            L, R, a = p
            s = curve(M, L, R, a, FD)
            f0, smin, s24, bw = features(FD, s)
            self.ax.plot(FD, s, label=f"PA{n+1}: L={L:.2f}, R={R:.1f}, a={a:.2f}")
            lines += [f"PA{n+1}: L={L:.2f}  R={R:.1f}  a={a:.2f} (mm)",
                      f"   f0 = {f0:.3f} GHz | S11 min = {smin:.1f} dB",
                      f"   S11 @2.4 GHz = {s24:.1f} dB | BW(-10 dB) = {bw*1000:.0f} MHz"]
            lines += check(M, p)
            if n == 0:
                self.last = (FD, s)
                self.tree.delete(*self.tree.get_children())
                for f in np.arange(1.8, 3.21, 0.1):
                    self.tree.insert("", "end", values=(f"{f:.1f}", f"{np.interp(f, FD, s):.1f}"))
        self.ax.axhline(-10, ls="--", c="gray", lw=1)
        self.ax.axvline(2.4, ls=":", c="r", lw=1)
        self.ax.set_xlabel("Tan so (GHz)"); self.ax.set_ylabel("S11 (dB)")
        self.ax.set_ylim(top=1); self.ax.grid(True); self.ax.legend(fontsize=8)
        self.ax.set_title("S1,1 AI dự đoán")
        self.fig.tight_layout(); self.canvas.draw()
        self.set_text(lines + ["", "Lưu ý: đây là dự đoán của AI, hãy kiểm chứng bằng CST."])

    def on_forward(self):
        try:
            p = (self.num(self.eL), self.num(self.eR), self.num(self.ea))
        except ValueError:
            return messagebox.showerror("Lỗi", "Hãy nhập L, R, a là số.")
        self.show([p], ["Dự đoán cho thiết kế đã nhập"])

    def on_inverse(self):
        try:
            ftar = self.num(self.ef)
            fixed = [self.optional(self.fL), self.optional(self.fR), self.optional(self.fa)]
        except ValueError:
            return messagebox.showerror("Lỗi", "Giá trị nhập không hợp lệ.")
        if not 1.9 <= ftar <= 3.1:
            return messagebox.showerror("Lỗi", "f0 cần nằm trong 1.9 - 3.1 GHz.")
        header = [f"ục tiêu f0 = {ftar} GHz (nhiều bộ thông số cho cùng f0; AI ưu tiên bộ có S11 sau nhất)"]
        if self.var_wide.get():
            header.append("Chế độ mở rộng: có thể ngoài suy ngoài vùng đã mô phỏng bằng CST (chưa kiểm chứng)")
        if not (self.M["f0_min"] - 0.05 <= ftar <= self.M["f0_max"] + 0.05):
            header.append(f"(!) f0 nằm ngoài dải trong dữ liệu ({self.M['f0_min']:.2f}-{self.M['f0_max']:.2f} GHz): kém tin cậy")
        self.root.config(cursor="watch"); self.set_text(["Đang tìm thiết kế, vui lòng chờ..."])
        self.root.update()
        try:
            designs = inverse(self.M, ftar, fixed, self.var_wide.get())
        finally:
            self.root.config(cursor="")
        self.show(designs, header)

    def on_save(self):
        if self.last is None:
            return messagebox.showinfo("Thông báo", "Chưa có kết quả để lưu.")
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="s11_ai.csv")
        if path:
            pd.DataFrame({"f_GHz": self.last[0], "S11_dB": self.last[1]}).to_csv(path, index=False)

    def on_retrain(self):
        self.root.config(cursor="watch"); self.set_text(["Đang huấn luyện lại..."]); self.root.update()
        try:
            self.M = train()
            self.refresh_info()
            self.set_text(["Huấn luyện xong."])
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
        finally:
            self.root.config(cursor="")

if __name__ == "__main__":
    M = load_or_train()
    root = tk.Tk()
    App(root, M)
    root.mainloop()