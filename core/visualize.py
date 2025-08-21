#!/usr/bin/env python3
# tools/plot_eta_hist.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_eta(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        eta = np.load(path)
    elif ext in (".pt", ".pth"):
        import torch
        t = torch.load(path, map_location="cpu")
        try:
            eta = t.detach().cpu().numpy()
        except Exception:
            eta = np.asarray(t)
    elif ext == ".mhd":
        try:
            import SimpleITK as sitk
        except Exception as e:
            raise RuntimeError("`.mhd` を読むには SimpleITK が必要です: pip install SimpleITK") from e
        img = sitk.ReadImage(path)
        eta = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    else:
        raise ValueError(f"未対応の拡張子です: {ext}")
    eta = np.asarray(eta, dtype=np.float32)
    if eta.ndim != 3:
        raise ValueError(f"3次元配列が必要です（今は {eta.shape}）")
    return eta

def save_histograms(data: np.ndarray, out_png: str, out_csv: str, bins: int = 100, log: bool = False, title: str = ""):
    # ヒストグラム
    fig = plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, log=log)
    plt.xlabel("eta")
    plt.ylabel("count (log)" if log else "count")
    plt.title(title or "Histogram of eta")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # CSV（ビン中心, カウント）
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    np.savetxt(out_csv, np.c_[centers, counts], delimiter=",", header="eta,count", comments="")

def save_cdf(data: np.ndarray, out_png: str, title: str = ""):
    # 累積分布（任意）
    x = np.sort(data)
    y = np.linspace(0, 1, x.size, endpoint=True)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.xlabel("eta")
    plt.ylabel("CDF")
    plt.title(title or "Cumulative Distribution of eta")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Plot histogram (and CDF) of eta volume from eta_final.*")
    ap.add_argument("path", help="eta_final.npy / .pt / .mhd へのパス")
    ap.add_argument("--bins", type=int, default=100, help="ヒストグラムのビン数")
    ap.add_argument("--delta", action="store_true", help="ηではなく Δn = η - 1 をプロット")
    ap.add_argument("--log", action="store_true", help="ヒストグラムのY軸を対数表示")
    ap.add_argument("--clip", type=float, nargs=2, metavar=("MIN", "MAX"),
                    help="範囲外をクリップしてから描画（例: --clip 0.9 1.1）")
    ap.add_argument("--outdir", default=None, help="出力先ディレクトリ（既定: 入力と同じ場所）")
    args = ap.parse_args()

    # 読み込み
    eta = load_eta(args.path)  # (Z, Y, X)
    data = eta.ravel().astype(np.float64)
    data = data[np.isfinite(data)]  # NaN/Inf除去

    # Δn にする場合
    if args.delta:
        data = data - 1.0

    # クリップ（任意）
    if args.clip is not None:
        lo, hi = float(args.clip[0]), float(args.clip[1])
        data = np.clip(data, lo, hi)

    # 統計の表示
    p = np.percentile(data, [0, 1, 5, 50, 95, 99, 100])
    print("eta stats:",
          f"min={data.min():.6f}",
          f"max={data.max():.6f}",
          f"mean={data.mean():.6f}",
          f"std={data.std():.6f}",
          f"p1={p[1]:.6f}",
          f"p99={p[5]:.6f}",
          f"shape={eta.shape}",
          sep="  ")

    # 出力先
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.path))
    os.makedirs(outdir, exist_ok=True)
    base = "eta_delta" if args.delta else "eta"
    out_png = os.path.join(outdir, f"{base}_hist.png")
    out_csv = os.path.join(outdir, f"{base}_hist.csv")
    out_cdf = os.path.join(outdir, f"{base}_cdf.png")

    # 保存
    ttl = f"Histogram of {'Δn' if args.delta else 'eta'}"
    save_histograms(data, out_png, out_csv, bins=args.bins, log=args.log, title=ttl)
    save_cdf(data, out_cdf, title=f"CDF of {'Δn' if args.delta else 'eta'}")

    print(f"saved: {out_png}")
    print(f"saved: {out_csv}")
    print(f"saved: {out_cdf}")

if __name__ == "__main__":
    main()
