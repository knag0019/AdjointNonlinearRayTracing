import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from PIL import Image

import grid
import source
import sensor
import tracer
import optimizer
from utils import plot_utils


# =========================================================
#  ηフィールド 保存／可視化ユーティリティ
# =========================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _normalize_img(img):
    vmin, vmax = np.min(img), np.max(img)
    if vmax - vmin < 1e-12:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin + 1e-12)

def save_eta_arrays(eta_t: torch.Tensor, result_dir: str, voxel_size: float):
    """ η を .pt / .npy に保存し、MetaImage(.mhd+.raw)も書き出す """
    _ensure_dir(result_dir)

    eta = eta_t.detach().float().cpu().numpy()  # (Z, Y, X) を想定
    # 基本保存
    torch.save(torch.from_numpy(eta), os.path.join(result_dir, "eta_final.pt"))
    np.save(os.path.join(result_dir, "eta_final.npy"), eta)

    # MetaImage（ParaView, 3D Slicerなどで読める）
    nx, ny, nz = eta.shape[2], eta.shape[1], eta.shape[0]  # X Y Z の順で書く必要あり
    raw_name = "eta_final.raw"
    hdr_path = os.path.join(result_dir, "eta_final.mhd")
    raw_path = os.path.join(result_dir, raw_name)
    with open(hdr_path, "w") as f:
        f.write("ObjectType = Image\n")
        f.write("NDims = 3\n")
        f.write(f"DimSize = {nx} {ny} {nz}\n")
        f.write("ElementType = MET_FLOAT\n")  # float32
        f.write(f"ElementSpacing = {voxel_size} {voxel_size} {voxel_size}\n")
        f.write(f"ElementDataFile = {raw_name}\n")
    eta.astype(np.float32).tofile(raw_path)

def save_eta_triplanar(eta_t: torch.Tensor, out_path: str):
    """ 中央断面（XY, XZ, YZ）を1枚に """
    eta = eta_t.detach().float().cpu().numpy()
    zc, yc, xc = np.array(eta.shape) // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_normalize_img(eta[zc, :, :]), cmap="gray", origin="lower"); axes[0].set_title(f"Axial (z={zc})")
    axes[1].imshow(_normalize_img(eta[:, yc, :]), cmap="gray", origin="lower"); axes[1].set_title(f"Coronal (y={yc})")
    axes[2].imshow(_normalize_img(eta[:, :, xc]), cmap="gray", origin="lower"); axes[2].set_title(f"Sagittal (x={xc})")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_eta_montage(eta_t: torch.Tensor, out_path: str, num_slices: int = 8):
    """ Z方向の等間隔スライスをモンタージュ保存 """
    eta = eta_t.detach().float().cpu().numpy()
    Z = eta.shape[0]
    idxs = np.linspace(0, Z - 1, num_slices, dtype=int)
    cols = min(num_slices, 8)
    rows = int(np.ceil(num_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
    axes = np.array(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(idxs):
            z = idxs[i]
            ax.imshow(_normalize_img(eta[z, :, :]), cmap="gray", origin="lower")
            ax.set_title(f"z={z}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_eta_mip(eta_t: torch.Tensor, out_path: str):
    """ 各軸方向の最大値投影（MIP） """
    eta = eta_t.detach().float().cpu().numpy()
    mip_z = _normalize_img(eta.max(axis=0))  # XY
    mip_y = _normalize_img(eta.max(axis=1))  # XZ
    mip_x = _normalize_img(eta.max(axis=2))  # YZ
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mip_z, cmap="gray", origin="lower"); axes[0].set_title("MIP (Z→)")
    axes[1].imshow(mip_y, cmap="gray", origin="lower"); axes[1].set_title("MIP (Y→)")
    axes[2].imshow(mip_x, cmap="gray", origin="lower"); axes[2].set_title("MIP (X→)")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_eta_hist(eta_t: torch.Tensor, out_path: str, bins: int = 100):
    """ η値のヒストグラム """
    eta = eta_t.detach().float().cpu().numpy()
    fig = plt.figure(figsize=(5, 4))
    plt.hist(eta.ravel(), bins=bins)
    plt.xlabel("eta"); plt.ylabel("count"); plt.title("Histogram of eta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def optional_save_eta_isosurface_ply(eta_t: torch.Tensor, out_path: str, iso=None):
    """ scikit-image があれば等値面を .ply へ（無ければスキップ） """
    try:
        from skimage.measure import marching_cubes
        eta = eta_t.detach().float().cpu().numpy()
        if iso is None:
            iso = float(np.percentile(eta, 75))  # 例: 上位25%の等値面
        verts, faces, normals, values = marching_cubes(eta, level=iso)
        with open(out_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\nend_header\n")
            # (z,y,x)→(x,y,z)
            for v in verts:
                f.write(f"{v[2]} {v[1]} {v[0]}\n")
            for tri in faces:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
    except Exception as e:
        print(f"[isosurface] skipped ({e})")

def save_eta_and_visuals(eta_t: torch.Tensor, result_dir: str, voxel_size: float, num_slices: int = 8):
    """ 配列保存＋基本可視化＋外部ツール用書き出しまで """
    _ensure_dir(result_dir)
    save_eta_arrays(eta_t, result_dir, voxel_size)
    save_eta_triplanar(eta_t, os.path.join(result_dir, "eta_triplanar.png"))
    save_eta_montage(eta_t, os.path.join(result_dir, "eta_slices.png"), num_slices=num_slices)
    save_eta_mip(eta_t, os.path.join(result_dir, "eta_mip.png"))
    save_eta_hist(eta_t, os.path.join(result_dir, "eta_hist.png"))
    optional_save_eta_isosurface_ply(eta_t, os.path.join(result_dir, "eta_iso.ply"))  # optional


# =========================================================
#  最適化本体
# =========================================================

def multires_opt(params, result_dir):
    _ensure_dir(result_dir)

    disp_ims = params.get('disp_ims', [None])
    defl_ims = params.get('defl_ims', [None])
    defl_weight = params.get('defl_weight', 1.0)
    sdf_loss = params.get('sdf_loss', False)
    sdf_disp = params.get('sdf_disp', [None])
    sdf_defl = params.get('sdf_defl', [None])
    res_list = params.get('res_list', [3, 5, 9, 17, 33, 65])
    vol_span = params.get('vol_span', 1)
    spp = params.get('spp', 1)
    sensor_dist = params.get('sensor_distance', 0)
    step_res = params.get('step_res', 2)
    angle_s = params.get('angle_span', 360)
    far_sensor_span = params.get('far_sensor_span', 120)
    nbins = params.get('nbins', 128)
    tdevice = params.get('device', 'cuda')
    lr = params.get('lr', 1e-4)
    src_type = params.get('source_type', 'planar')
    autodiff = params.get('autodiff', False)
    optim_iters = params.get("optim_iters", 300)
    record_iters = params.get("record_iters", optim_iters//10 + 1)

    h = vol_span / np.maximum(res_list[-1] - 1, 1)
    ds = h/step_res

    span = vol_span
    nviews = max(len(disp_ims), len(defl_ims))

    def gen_start_rays(samples=1):
        if src_type == 'planar':
            iv, rpv = source.rand_rays_in_sphere(
                nviews, (nbins, nbins), samples, span,
                angle_span=angle_s, circle=False, xaxis=False, sensor_dist=sensor_dist
            )
            tpv = torch.ones(iv[0].shape[0])
        elif src_type == 'point':
            iv, rpv = source.rand_ptrays_in_sphere(
                nviews, (nbins, nbins), samples, span,
                angle_span=angle_s, circle=False, xaxis=False, sensor_dist=sensor_dist
            )
            tpv = torch.ones(iv[0].shape[0])
        else:
            iv, _, tpv, rpv = source.rand_area_in_sphere(
                nviews, (nbins, nbins), samples, span,
                angle_span=angle_s, circle=False, xaxis=False, sensor_dist=sensor_dist
            )
        return [x.to(device=tdevice) for x in iv], rpv, tpv

    (x, v, planes), rpv, tpv = gen_start_rays(spp)

    def get_sensor_list(planes, rpv):
        sensor_n, sensor_p, sensor_t = [], [], []
        offset = 0
        for i in range(nviews):
            sensor_n.append(planes[None, offset, 1, :])
            sensor_t.append(planes[None, offset, 2, :])
            sensor_p.append(planes[None, offset, 0, :])  # + sensor_dist*sensor_n[-1])
            offset += rpv[i]
        return sensor_p, sensor_n, sensor_t

    loss_fn = torch.nn.MSELoss(reduction='mean')

    if autodiff:
        trace_fun = tracer.ADTracerC.apply
    else:
        trace_fun = tracer.BackTracerC.apply

    def trace(nt, rays):
        x, v = rays
        h = vol_span / np.maximum(nt.shape[0]-1, 1)
        xt, vt = trace_fun(nt, x, v, h, ds)
        return xt, vt

    n = params.get('init', torch.ones((res_list[0],)*3))

    def loss_function(eta):
        # n.requires_grad_ は元コード準拠で残す（optimizer側で使う可能性）
        n.requires_grad_(True)
        rays_ic, rpv, tpv = gen_start_rays(spp)
        sensor_p, sensor_n, sensor_t = get_sensor_list(rays_ic[2], rpv)

        x, v, planes = rays_ic
        xm, vm = trace(eta, (x, v))
        sn = planes[:, 1, :]
        sp = planes[:, 0, :]
        xmp, vmp = sensor.trace_rays_to_plane((xm, vm), (sp, sn))

        xm_s, vm_s = xmp.split(rpv), vmp.split(rpv)
        dists = (1/(tpv**2)).split(rpv)

        # --- Near (display) ---
        near_images = [sensor.generate_sensor((xv, vv), d, (sp, sn), nbins, span, st)
                       for xv, vv, sp, sn, st, d in zip(xm_s, vm_s, sensor_p, sensor_n, sensor_t, dists)]
        near_images = [source.sum_norm(ni) for ni in near_images]
        if sdf_loss and (sdf_disp[0] is not None):
            near_sdf = [sensor.get_sdf_vals_near((xv, vv), sdi, (sp, sn), span, st)
                        for xv, vv, sdi, sp, sn, st in zip(xm_s, vm_s, sdf_disp, sensor_p, sensor_n, sensor_t)]
            near_loss = sum([(sdi**2).sum() / sdi.numel() for sdi in near_sdf])
        elif disp_ims[0] is not None:
            near_loss = sum([loss_fn(im, meas) for im, meas in zip(near_images, disp_ims)]) / len(disp_ims)
        else:
            near_loss = 0

        # --- Far (deflection) -> 使わない時は完全スキップ ---
        far_loss = 0
        use_far = (sdf_loss and (sdf_defl[0] is not None)) or (defl_ims[0] is not None)
        if use_far:
            far_images = [sensor.generate_inf_sensor((xv, vv), 1, (sp, sn), nbins, far_sensor_span, st)
                          for xv, vv, sp, sn, st in zip(xm_s, vm_s, sensor_p, sensor_n, sensor_t)]
            far_images = [source.sum_norm(fi) for fi in far_images]
            if sdf_loss and (sdf_defl[0] is not None):
                far_sdf = [sensor.get_sdf_vals_far((xv, vv), sdi, (sp, sn), far_sensor_span, st)
                           for xv, vv, sdi, sp, sn, st in zip(xm_s, vm_s, sdf_defl, sensor_p, sensor_n, sensor_t)]
                far_loss = defl_weight * sum([(sdi**2).sum() / sdi.numel() for sdi in far_sdf])
            elif defl_ims[0] is not None:
                far_loss = defl_weight * sum([loss_fn(im, meas) for im, meas in zip(far_images, defl_ims)])

        loss = near_loss + far_loss

        # 後始末
        del xm, vm
        if 'far_images' in locals():
            del far_images
        del near_images
        del x, v, planes

        return loss

    def log_function(iter_count, eta):
        if iter_count % record_iters == 0 or iter_count == optim_iters-1:
            (x, v, planes), rpv, tpv = gen_start_rays(spp*2)
            sensor_p, sensor_n, sensor_t = get_sensor_list(planes, rpv)
            xm, vm = trace(eta, (x, v))
            xm_s, vm_s = xm.split(rpv), vm.split(rpv)
            dists = (1/(tpv**2)).split(rpv)

            images = [sensor.generate_sensor((xv, vv), d, (sp, sn), nbins, span, st)
                      for xv, vv, sp, sn, st, d in zip(xm_s, vm_s, sensor_p, sensor_n, sensor_t, dists)]
            images = [source.sum_norm(im) for im in images]
            plot_utils.save_multiple_images(images, os.path.join(result_dir, f'multiview_{iter_count}.png'))

            # 進捗の triplanar を軽量保存（必要ならコメント解除）
            # final_N = eta.shape[0]
            # voxel_size = float(span / max(final_N - 1, 1))
            # save_eta_triplanar(eta, os.path.join(result_dir, f"eta_triplanar_{iter_count:06d}.png"))

    # ===== マルチレゾ最適化呼び出し =====
    final_eta, loss_hist = optimizer.multires_opt(
        loss_function, n, optim_iters, res_list, log_function, lr=lr, statename='results/luneburg/result'
    )

    # 損失プロット
    plt.figure()
    plt.plot(loss_hist)
    plt.savefig(os.path.join(result_dir, 'loss_plot.png'))
    plt.close()

    # ----- 最終 η の保存＆可視化 -----
    final_N = final_eta.shape[0]  # (Z=Y=X=final_N) の等方格子を仮定
    voxel_size = float(span / max(final_N - 1, 1))
    save_eta_and_visuals(final_eta, result_dir, voxel_size, num_slices=8)

    return final_eta


# =========================================================
#  実験ランナー
# =========================================================

def run_multiview_exp():
    resolution = 128
    einstein_im = Image.open("data/einstein.png").resize((resolution, resolution))
    einstein_im = torch.from_numpy(np.asarray(einstein_im).astype(np.float32)).cuda()
    turing_im = Image.open("data/turing.png").resize((resolution, resolution))
    turing_im = torch.from_numpy(np.asarray(turing_im).astype(np.float32)).cuda()

    disp_images = [
        source.sum_norm(einstein_im),
        source.sum_norm(turing_im)
    ]
    params = dict(
        disp_ims=disp_images,
        optim_iters=10,
        record_iters=10
    )

    multires_opt(params, 'results/multiview')


def run_singleview_exp():
    resolution = 128
    # 任意の1枚（例：einstein）
    target_im = Image.open("data/einstein.png").resize((resolution, resolution))
    target_im = torch.from_numpy(np.asarray(target_im).astype(np.float32)).cuda()

    disp_images = [source.sum_norm(target_im)]
    params = dict(
        disp_ims=disp_images,   # 1枚だけ渡す
        defl_ims=[None],        # deflectionは使わない
        optim_iters=300,
        record_iters=30
    )

    multires_opt(params, 'results/singleview')


if __name__ == '__main__':
    # CUDA が無い環境でも動くようフォールバック（必要なければ元の1行に戻してOK）
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        print("[Info] CUDA not available. Using CPU float tensors.")
    # 単一画像の最適化
    run_singleview_exp()
    # 複数画像の実験を動かす場合は下を使用
    # run_multiview_exp()