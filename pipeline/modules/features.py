# -*- coding: utf-8 -*-
"""
정중시상 혀(상단 표면) + 경구개 윤곽에서 '화자 불변' 조음 특징 벡터를 추출한다.
raw 윤곽 MSE는 화자 해부학이 지배 → 대신 팔라트 기준 정규화 특징으로 비교한다.

입력: tongue(Nx2, 앞→뒤 상단표면), ref_palate(Mx2, 고정 기준 지붕=모델 팔라트).
      (MRI측은 먼저 자기 팔라트를 ref_palate에 Procrustes 정합한 뒤 혀를 넘긴다.)
모든 길이는 팔라트 호길이 L로 나눠 정규화, 위치는 팔라트 x-범위 상의 t∈[0,1](앞0..뒤1).

특징(≈32개):
  거리함수/협착:  cd_min, cl_t, gap_mean, gap_std, gap_front/mid/back, constr_width, df_0..df_7
  혀 랜드마크:    tip_xn, tip_z, peak_xn, peak_z, h_front/mid/back, mean_z, com_xn, com_z
  형상/곡률:      arc_len, tilt, ant_slope, post_slope, curv_peak, doming
2D 정중시상 한계: /s/ 홈폭·/l/ 측면은 이 특징으로 정의 불가(문헌 근거만; Tier D).
"""
import numpy as np

NST = 15          # 거리함수 스테이션 수
NDF = 8           # 특징벡터에 넣을 거리함수 샘플 수


# ---------- Procrustes 유사변환(scale+rot+trans): src→dst ----------
def umeyama(src, dst):
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = np.sum(d) / var_s if var_s > 0 else 1.0
    t = mu_d - s * (R @ mu_s)
    return lambda P: (s * (np.asarray(P, float) @ R.T) + t)


def align_tongue(tongue, palate, ref_palate, n=40):
    """MRI 팔라트를 ref(모델)팔라트에 맞추는 변환으로 혀를 ref 좌표계로.
    두 팔라트 점 개수가 달라도 되도록 호길이로 n점 리샘플 후 대응 정합."""
    pal = np.asarray(palate, float); ref = np.asarray(ref_palate, float)
    ps = _resample_by_arc(pal[np.argsort(pal[:, 0])], n)     # 앞→뒤 정렬 후 균등 리샘플
    rs = _resample_by_arc(ref[np.argsort(ref[:, 0])], n)
    T = umeyama(ps, rs)
    return T(np.asarray(tongue, float))


# ---------- 기하 헬퍼 ----------
def _resample_by_arc(poly, n):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, 0)
    u = np.linspace(0, s[-1], n)
    x = np.interp(u, s, poly[:, 0]); z = np.interp(u, s, poly[:, 1])
    return np.c_[x, z]


def _pt_seg_dist(p, a, b):
    ab = b - a; t = 0.0
    denom = ab @ ab
    if denom > 0:
        t = np.clip((p - a) @ ab / denom, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def _pt_poly_dist(p, poly):
    return min(_pt_seg_dist(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def _tongue_zx(tongue, xn_grid, px0, xspan):
    """정규화 x(xn)에서 혀 높이 z (ref 프레임 z 그대로, 나중에 /L)."""
    tg = tongue[np.argsort(tongue[:, 0])]
    xs = (tg[:, 0] - px0) / xspan
    zs = tg[:, 1]
    # 중복 xn 평균(단조 x 보장)
    xu, inv = np.unique(xs, return_inverse=True)
    if len(xu) < len(xs):
        zs = np.array([zs[inv == i].mean() for i in range(len(xu))])
        xs = xu
    return np.interp(xn_grid, xs, zs, left=zs[0], right=zs[-1])


# ---------- 특징 추출 ----------
FEATURE_KEYS = (
    ["cd_min", "cl_t", "gap_mean", "gap_std", "gap_front", "gap_mid", "gap_back", "constr_width"]
    + [f"df_{i}" for i in range(NDF)]
    + ["tip_xn", "tip_z", "peak_xn", "peak_z", "h_front", "h_mid", "h_back", "mean_z", "com_xn", "com_z"]
    + ["arc_len", "tilt", "ant_slope", "post_slope", "curv_peak", "doming"]
)


def extract_features(tongue, ref_palate):
    tongue = np.asarray(tongue, float)
    pal = ref_palate[np.argsort(ref_palate[:, 0])]
    seg = np.linalg.norm(np.diff(pal, axis=0), axis=1)
    L = float(seg.sum()) or 1.0
    px0, px1 = pal[0, 0], pal[-1, 0]
    pz0 = pal[0, 1]
    xspan = (px1 - px0) or 1.0

    # --- 거리함수 d(t): 팔라트 스테이션 → 혀 최근접거리 ---
    pal_st = _resample_by_arc(pal, NST)
    tg_poly = tongue[np.argsort(tongue[:, 0])]
    d = np.array([_pt_poly_dist(pal_st[i], tg_poly) for i in range(NST)]) / L
    t_axis = (pal_st[:, 0] - px0) / xspan
    cd_i = int(np.argmin(d))
    cd_min = float(d[cd_i]); cl_t = float(np.clip(t_axis[cd_i], 0, 1))
    front = d[t_axis < 1 / 3]; mid = d[(t_axis >= 1 / 3) & (t_axis < 2 / 3)]; back = d[t_axis >= 2 / 3]
    constr_width = float(np.mean(d < 1.5 * cd_min + 1e-9))
    df_samp = np.interp(np.linspace(0, 1, NDF), np.linspace(0, 1, NST), d)

    # --- 혀 랜드마크 (z는 팔라트 앞점 기준 상대, /L) ---
    zg = _tongue_zx(tongue, np.linspace(0, 1, 60), px0, xspan)
    zrel = (zg - pz0) / L
    xn = np.linspace(0, 1, 60)
    tg = tongue[np.argsort(tongue[:, 0])]
    tip_xn = float((tg[0, 0] - px0) / xspan)
    tip_z = float((tg[0, 1] - pz0) / L)
    peak_i = int(np.argmax(zrel))
    peak_xn = float(xn[peak_i]); peak_z = float(zrel[peak_i])
    h_front = float(np.interp(0.25, xn, zrel)); h_mid = float(np.interp(0.5, xn, zrel)); h_back = float(np.interp(0.75, xn, zrel))
    mean_z = float(zrel.mean())
    com_xn = float(((tg[:, 0] - px0) / xspan).mean()); com_z = float(((tg[:, 1] - pz0) / L).mean())

    # --- 형상/곡률 ---
    arc = float(np.linalg.norm(np.diff(tg, axis=0), axis=1).sum() / L)
    tilt = float(np.polyfit(xn, zrel, 1)[0])
    ant = zrel[:peak_i + 1]; post = zrel[peak_i:]
    ant_slope = float(np.polyfit(xn[:peak_i + 1], ant, 1)[0]) if peak_i >= 1 else 0.0
    post_slope = float(np.polyfit(xn[peak_i:], post, 1)[0]) if peak_i <= 58 else 0.0
    dz = np.gradient(zrel, xn); d2z = np.gradient(dz, xn)
    curv = np.abs(d2z) / (1 + dz ** 2) ** 1.5
    curv_peak = float(np.nanmax(curv))
    chord = np.linspace(zrel[0], zrel[-1], 60)
    doming = float(np.max(zrel - chord))

    vals = [cd_min, cl_t, float(front.mean() if len(front) else np.nan),
            float(d.std()), float(front.mean() if len(front) else 0),
            float(mid.mean() if len(mid) else 0), float(back.mean() if len(back) else 0),
            constr_width] + list(df_samp) + [
            tip_xn, tip_z, peak_xn, peak_z, h_front, h_mid, h_back, mean_z, com_xn, com_z,
            arc, tilt, ant_slope, post_slope, curv_peak, doming]
    # gap_mean 자리 교정(위에서 front.mean 잘못 넣음)
    vals[2] = float(d.mean())
    return dict(zip(FEATURE_KEYS, vals))


# ---------- self-test ----------
def _synth(kind):
    """합성 팔라트+혀. kind: 'i'(전방고설), 'a'(저설후방), 'u'(후방고설)."""
    x = np.linspace(0, 1, 40)
    palate = np.c_[x, 1.0 - 0.12 * x - 0.05 * np.sin(np.pi * x)]  # 살짝 볼록한 지붕
    if kind == "i":
        z = 0.55 + 0.30 * np.exp(-((x - 0.25) ** 2) / 0.03)     # 앞쪽 융기, 지붕 근접
    elif kind == "a":
        z = 0.45 + 0.10 * np.exp(-((x - 0.7) ** 2) / 0.05)      # 낮고 뒤쪽 약간
    elif kind == "u":
        z = 0.55 + 0.28 * np.exp(-((x - 0.72) ** 2) / 0.03)     # 뒤쪽 융기
    return np.c_[x, z], palate


def self_test():
    ref_pal = _synth("i")[1]
    for k in ["i", "a", "u"]:
        tg, pal = _synth(k)
        tg_al = align_tongue(tg, pal, ref_pal)
        f = extract_features(tg_al, ref_pal)
        print(f"[{k}] CL={f['cl_t']:.2f} CD={f['cd_min']:.3f} peak_xn={f['peak_xn']:.2f} "
              f"peak_z={f['peak_z']:.2f} doming={f['doming']:.3f} h_front={f['h_front']:.2f} h_back={f['h_back']:.2f}")
    print(f"특징 개수: {len(FEATURE_KEYS)}")
    # 기대: i→CL 앞(작은 t)·CD 작음, a→CD 큼, u→CL 뒤(큰 t)
    assert len(FEATURE_KEYS) == len(extract_features(*[_synth('i')[0], _synth('i')[1]]))


if __name__ == "__main__":
    self_test()
