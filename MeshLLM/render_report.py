"""Set 3 사람 검토용 리포트 — 메쉬 렌더 + 질문 + 모델 답변을 한 장에 담는다.

진단형(Set 3)은 자동 채점이 프록시뿐이라 결국 사람이 본다. 그때 메쉬를 따로 띄워
비교하는 건 느리므로, 항목마다 PNG 한 장(렌더 2뷰 + 질문 + 예측 + GT 근거)을 굽고
index.html 로 묶는다.

렌더는 matplotlib 3D trisurf 이고 색은 rest 대비 변위 크기다. 외부 렌더러가 필요 없다.
"""

from __future__ import annotations

import html
import textwrap
from pathlib import Path

import numpy as np

VIEWS = [(12, -75, "lateral"), (28, -35, "oblique")]
# 한국어 QA 를 그림 안에 넣으려면 CJK 폰트가 필요하다 (없으면 □ 로 깨진다).
# 저장소 동봉 폰트를 먼저 보고, 없으면 시스템 폰트를 찾고, 그래도 없으면 기본 폰트로 간다.
FONT_CANDIDATES = [
    Path(__file__).resolve().parent / "assets" / "NotoSansKR.ttf",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
]
_font_ready = False


def _ensure_font():
    """CJK 폰트를 matplotlib 에 등록한다. 등록된 폰트 이름 또는 None."""
    global _font_ready
    if _font_ready:
        return _font_ready if isinstance(_font_ready, str) else None
    import matplotlib
    from matplotlib import font_manager as fm
    for fp in FONT_CANDIDATES:
        if fp.is_file():
            fm.fontManager.addfont(str(fp))
            name = fm.FontProperties(fname=str(fp)).get_name()
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            _font_ready = name
            return name
    print("[render][경고] CJK 폰트가 없어 한국어가 깨질 수 있다 "
          "(assets/NotoSansKR.ttf 를 두면 해결)")
    _font_ready = True
    return None


def _wrap(text: str, width: int = 100, max_lines: int = 14) -> str:
    lines: list[str] = []
    for para in (text or "").strip().splitlines():
        lines.extend(textwrap.wrap(para, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["… (생략)"]
    return "\n".join(lines)


def render_item(rest: np.ndarray, faces: np.ndarray, disp: np.ndarray, *,
                question: str, pred: str, grounding: dict | None,
                title: str, out_png: Path) -> Path:
    """메쉬 1개 + 텍스트를 PNG 한 장으로."""
    import matplotlib
    matplotlib.use("Agg")
    _ensure_font()
    import matplotlib.pyplot as plt

    verts = rest + disp
    mag = np.linalg.norm(disp, axis=-1)
    face_mag = mag[faces].mean(axis=1)
    vmin, vmax = float(face_mag.min()), float(face_mag.max())

    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.0, wspace=0.0)
    fig.suptitle(title, fontsize=12.5, y=0.985)

    tri = None
    for i, (elev, azim, name) in enumerate(VIEWS):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        tri = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
                              cmap="viridis", linewidth=0.15, edgecolor="none",
                              antialiased=True)
        tri.set_array(face_mag)
        # clim 을 안 잡으면 기본 (0,1) 이라 변위(≈0.01)가 전부 최저색으로 뭉개진다
        tri.set_clim(vmin, vmax if vmax > vmin else vmin + 1e-6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} view", fontsize=9.5)
        ax.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), np.ptp(verts[:, 2])))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    if tri is not None:
        # 전용 축에 놓는다. ax=[...] 로 붙이면 오른쪽 메쉬 위를 덮는다.
        cax = fig.add_axes((0.945, 0.50, 0.011, 0.30))
        cb = fig.colorbar(tri, cax=cax)
        cb.set_label("|displacement|", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # 아래: 질문 / 예측 / GT 근거
    # 주의: family="monospace" 를 주면 등록한 CJK 폰트가 무시돼 한글이 □ 로 깨진다.
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    blocks = [("Q", _wrap(question, 104, 5)), ("예측", _wrap(pred, 104, 12))]
    if grounding:
        g = grounding
        blocks.append(("GT", _wrap(
            f"target /{g.get('target','?')}/    올려야: {', '.join(g.get('should_increase', [])) or '-'}"
            f"    내려야: {', '.join(g.get('should_decrease', [])) or '-'}", 104, 3)))
    y = 1.02
    for label, body in blocks:
        nline = body.count("\n") + 1
        ax.text(0.0, y, f"[{label}]", fontsize=10, va="top", color="#0b5")
        ax.text(0.055, y, body, fontsize=9, va="top", linespacing=1.45)
        y -= 0.06 + 0.105 * nline
    fig.subplots_adjust(left=0.02, right=0.93, top=0.95, bottom=0.01)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def build_index(entries: list[dict], out_html: Path, title: str) -> Path:
    """PNG 목록 → 한 페이지. 판정 메모를 적을 수 있게 항목마다 자리를 둔다."""
    rows = []
    for e in entries:
        rows.append(f"""
  <section>
    <h2>#{e['index']} <small>{html.escape(str(e.get('section','')))}
        · target /{html.escape(str(e.get('target','')))}/</small></h2>
    <img src="{html.escape(e['png'])}" loading="lazy">
    <details><summary>텍스트로 보기</summary>
      <p><b>Q</b> {html.escape(e.get('question',''))}</p>
      <p><b>예측</b> {html.escape(e.get('pred',''))}</p>
      <p><b>GT</b> 올려야 {html.escape(', '.join(e.get('should_increase', [])))} /
         내려야 {html.escape(', '.join(e.get('should_decrease', [])))}</p>
    </details>
  </section>""")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body{{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;
       line-height:1.5;color:#111;background:#fafafa}}
 section{{background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:1rem;margin:1.2rem 0}}
 h1{{font-size:1.4rem}} h2{{font-size:1.05rem;margin:.2rem 0 .6rem}}
 small{{font-weight:400;color:#666}}
 img{{width:100%;height:auto;border-radius:4px}}
 details{{margin-top:.6rem;font-size:.9rem}} summary{{cursor:pointer;color:#555}}
 p{{margin:.35rem 0}}
</style>
<h1>{html.escape(title)}</h1>
<p>총 {len(entries)}개. 각 항목은 메쉬 2뷰(색=rest 대비 변위 크기) + 질문 + 모델 예측 + GT 근거.</p>
{''.join(rows)}
""", encoding="utf-8")
    return out_html
