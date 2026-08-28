"""보일러플레이트 수치를 걸러내기 위한 document-frequency 표.

`number` 카테고리는 Mask F1 에서 유일하게 mesh 를 봐야만 맞출 수 있는 카테고리다.
그런데 답변에는 mesh 와 무관한 상수 수치가 섞여 있다 — A1 답변의 100% 가 범례
("cl_t=0.84, 0=앞·1=뒤")의 `0`·`1` 을 포함하고, C1 답변의 100% 가 `1.0`(부피비)을 포함한다.
이런 값은 상수 문자열만 뱉어도 맞으므로 `number` F1 을 부풀린다.

여기서는 **train split** 에서 turn_type 별 수치의 document frequency 를 세어
`df >= min_df` 인 값을 보일러플레이트로 지정한다. 표는 파일로 고정해 재현성을 확보한다
(평가 때마다 다시 세면 평가셋에 따라 지표 정의가 흔들린다).

  python -m metrics.number_df --lang ko --out DATA/qa/number_df_ko.json

⚠️ **표본이 작은 turn_type 은 표에서 뺀다** (`min_turns`, 2026-08-03 추가).
df 해상도는 1/n 이라 n 이 작으면 우연히 두어 번 겹친 값이 임계를 넘는다. 실측:
`max_per_file=60` 이던 시절의 B3 는 n=60 이라 `1.66 · 2.59 · 4.51` 같은 **mm 변위값**
(= 지표가 재야 할 mesh 의존 수치 그 자체)이 df=0.050(3/60)으로 보일러플레이트가 됐다.
B1 은 train split 전체에 29턴뿐이라 상한을 아무리 올려도 안정되지 않는다 — 이런 turn_type 은
표에서 빼고, `boilerplate()` 가 빈 집합을 돌려주게 둔다(= 보정 안 함). 잘못 보정하느니 낫다.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

DEFAULT_MIN_DF = 0.05
# turn_type 당 최소 턴 수. 이보다 적으면 df 추정이 불안정해 표에서 뺀다.
# 200 이면 min_df=0.05 가 "10턴 이상"을 뜻해 우연 일치와 구분된다.
DEFAULT_MIN_TURNS = 200
# 파일당 상한. B3 는 파일이 1개뿐(train·variant0 기준 1,018턴)이라 이 값이 곧 B3 의 표본
# 크기가 된다. 옛 기본값 60 은 B3 를 n=60 으로 잘라 위의 오판정을 만들었다. 전체 소요 ~15초.
DEFAULT_MAX_PER_FILE = 1200


def path_for(lang: str) -> Path:
    return Path("DATA/qa") / f"number_df_{lang}.json"


def load(lang: str) -> dict | None:
    """{"min_df": float, "df": {turn_type: {value: df}}} 또는 표가 없으면 None."""
    fp = path_for(lang)
    if not fp.is_file():
        return None
    return json.loads(fp.read_text(encoding="utf-8"))


def boilerplate(table: dict | None, turn_type: str) -> set[str]:
    """해당 turn_type 에서 걸러낼 수치 값 집합."""
    if not table:
        return set()
    min_df = float(table.get("min_df", DEFAULT_MIN_DF))
    per_tt = (table.get("df") or {}).get(str(turn_type) or "", {})
    return {v for v, d in per_tt.items() if float(d) >= min_df}


def build(qa_glob: str, lang: str, *, split_dir: str = "DATA/mesh",
          min_df: float = DEFAULT_MIN_DF, max_per_file: int = DEFAULT_MAX_PER_FILE,
          min_turns: int = DEFAULT_MIN_TURNS) -> dict:
    """train split 레코드에서 turn_type 별 수치 df 를 센다.

    파일마다 같은 몫(`max_per_file`)을 가져온다. 앞에서부터 자르면 파일 정렬 때문에
    A1 만 뽑히고 physics_chain 이 통째로 빠진다.

    `n_turns < min_turns` 인 turn_type 은 `df` 에서 빼고 `dropped_turn_types` 에 남긴다
    (모듈 docstring 참고). 뺀 turn_type 은 보일러플레이트 보정 없이 채점된다.
    """
    from datasets.split_trainvaltest import load_split_indices
    from metrics.spans import facts_from_spans

    train = set(load_split_indices(Path(split_dir), "train"))
    n_turns: collections.Counter = collections.Counter()
    hits: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    seen = 0
    for fp in sorted(glob.glob(qa_glob)):
        took = 0
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if took >= max_per_file:
                    break
                rec = json.loads(line)
                if rec.get("variant") not in (None, 0):
                    continue
                ref = rec.get("mesh_ref") or {}
                if "indices" in ref:
                    idx = int(ref["indices"][0])
                else:
                    idx = int(ref.get("verts_shard", 0)) * 1000 + int(ref.get("row_in_shard", 0))
                if idx not in train:
                    continue
                seen += 1
                took += 1
                tts = rec.get("turn_types") or []
                k = 0
                for turn in rec.get("conversations", []):
                    if turn.get("from") != "gpt":
                        continue
                    tt = tts[k] if k < len(tts) else ""
                    k += 1
                    nums = facts_from_spans(turn.get("mask_spans", []), lang)["number"]
                    n_turns[tt] += 1
                    for v in nums:                     # 집합이므로 turn 당 1회 = document freq
                        hits[tt][v] += 1

    df = {tt: {v: c / n_turns[tt] for v, c in cnt.items() if c / n_turns[tt] >= min_df}
          for tt, cnt in hits.items() if n_turns[tt] >= min_turns}
    dropped = {tt: n_turns[tt] for tt in hits if n_turns[tt] < min_turns}
    return {"lang": lang, "min_df": min_df, "min_turns": min_turns,
            "n_turns": dict(n_turns), "dropped_turn_types": dropped,
            "qa_glob": qa_glob, "df": df}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--qa-glob", default=None)
    ap.add_argument("--min-df", type=float, default=DEFAULT_MIN_DF)
    ap.add_argument("--max-per-file", type=int, default=DEFAULT_MAX_PER_FILE)
    ap.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    qa_glob = a.qa_glob or f"DATA/qa/{a.lang}/nat_out/nat_*.jsonl"
    table = build(qa_glob, a.lang, min_df=a.min_df, max_per_file=a.max_per_file,
                  min_turns=a.min_turns)
    out = Path(a.out or path_for(a.lang))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[number_df] {out}  (min_df={a.min_df}, min_turns={a.min_turns}, "
          f"{sum(table['n_turns'].values())} turns)")
    for tt in sorted(table["df"]):
        vals = sorted(table["df"][tt].items(), key=lambda kv: -kv[1])[:6]
        print(f"  {tt:14s} n={table['n_turns'][tt]:6d} 보일러플레이트 {len(table['df'][tt]):3d}개"
              f"  상위: {', '.join(f'{v}({d:.2f})' for v, d in vals)}")
    for tt, n in sorted(table["dropped_turn_types"].items()):
        print(f"  {tt:14s} n={n:6d} → 표본 부족으로 제외 (보정 없이 채점된다)")


if __name__ == "__main__":
    main()
