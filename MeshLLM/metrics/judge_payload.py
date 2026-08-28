"""GPT-judge 페이로드 생성.  [Set 3 주 지표]

여기서는 페이로드(JSONL)만 만든다. 실제 호출은 API 키·비용이 걸려 있어 분리했다.
룹릭·모델·temperature 는 configs/eval/judge.yaml 과 이 파일이 단일 출처이며,
부록에 그대로 인용할 수 있게 문자열을 고정해 둔다.

judge 입력에 **정답 문장은 넣지 않는다** — 넣으면 문체 유사도로 판정해 버린다.
대신 GT 근거(현재/타깃 근육 벡터, 올려야/내려야 할 근육)만 준다.
"""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_SYSTEM = (
    "You are grading a model's diagnostic explanation of a simulated tongue shape. "
    "You are given the ground-truth muscle activations, the intended target phoneme, and which "
    "muscles should increase or decrease to reach it. Judge whether the model's answer is "
    "CORRECT, PARTIAL, or WRONG. "
    "Rules: (1) reward identifying the right corrective direction (which muscles up/down). "
    "(2) If the shape-to-muscle cause is genuinely ambiguous, a hedged/abstaining answer is "
    "acceptable, but asserting a single wrong specific cause is WRONG. "
    "(3) Ignore style; grade only factual/directional correctness. "
    'Output strict JSON: {"verdict":"CORRECT|PARTIAL|WRONG","reason":"..."}.'
)


def build(items: list[dict], out_path: str | Path, *, model: str = "gpt-4o",
          temperature: float = 0.0, n_votes: int = 3) -> dict:
    """items: [{"index", "question", "pred", "grounding"}] → judge 입력 JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fo:
        for it in items:
            g = it["grounding"]
            user = (
                f"Target phoneme: /{g['target']}/\n"
                f"Ground-truth current muscle activations: {json.dumps(g['current_vec'])}\n"
                f"Target muscle pattern: {json.dumps(g['target_vec'])}\n"
                f"Should INCREASE: {g['should_increase']}  "
                f"Should DECREASE: {g['should_decrease']}\n"
                f"Question: {it['question']}\n"
                f"Model answer: {it['pred']}\n\nGrade it."
            )
            fo.write(json.dumps({
                "index": it["index"],
                "model": model, "temperature": temperature, "n_votes": n_votes,
                "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                             {"role": "user", "content": user}],
            }, ensure_ascii=False) + "\n")
            n += 1
    return {"n_payloads": n, "path": str(out_path), "judge_model": model,
            "temperature": temperature, "n_votes": n_votes,
            "note": "실제 판정은 별도 실행. 사람 50쌍과 κ 비교 필요."}
