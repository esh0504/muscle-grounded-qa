"""Stage-2 학습 중 검증 지표 집계.  [Stage-2 val]

생성 루프(배치 만들기·generate_answer·디코딩)와 turn_type 층화 추출은 모델·토크나이저가
필요하므로 트레이너에 남기고, 여기서는 **생성 결과만 받아** wandb 로 그대로 넘길 수 있는
`val/*` dict 를 만든다. (원본: train_s2.py validate())
"""

from __future__ import annotations

from metrics import abstention_f1, mask_f1


def score_generations(preds, golds, turn_types, abstain_flags, lang):
    """생성 결과 → {"val/mask_f1": ..., "val/abstention_f1": ..., ...} dict.

    preds         생성한 답변 문자열 리스트
    golds         항목별 gold answer span 리스트
    turn_types    항목별 대상 턴의 turn_type ("" 가능)
    abstain_flags 항목별 "기권해야 하는가" bool
    lang          "ko" | "en" (span 추출·기권 표현이 언어별로 다르다)

    원본 train_s2.py validate() 의 채점 블록을 그대로 옮긴 것이다.
    per_category F1 도 val/mask_f1_<cat> 로 펼친다.
    """
    out: dict = {}
    mk = mask_f1.score([{"gold_spans": s, "pred": p, "turn_type": t}
                        for s, p, t in zip(golds, preds, turn_types)], lang)
    ab = abstention_f1.score(
        [{"pred": p, "should_abstain": a} for p, a in zip(preds, abstain_flags)], lang)
    out.update({
        "val/mask_f1": mk["micro"]["f1"],
        "val/mask_precision": mk["micro"]["precision"],
        "val/mask_recall": mk["micro"]["recall"],
        "val/abstention_f1": ab["f1"],
        "val/abstain_rate": ab["abstain_rate"],
        "val/n_gen_samples": len(preds),
        "val/pred_chars": sum(len(p) for p in preds) / max(len(preds), 1),
    })
    for c, v in (mk["per_category"] or {}).items():
        if v is not None:
            out[f"val/mask_f1_{c}"] = v["f1"]
    out["_sample"] = preds[0][:300] if preds else ""
    return out
