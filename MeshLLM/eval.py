#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 평가 — Config만 넣으면 best checkpoint로 추론·채점·렌더.

기본: Set-1 probe (SET1_PROBE_SPEC.md / SET1_METRICS_DETAIL.md)
  python eval.py +experiment=ours_en
  python eval.py +experiment=ours_ko evaluators=set1_probe_ko

옛 unseen 3-세트:
  python eval.py +experiment=ours_en evaluators=unseen

산출물: <output_dir>/eval/  (= {output})
  preds.jsonl · metrics.json · summary.json · index.html
  pred/render/{testdata_index}.png
      정면 Mesh + Q + A + (reference_answer) 큰 글씨
  pred/auto_metric_json/
      muscle_set.json
      value_set.json
      direction_set.json
      abstention_set.json
  pred/judge_metric_json/gpt/
      imgs_preds.json · mesh_preds.json   (NO reference — GPT 업로드용)
      reference_answer.json               (offline only — GPT에 올리지 말 것)
  pred/judge_metric_json/human/{testdata_index}.png
      utility 항목: 정면 Mesh + Q + A + (reference_answer)
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from evaluators import find_evaluator_def


@hydra.main(version_base=None, config_path="configs", config_name="eval_s2")
def main(cfg: DictConfig):
    print("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    Evaluator = find_evaluator_def(cfg.evaluators.name, cfg.evaluators.class_name)
    evaluator = Evaluator(cfg.evaluators, experiment_cfg=cfg)

    print(f"[eval] evaluator={type(evaluator).__name__}  "
          f"experiment={cfg.get('name', '')}  "
          f"output_dir={cfg.get('output_dir')}")
    evaluator.run()
    return 0


if __name__ == "__main__":
    main()
