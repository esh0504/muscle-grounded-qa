#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 평가 — 학습된 회귀 헤드로 Set-1 500 mesh 의 Muscle-EM · Activation MAE.

  python eval_s1.py                                   # 3D + 2D 렌더 + 기준선
  python eval_s1.py evaluators.rows='[mesh3d]'        # 3D 만
  python eval_s1.py evaluators.threshold.mode=fixed evaluators.threshold.value=0.05

`eval.py` 가 Stage-2(LLM) 를 채점한다면 이쪽은 Stage-1 모델의 11-D 활성 출력을 직접
채점한다. 정의는 `metrics/muscle_regression.py`, 절차는 `evaluators/stage1_muscle.py`.

산출물: {evaluators.output_dir}/ (metrics.json · preds.jsonl · summary.md)
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from evaluators import find_evaluator_def


@hydra.main(version_base=None, config_path="configs", config_name="eval_s1")
def main(cfg: DictConfig):
    print("Config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    Evaluator = find_evaluator_def(cfg.evaluators.name, cfg.evaluators.class_name)
    evaluator = Evaluator(cfg.evaluators, experiment_cfg=cfg)

    print(f"[eval_s1] evaluator={type(evaluator).__name__}  "
          f"output_dir={cfg.evaluators.output_dir}")
    evaluator.run()
    return 0


if __name__ == "__main__":
    main()
