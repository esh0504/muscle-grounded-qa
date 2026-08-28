"""리팩토링으로 옮긴 모듈이 전부 import 되는지 + 엔트리포인트가 얇은지.

GPU·LLM·체크포인트를 건드리지 않는다. 팩토리는 **클래스만 돌려받고 인스턴스화하지
않는다** (Stage2Model 은 생성자에서 Qwen3-8B 를 내려받는다).
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# 명세 §1 의 최종 파일 배치에서 새로 생기거나 갈라진 모듈 전부.
NEW_MODULES = [
    "models.stage2",
    "models.stage2.mesh_encoder",
    "models.stage2.bridge",
    "models.stage2.model_s2",
    "losses.lm_loss",
    "datasets.mesh_store",
    "metrics.stage2_val",
    "trainers.trainer_s2",
    "evaluators",
    "evaluators.unseen",
    "evaluators.set1_probe",
    "evaluators.stage1_muscle",
    "metrics.set1_probe",
    "metrics.muscle_regression",
    "utils",
    "train_s2",
    "eval",
    "eval_s1",
]


@pytest.mark.parametrize("module_name", NEW_MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


# 옮긴 심볼이 실제로 그 모듈에 있는지 (이동표 §2 기준).
MOVED_SYMBOLS = [
    ("models.stage2.mesh_encoder", "compute_feature_stats"),
    ("models.stage2.mesh_encoder", "FrozenMeshEncoder"),
    ("models.stage2.bridge", "FusionLayer"),
    ("models.stage2.bridge", "CrossAttentionFusion"),
    ("models.stage2.bridge", "MlpBridge"),
    ("models.stage2.bridge", "splice_prefix"),       # 원본 _splice_prefix 의 public 이름
    ("models.stage2.model_s2", "Stage2Model"),
    ("losses.lm_loss", "weighted_causal_lm_loss"),
    ("losses.lm_loss", "WeightedLmLoss"),
    ("datasets.mesh_store", "MeshStore"),
    ("datasets.mesh_store", "apply_mesh_control"),
    ("datasets.mesh_store", "MESH_CONTROLS"),
    ("metrics.stage2_val", "score_generations"),
    ("trainers.trainer_s2", "TrainerS2"),
    ("trainers.trainer_s2", "dist_setup"),
    ("trainers.trainer_s2", "grad_norm"),
    ("evaluators.unseen", "EvaluatorUnseen"),
    ("evaluators.set1_probe", "EvaluatorSet1Probe"),
    ("metrics.set1_probe", "score_all"),
    ("evaluators.stage1_muscle", "EvaluatorStage1Muscle"),
    ("metrics.muscle_regression", "score_muscle_em"),
    ("metrics.muscle_regression", "score_activation_mae"),
    ("utils", "move"),
    ("utils", "write_jsonl"),
]


@pytest.mark.parametrize("module_name,symbol", MOVED_SYMBOLS)
def test_moved_symbol_exists(module_name, symbol):
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name}.{symbol} 가 없다"


def test_private_splice_prefix_is_gone():
    """`_splice_prefix` → `splice_prefix` rename (명세 §2)."""
    bridge = importlib.import_module("models.stage2.bridge")
    assert not hasattr(bridge, "_splice_prefix")


# --------------------------------------------------------------------------- #
# 엔트리포인트가 다시 뚱뚱해지는 회귀 방지
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entrypoint", ["train_s2.py", "eval.py", "eval_s1.py"])
def test_entrypoint_is_thin(entrypoint):
    """원래 1330/570줄이었다. @hydra.main + main() 만 남아야 한다 (명세 §7).

    독스트링을 뺀 **코드 줄 수**로 잰다. 파이프라인 설명 독스트링은 길어도 되지만
    로직이 돌아오면 안 된다. 상한 40 은 여유를 준 값이다 (현재 train_s2 는 ~30줄).
    """
    src = (REPO / entrypoint).read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    n_doc = len(doc.splitlines()) + 2 if doc else 0
    n_code = len([l for l in src.splitlines() if l.strip()]) - n_doc
    assert n_code < 40, f"{entrypoint} 의 코드가 {n_code}줄이다 — 엔트리포인트가 다시 두꺼워졌다"


@pytest.mark.parametrize("entrypoint", ["train_s2.py", "eval.py", "eval_s1.py"])
def test_entrypoint_has_hydra_main(entrypoint):
    src = (REPO / entrypoint).read_text(encoding="utf-8")
    assert "@hydra.main(" in src
    assert "argparse" not in src, f"{entrypoint} 에 argparse 가 남아 있다"


# --------------------------------------------------------------------------- #
# find_*_def 팩토리 (CasMVSNet 관례). 클래스만 확인하고 **인스턴스화하지 않는다**.
# --------------------------------------------------------------------------- #
def test_find_model_def():
    from models import find_model_def
    cls = find_model_def("stage2.model_s2", "Stage2Model")
    assert isinstance(cls, type) and cls.__name__ == "Stage2Model"


def test_find_loss_def():
    from losses import find_loss_def
    cls = find_loss_def("lm_loss", "WeightedLmLoss")
    assert isinstance(cls, type) and cls.__name__ == "WeightedLmLoss"


def test_find_trainer_def():
    from trainers import find_trainer_def
    cls = find_trainer_def("trainer_s2", "TrainerS2")
    assert isinstance(cls, type) and cls.__name__ == "TrainerS2"


def test_find_dataset_def():
    from datasets import find_dataset_def
    cls = find_dataset_def("qa_dataset", "MeshQaDataset")
    assert isinstance(cls, type) and cls.__name__ == "MeshQaDataset"


def test_find_evaluator_def():
    from evaluators import find_evaluator_def
    cls = find_evaluator_def("unseen", "EvaluatorUnseen")
    assert isinstance(cls, type) and cls.__name__ == "EvaluatorUnseen"
    cls2 = find_evaluator_def("set1_probe", "EvaluatorSet1Probe")
    assert isinstance(cls2, type) and cls2.__name__ == "EvaluatorSet1Probe"


# 설정 파일의 name/class_name 이 실제 모듈/클래스와 맞는지 — eval.py 는
# find_evaluator_def(cfg.evaluators.name, cfg.evaluators.class_name) 로 부른다.
GROUP_FACTORIES = [
    ("models", "models", "find_model_def"),
    ("losses", "losses", "find_loss_def"),
    ("trainers", "trainers", "find_trainer_def"),
    ("datasets", "datasets", "find_dataset_def"),
    ("evaluators", "evaluators", "find_evaluator_def"),
]


def _group_config_files(group):
    return sorted((REPO / "configs" / group).glob("*.yaml"))


@pytest.mark.parametrize("group,package,factory", GROUP_FACTORIES)
def test_group_configs_resolve_through_factory(group, package, factory):
    """configs/<group>/*.yaml 의 name/class_name 이 실제로 클래스로 풀리는지.

    stage-2 용 설정만 본다 (stage-1 설정은 이 리팩토링 대상이 아니다).
    """
    from omegaconf import OmegaConf

    find = getattr(importlib.import_module(package), factory)
    checked = 0
    for path in _group_config_files(group):
        cfg = OmegaConf.load(path)
        name, class_name = cfg.get("name"), cfg.get("class_name")
        if name is None or class_name is None:
            continue
        if group in ("models", "losses", "trainers", "datasets") and "stage2" not in str(path) \
                and name not in ("lm_loss", "trainer_s2", "qa_dataset"):
            continue                      # stage-1 그룹 설정은 건너뛴다
        checked += 1
        cls = find(name, class_name)
        assert isinstance(cls, type), f"{path}: {name}.{class_name} 이 클래스가 아니다"
    assert checked > 0, f"configs/{group}/ 에서 검사할 설정을 못 찾았다"
