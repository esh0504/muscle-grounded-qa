"""Stage-2 QA dataset — 자연화된 QA + mesh 변위.

레코드 하나 = 멀티턴 대화 + 그 대화가 참조하는 mesh(들).

  DATA/qa/{ko,en}/nat_out/nat_*.jsonl
    {"mesh_ref": {"verts_shard": 0, "row_in_shard": 0},   # 단일 mesh
     "scenario": "shape_desc", "conversations": [...], "variant": 0, ...}
    {"mesh_ref": {"indices": [264199, ...]},              # dose_response 는 mesh 6개
     "scenario": "dose_response", ...}

gpt 턴에는 `mask_spans` (자연화 후 재계산된 사실 주석)가 붙어 있고, 이 파일이 그것을
(1) 손실 가중과 (2) 이전 턴 컨텍스트 마스킹에 쓴다. 설계 근거는 doc/baseline.md 참고.
"""

from __future__ import annotations

import glob
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.mesh_dataset import DTYPE, MUSCLE_NAMES, N_SURF, SHARD_SIZE
from datasets.split_trainvaltest import load_split_indices

IGNORE_INDEX = -100

SYSTEM_PROMPT = {
    "ko": (
        "당신은 3D 혀 mesh와 근육 활성 상태를 해석하는 조음(調音) 분석 도우미입니다. "
        "제공된 3D 형상 정보에 근거하여 사실만 답하고, 데이터로 판단할 수 없으면 그렇게 말하세요."
    ),
    "en": (
        "You are an articulation analysis assistant that interprets 3D tongue meshes and "
        "muscle activation states. Answer only from the provided 3D shape information, and "
        "say so when the data does not support an answer."
    ),
}


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        v = cfg.get(key, default)
    else:
        v = getattr(cfg, key, default)
    return default if v is None else v


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def pad_inputs(items: list[dict], key: str = "disp"):
    """레코드마다 mesh 개수가 다르다 (dose_response 는 6개) → M 축 패딩 + valid mask.

    (B, M, ...) 텐서와 (B, M) bool 마스크를 돌려준다. 항목 텐서의 뒤쪽 차원은 그대로
    쓰므로 3D 변위 (M, 370, 3) 든 3뷰 렌더 (M, 3, 3, 128, 128) 든 같은 코드로 묶인다.
    collator 와 평가 쪽 생성 경로가 이 함수 하나를 공유한다.
    """
    m_max = max(int(b["n_mesh"]) for b in items)
    tail = items[0][key].shape[1:]
    out = torch.zeros(len(items), m_max, *tail, dtype=torch.float32)
    valid = torch.zeros(len(items), m_max, dtype=torch.bool)
    for bi, b in enumerate(items):
        m = int(b["n_mesh"])
        out[bi, :m] = b[key]
        valid[bi, :m] = True
    return out, valid


class MeshQaDataset(Dataset):
    """자연화 QA 레코드 → (mesh 변위들, 대화, 근육 라벨).

    레코드 본문은 메모리에 안 들고, (파일, 바이트 오프셋)만 색인해두고 읽는다.
    89만 레코드를 통째로 파싱하면 수 GB 라서.
    """

    # 인코더 입력이 담기는 키.
    # 항목 dict 에도 같이 실어 보내 collator·평가 코드가 클래스를 몰라도 되게 한다.
    input_key = "disp"

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        p = dict(cfg) if isinstance(cfg, dict) else {}
        if cfg is not None and not isinstance(cfg, dict):
            from omegaconf import OmegaConf

            p = dict(OmegaConf.to_container(cfg, resolve=True))
        p.update(kwargs)

        self.lang = str(p.get("lang", "ko"))
        self.mesh_root = Path(p.get("mesh_root", "DATA/mesh"))
        self.split = str(p.get("split", "train"))
        split_dir = p.get("split_dir", None)
        self.exclude = set(p.get("exclude_scenarios") or [])
        variants = p.get("variants", None)
        self.variants = set(variants) if variants else None
        self.faithful_only = bool(p.get("faithful_only", False))
        # split 경계 판정 방식.
        #   all     — 참조 mesh 가 전부 그 split 에 있어야 채택 (기본, 누수 0)
        #   primary — 첫 mesh 만 보고 판정
        #   any     — 하나라도 있으면 채택
        # dose_response(B3)는 레코드당 mesh 6개라 'all' 로는 test(5%)에 거의 안 걸린다
        # (0.05^6). 그런 세트를 평가에 쓰려면 'primary' 로 완화하되, 나머지 mesh 가 train 에
        # 있을 수 있으므로 오염도를 함께 보고해야 한다 (eval.py 가 train_mesh_overlap 로 낸다).
        self.split_policy = str(p.get("split_policy", "all"))
        # turn_wise: assistant 턴 하나 = 학습 샘플 하나.
        #   False(옛 동작) 면 대화 전체가 한 샘플이고 fusion 질문에 **미래 질문까지** 들어간다.
        #   A1 을 예측할 때 Q2·Q3 를 보게 되고(future-question leakage), 평가는 현재 질문만
        #   주므로 train–test mismatch 까지 생긴다.
        #   True 면 컨텍스트는 이전 턴까지만, 질문은 직전 user 턴만, 감독은 대상 턴만.
        self.turn_wise = bool(p.get("turn_wise", True))
        self.max_records = p.get("max_records", None)
        self.system_prompt = SYSTEM_PROMPT.get(self.lang, SYSTEM_PROMPT["ko"])

        qa_glob = str(p.get("qa_glob"))
        self.files = sorted(glob.glob(qa_glob))
        if not self.files and "/nat_out/" in qa_glob:
            # 자연화 코퍼스가 없으면 파이프라인 템플릿 QA 로 폴백 (스키마 동일 — docs/data.md)
            fallback = f"DATA/qa/*_{self.lang}/qa_*.jsonl"
            files = sorted(glob.glob(fallback))
            if files:
                print(f"[MeshQaDataset] {qa_glob} 없음 → 템플릿 QA 로 폴백: {fallback} ({len(files)}개 파일)")
                qa_glob, self.files = fallback, files
        if not self.files:
            raise FileNotFoundError(
                f"qa_glob 에 맞는 파일이 없습니다: {qa_glob} — 파이프라인 템플릿 QA 로 학습하려면 "
                f"datasets.qa_glob='DATA/qa/*_{self.lang}/qa_*.jsonl' 를 넘기세요 (docs/data.md)")

        allowed = None
        if self.split != "all":
            allowed = set(load_split_indices(Path(split_dir or self.mesh_root), self.split))
        self.index = self._build_index(allowed, p.get("index_cache"))
        if not self.index:
            raise RuntimeError("조건에 맞는 QA 레코드가 없습니다 (split/필터 확인)")

        self.rest = self._load_rest(self.mesh_root / "topology.obj")
        self._mmap: dict[int, np.ndarray] = {}
        self._fh: dict[int, Any] = {}
        self._muscles: np.ndarray | None = None

    # ---- 색인 --------------------------------------------------------------
    def _build_index(self, allowed: set[int] | None, cache: str | None):
        if cache and Path(cache).is_file():
            return json.loads(Path(cache).read_text())

        # max_records 를 앞에서부터 자르면 파일 이름 순서(nat_A1_* → nat_B3_* → nat_PH_*) 때문에
        # A1(형상 기술)만 뽑히고 physics_chain·dose_response 를 한 건도 못 본다.
        # 파일마다 같은 몫을 가져와 시나리오가 고르게 섞이게 한다.
        # 몫은 **항목(entry) 기준**이다. turn_wise 면 레코드 하나가 3~4 항목으로 불어나므로
        # 레코드로 세면 파일마다 항목 수가 달라지고, 뒤에서 자를 때 앞쪽 파일(nat_A1_*)만 남는다.
        quota = None
        if self.max_records:
            quota = max(1, -(-int(self.max_records) // max(1, len(self.files))))

        per_file: list[list] = []
        for fi, path in enumerate(self.files):
            entries: list = []
            taken = 0
            with open(path, "rb") as fh:
                off = 0
                for raw in fh:
                    cur, off = off, off + len(raw)
                    if quota is not None and taken >= quota:
                        break
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("scenario") in self.exclude:
                        continue
                    if self.variants is not None and rec.get("variant", 0) not in self.variants:
                        continue
                    if self.faithful_only and not rec.get("nat_all_faithful", True):
                        continue
                    idxs = self._mesh_indices(rec)
                    if not idxs:
                        continue
                    if allowed is not None and not self._in_split(idxs, allowed):
                        continue
                    if self.turn_wise:
                        n_gpt = sum(1 for t in rec.get("conversations", [])
                                    if t.get("from") == "gpt")
                        for ti in range(n_gpt):
                            entries.append([fi, cur, idxs, ti])
                    else:
                        entries.append([fi, cur, idxs])
                    taken = len(entries)
            per_file.append(entries)

        # 파일을 **교대로** 섞어 합친다. 파일 순서대로 이어붙인 뒤 자르면
        # nat_A1_* 만 남아 physics_chain·dose_response 가 통째로 빠진다.
        index = []
        for row in itertools.zip_longest(*per_file):
            index.extend(e for e in row if e is not None)
        if self.max_records:
            index = index[: int(self.max_records)]
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps(index))
        return index

    def _in_split(self, idxs: list[int], allowed: set[int]) -> bool:
        if self.split_policy == "primary":
            return idxs[0] in allowed
        if self.split_policy == "any":
            return any(i in allowed for i in idxs)
        return all(i in allowed for i in idxs)

    @staticmethod
    def _mesh_indices(rec: dict) -> list[int]:
        ref = rec.get("mesh_ref") or {}
        if "indices" in ref:
            return [int(i) for i in ref["indices"]]
        if "verts_shard" in ref and "row_in_shard" in ref:
            return [int(ref["verts_shard"]) * SHARD_SIZE + int(ref["row_in_shard"])]
        return []

    # ---- 정적 자산 ---------------------------------------------------------
    @staticmethod
    def _load_rest(path: Path) -> np.ndarray:
        verts = [
            [float(x) for x in line.split()[1:4]]
            for line in Path(path).read_text().splitlines()
            if line.startswith("v ")
        ]
        arr = np.asarray(verts, dtype=np.float32)
        if arr.shape[0] != N_SURF:
            raise ValueError(f"topology.obj 정점 수 {arr.shape[0]}, 기대값 {N_SURF}")
        return arr

    @property
    def muscles(self) -> np.ndarray:
        """근육 활성 (보조 손실용). 처음 쓸 때만 읽는다."""
        if self._muscles is None:
            import csv

            path = self.mesh_root / "pool_meta.csv"
            rows = list(csv.DictReader(path.open()))
            n = max(int(r["index"]) for r in rows) + 1
            acts = np.zeros((n, len(MUSCLE_NAMES)), dtype=np.float32)
            for r in rows:
                acts[int(r["index"])] = [float(r[m]) for m in MUSCLE_NAMES]
            self._muscles = acts
        return self._muscles

    # ---- 기하 IO -----------------------------------------------------------
    def _shard(self, shard: int) -> np.ndarray:
        arr = self._mmap.get(shard)
        if arr is None:
            path = self.mesh_root / "verts" / f"shard_{shard:05d}.bin"
            arr = np.asarray(np.memmap(path, dtype=DTYPE, mode="r")).reshape(-1, N_SURF, 3)
            self._mmap[shard] = arr
        return arr

    def _displacement(self, index: int) -> np.ndarray:
        sh, lo = divmod(index, SHARD_SIZE)
        return np.asarray(self._shard(sh)[lo], dtype=np.float32) - self.rest

    def _load_inputs(self, idxs: list[int]) -> torch.Tensor:
        """참조 mesh 들의 인코더 입력 (M, ...). 2D row 는 여기만 갈아 끼운다."""
        return torch.from_numpy(np.stack([self._displacement(int(k)) for k in idxs], axis=0))

    # ---- Dataset API -------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def _read_record(self, fi: int, offset: int) -> dict:
        fh = self._fh.get(fi)
        if fh is None:
            fh = open(self.files[fi], "rb")
            self._fh[fi] = fh
        fh.seek(offset)
        return json.loads(fh.readline())

    def __getitem__(self, i: int) -> dict:
        entry = self.index[i]
        fi, offset, idxs = entry[0], entry[1], entry[2]
        target_turn = entry[3] if len(entry) > 3 else None
        rec = self._read_record(fi, offset)

        inputs = self._load_inputs(idxs)        # (M, ...) — 3D 변위 또는 3뷰 렌더

        messages = [{"role": "system", "content": self.system_prompt}]
        spans: list[list[dict]] = []   # assistant 턴별 mask_spans (messages 의 assistant 순서와 정렬)
        questions = []
        ai = 0
        last_question = ""
        for turn in rec["conversations"]:
            if turn["from"] == "human":
                messages.append({"role": "user", "content": turn["value"]})
                questions.append(turn["value"])
                last_question = turn["value"]
            else:
                messages.append({"role": "assistant", "content": turn["value"]})
                spans.append(turn.get("mask_spans", []) or [])
                if target_turn is not None and ai == target_turn:
                    break          # 미래 턴은 컨텍스트에서 통째로 제거
                ai += 1

        if target_turn is None:
            question = "\n".join(questions)     # 옛 동작 (미래 질문 포함 — leakage)
        else:
            question = last_question            # 대상 답변 직전의 질문만

        return {
            self.input_key: inputs,
            "input_key": self.input_key,
            "messages": messages,
            "answer_spans": spans,
            "question": question,
            # 턴 단위일 때는 **마지막 assistant 턴만** 감독한다 (앞 턴은 컨텍스트)
            "supervise_last_only": target_turn is not None,
            "target_turn": target_turn,
            "muscle": torch.from_numpy(self.muscles[idxs[0]].copy()),
            "index": idxs[0],
            "mesh_indices": idxs,
            "n_mesh": len(idxs),
            "scenario": rec.get("scenario", ""),
            "turn_types": rec.get("turn_types", []) or [],
        }


# --------------------------------------------------------------------------- #
# Collate — chat 렌더 + 감독 마스크 + fact 가중 + 컨텍스트 마스킹
# --------------------------------------------------------------------------- #
class QaCollator:
    """대화를 렌더하고 (input_ids, labels, loss_weights) 를 만든다.

    - labels: assistant 턴 토큰만. 나머지는 IGNORE_INDEX.
    - loss_weights: 선택된 fact span 토큰은 λ, 그 외 감독 토큰은 1.0, 나머지 0.0.
    - 컨텍스트 마스킹: 마지막 답변 턴을 제외한 이전 답변의 span 토큰을
      확률 p 로 mask 토큰으로 바꾼다. **label 은 원본 그대로 둔다** — causal 이라
      그 자리의 예측은 영향받지 않고, 뒤 턴이 앞 턴 값을 베끼는 경로만 끊긴다.
    """

    def __init__(self, tokenizer, cfg=None, seed: int = 0):
        m = _cfg_get(cfg, "masking", {}) or {}
        m = dict(m) if not isinstance(m, dict) else m

        self.tok = tokenizer
        self.max_len = int(_cfg_get(cfg, "max_len", 2048))
        self.q_max_len = int(_cfg_get(cfg, "q_max_len", 256))

        self.span_source = str(m.get("span_source", "mask_spans"))
        self.mask_types = set(m.get("mask_types") or ["number", "muscle", "region", "movement"])
        self.r = float(m.get("span_select_ratio", 0.5))
        self.tau = m.get("target_fact_share", None)
        self.tau = float(self.tau) if self.tau is not None else None
        self.fact_weight = float(m.get("fact_weight", 3.0))
        self.max_fact_weight = float(m.get("max_fact_weight", 20.0))
        self.p_ctx = float(m.get("context_mask_prob", 0.0))
        # 학습 렌더를 평가와 같게 고정 (eval.py 의 generation.enable_thinking 와 일치시킬 것)
        self.enable_thinking = bool(_cfg_get(cfg, "enable_thinking", False))

        tokname = str(m.get("context_mask_token", "<|fim_pad|>"))
        self.mask_id = tokenizer.convert_tokens_to_ids(tokname)
        if self.mask_id is None or self.mask_id == tokenizer.unk_token_id:
            raise ValueError(f"context_mask_token 을 어휘에서 못 찾았습니다: {tokname}")

        # mesh prefix 를 어디에 넣을지.
        #   prepend — 대화 전체 앞(= <|im_start|>system 보다도 앞)에 붙인다. 초기 구현.
        #   inline  — 현재 user 턴 **본문 맨 앞**에 자리표시자 토큰을 깔고 그 자리에 끼운다.
        #             LLaVA 계열 표준. prepend 는 prefix 가 대화 시작 이전에 놓여
        #             instruction-tuned LLM 이 무시하기 쉽다 (진단 SUMMARY.md 참조:
        #             기하 정보가 prefix 까지 R²=0.932 로 도달하는데 모델이 안 읽었다).
        self.mesh_inject = str(_cfg_get(cfg, "mesh_inject", "prepend"))
        self.n_mesh_tokens = int(_cfg_get(cfg, "n_mesh_tokens", 32))
        mtok = str(_cfg_get(cfg, "mesh_token", "<|vision_pad|>"))
        self.mesh_token = mtok
        self.mesh_token_id = tokenizer.convert_tokens_to_ids(mtok)
        if self.mesh_inject == "inline" and (self.mesh_token_id is None
                                             or self.mesh_token_id == tokenizer.unk_token_id):
            raise ValueError(f"mesh_token 을 어휘에서 못 찾았습니다: {mtok}")
        if self.mesh_inject not in ("prepend", "inline"):
            raise ValueError(f"mesh_inject 는 prepend|inline 이어야 합니다: {self.mesh_inject}")

        self.im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assistant_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
        self.nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.header = [self.im_start, self.assistant_id, self.nl_id]
        self.rng = random.Random(seed)

    # ---- 한 대화 인코딩 ----------------------------------------------------
    def _assistant_spans_in_text(self, text: str, messages, answer_spans):
        """assistant 턴별 fact span 목록과, 각 턴 **본문의 char 범위**를 함께 돌려준다.

        본문 범위가 필요한 이유: Qwen3 chat template 는 assistant 메시지 앞에
        `<think>\\n\\n</think>\\n\\n` 를 끼워 넣는다. 그런데 평가용 generation prompt 는 그 블록으로
        **끝난다**(모델은 본문부터 생성). 학습에서 그 블록까지 감독하면 상수 토큰에 손실이 새고
        fact 비율(τ) 계산도 왜곡된다.
        """
        out, content_bounds = [], []
        cursor = 0
        ai = 0
        for msg in messages:
            content = msg["content"]
            at = text.find(content, cursor)
            if at < 0:                       # 템플릿이 내용을 변형한 경우 — 건너뛴다
                continue
            cursor = at + len(content)
            if msg["role"] != "assistant":
                continue
            spans = answer_spans[ai] if ai < len(answer_spans) else []
            ai += 1
            keep = []
            content_bounds.append((at, at + len(content)))
            for s in spans:
                if s.get("type") not in self.mask_types:
                    continue
                cs, ce = at + int(s["start"]), at + int(s["end"])
                if ce <= cs or ce > len(text):
                    continue
                if text[cs:ce] != s.get("value"):   # 오프셋이 안 맞으면 버린다
                    continue
                keep.append((cs, ce))
            out.append(keep)
        return out, content_bounds

    def _mesh_placeholder(self, messages):
        """inline 주입: 마지막 user 턴 본문 맨 앞에 자리표시자 토큰을 n_mesh_tokens 개 깐다.

        messages 사본을 돌려준다 (원본 item 을 건드리면 epoch 마다 누적된다).
        """
        msgs = [dict(m) for m in messages]
        last_user = next((i for i in range(len(msgs) - 1, -1, -1)
                          if msgs[i]["role"] == "user"), None)
        if last_user is None:
            return msgs
        pad = self.mesh_token * self.n_mesh_tokens
        msgs[last_user]["content"] = pad + msgs[last_user]["content"]
        return msgs

    def _find_mesh_slot(self, ids):
        """자리표시자 토큰이 연속으로 놓인 시작 위치. 못 찾거나 잘렸으면 -1."""
        want = self.n_mesh_tokens
        run = start = 0
        for k, t in enumerate(ids):
            if t == self.mesh_token_id:
                if run == 0:
                    start = k
                run += 1
                if run == want:
                    return start
            else:
                run = 0
        return -1

    def _encode(self, item: dict):
        messages = item["messages"]
        if self.mesh_inject == "inline":
            messages = self._mesh_placeholder(messages)
        # enable_thinking 을 평가와 똑같이 고정한다. 안 맞추면 학습 때 본 렌더와
        # 생성 때 렌더가 달라진다 (Qwen3 는 빈 <think></think> 를 넣기도 한다).
        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=self.enable_thinking)
        except TypeError:
            text = self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=False)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        n = len(ids)

        labels = [IGNORE_INDEX] * n
        weights = [0.0] * n

        # 1) assistant 구간 찾기 (헤더 ~ 다음 <|im_end|> 포함)
        regions = []
        i = 0
        while i < n:
            if ids[i:i + 3] == self.header:
                start = i + 3
                j = start
                while j < n and ids[j] != self.im_end:
                    j += 1
                end = min(j + 1, n)
                regions.append((start, end))
                i = end
            else:
                i += 1

        per_turn, bounds = self._assistant_spans_in_text(text, messages, item["answer_spans"])
        n_turns = len(per_turn)

        # 턴 단위 학습이면 **마지막 턴만** 감독한다. 앞 턴 답변은 컨텍스트로만 둔다.
        keep_idx = [len(regions) - 1] if (item.get("supervise_last_only") and regions) \
            else list(range(len(regions)))
        for ri in keep_idx:
            start, end = regions[ri]
            # 본문 시작 전(<think> 블록)은 감독에서 제외. 종료 토큰(<|im_end|>)은 남긴다.
            c0 = bounds[ri][0] if ri < len(bounds) else -1
            for k in range(start, end):
                cs, ce = offsets[k]
                if c0 >= 0 and ce > cs and ce <= c0:
                    continue
                labels[k] = ids[k]
                weights[k] = 1.0

        # 자리표시자 위치는 **자르기 이후** 기준으로 찾는다 (max_len 에 잘리면 -1).
        mesh_at = self._find_mesh_slot(ids[:self.max_len]) if self.mesh_inject == "inline" else -1

        if self.span_source != "mask_spans":
            return ids[:self.max_len], labels[:self.max_len], weights[:self.max_len], mesh_at

        # 2) 손실 가중 대상 선택 (턴마다 비율 r)
        sel_chars = np.zeros(len(text), dtype=bool)
        for spans in per_turn:
            if not spans or self.r <= 0:
                continue
            # round() 로 뽑으면 span 이 1개일 때 r=0.5 에서 round(0.5)=0 (은행가 반올림)이라
            # **유일한 fact 가 한 번도 선택되지 않는다**. ceil 로 최소 1개는 남긴다.
            k = min(len(spans), max(1, math.ceil(len(spans) * self.r)))
            for cs, ce in self.rng.sample(spans, k):
                sel_chars[cs:ce] = True

        # 3) 컨텍스트 마스킹 대상 선택 (마지막 답변 턴 제외)
        ctx_chars = np.zeros(len(text), dtype=bool)
        if self.p_ctx > 0 and n_turns > 1:
            for spans in per_turn[:-1]:
                for cs, ce in spans:
                    if self.rng.random() < self.p_ctx:
                        ctx_chars[cs:ce] = True

        fact_tok = []
        for k in range(min(n, self.max_len)):
            if labels[k] == IGNORE_INDEX:
                continue
            cs, ce = offsets[k]
            if ce <= cs:
                continue
            if sel_chars[cs:ce].any():
                fact_tok.append(k)
            if ctx_chars[cs:ce].any():
                ids[k] = self.mask_id      # 입력만 가린다. labels[k] 는 원본 유지.

        # 4) λ 결정 — τ 가 주어지면 이 예시의 span 비율로 역산
        n_sup = sum(1 for k in range(min(n, self.max_len)) if labels[k] != IGNORE_INDEX)
        lam = self.fact_weight
        if self.tau is not None and fact_tok and n_sup > len(fact_tok):
            s = len(fact_tok) / n_sup
            lam = (self.tau / (1.0 - self.tau)) * ((1.0 - s) / s)
            lam = float(min(max(lam, 1.0), self.max_fact_weight))
        for k in fact_tok:
            weights[k] = lam

        return ids[:self.max_len], labels[:self.max_len], weights[:self.max_len], mesh_at

    # ---- batch -------------------------------------------------------------
    def __call__(self, batch: list[dict]) -> dict:
        # 인코더 입력 키는 dataset 이 항목에 실어 보낸다 ("disp" 3D · "imgs" 2D 렌더)
        key = batch[0].get("input_key", "disp")
        inputs, mesh_valid = pad_inputs(batch, key)

        enc = [self._encode(b) for b in batch]
        max_l = max(len(x[0]) for x in enc)
        pad_id = self.tok.pad_token_id

        input_ids, labels, attn, weights, mesh_at = [], [], [], [], []
        for ids, lab, w, mat in enc:
            k = max_l - len(ids)
            input_ids.append(ids + [pad_id] * k)
            labels.append(lab + [IGNORE_INDEX] * k)
            attn.append([1] * len(ids) + [0] * k)
            weights.append(w + [0.0] * k)
            mesh_at.append(mat)   # 오른쪽 패딩이라 위치는 그대로 유효하다

        q = self.tok(
            [b["question"] for b in batch],
            padding=True, truncation=True, max_length=self.q_max_len, return_tensors="pt",
        )
        return {
            key: inputs,
            "mesh_valid": mesh_valid,
            "muscle": torch.stack([b["muscle"] for b in batch], dim=0),
            "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "loss_weights": torch.tensor(weights, dtype=torch.float32),
            "q_input_ids": q["input_ids"],
            "q_attention_mask": q["attention_mask"],
            # inline 주입일 때 prefix 를 끼워 넣을 시작 위치. -1 = 없음(또는 잘림) → prepend 로 fallback.
            "mesh_at": torch.tensor(mesh_at, dtype=torch.long),
        }
