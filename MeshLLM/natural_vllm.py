# -*- coding: utf-8 -*-
# ⚠️ 이 파일은 dummy/ 로 옮기면 안 된다. 데이터 자연화 스크립트이면서 동시에
#    metrics/spans.py 가 쓰는 **채점 어휘의 단일 출처**다 (NUMRE·VOCAB·_concepts).
#    채점기가 mask_spans 를 만든 코드와 다른 어휘를 쓰면 점수가 어휘 차이를 재게 된다.
"""Track B 자연화 — vLLM 오프라인 배치 (2xH200 권장, TP=2).
레코드(멀티턴) 단위로 각 human/gpt 턴을 자연화 → 같은 스키마로 재구성(mask_spans 재계산).
인라인 verify(숫자 멀티셋 + 근육 정식명 생존) → 실패 턴은 재시도, 최종 실패 시 원문 폴백(faithful 유지).
shard resume. 같은 FIXED_PROMPT 사용. ingest_web.py의 검증 로직과 동일 규칙.

설치: pip install vllm
실행 예:
  python natural_vllm.py --src 'qa_feat/qa_A1_*.jsonl' --tag A1 --prompt path/to/FIXED_PROMPT.txt \
      --model Qwen/Qwen2.5-72B-Instruct --tp 2 --variants 3 --out nat_out
  python natural_vllm.py --src 'qa_full_300k/qa_*.jsonl' --tag PH --model Qwen/Qwen2.5-72B-Instruct --tp 2 --variants 3 --out nat_out
드라이런(모델 없이 파싱·verify·mask 로직 점검):  python natural_vllm.py --selftest
"""
import json, glob, re, os, argparse, sys, time
from collections import Counter

# ---------------- 불변식 규칙 (ingest_web.py와 동일) ----------------
# 근육 정식명칭: ko는 "한국어명(라틴어)", en은 "latin name (ABBR)" — 각 언어 데이터의 실제 표기.
FULL_KO={"GGP":"이설근 후부(genioglossus posterior)","GGM":"이설근 중부(genioglossus medius)","GGA":"이설근 전부(genioglossus anterior)",
"STY":"경돌설근(styloglossus)","GH":"이설골근(geniohyoid)","MH":"악설골근(mylohyoid)","HG":"설골설근(hyoglossus)",
"VERT":"수직근(verticalis)","TRANS":"횡근(transversus)","IL":"하종설근(inferior longitudinal)","SL":"상종설근(superior longitudinal)"}
FULL_EN={"GGP":"genioglossus posterior (GGP)","GGM":"genioglossus medius (GGM)","GGA":"genioglossus anterior (GGA)",
"STY":"styloglossus (STY)","GH":"geniohyoid (GH)","MH":"mylohyoid (MH)","HG":"hyoglossus (HG)",
"VERT":"verticalis (VERT)","TRANS":"transversus (TRANS)","IL":"inferior longitudinal (IL)","SL":"superior longitudinal (SL)"}
NUMRE=re.compile(r"-?\d+\.?\d*")
# mask용 방향/움직임 어휘 (physics DIRW + feature MOVE 합집합). 굴절형이 있는 en은 긴 형태를 앞에 둬야
# 겹침 제거에서 긴 쪽이 살아남는다 ("increases" 앞에 "increase"를 두면 안 됨).
MOVE_KO=["단조적으로 증가","단조 증가","전진","후퇴","상승","하강","보존","보상","상쇄","증가","감소","포화","역전","납작","아치","단조","narrow"]
# en 목록은 EN 데이터의 원본 mask_spans에서 그대로 추출한 18종(어간형 포함: "retracts" 안의 "retract").
# 추측형을 넣으면 원본이 표시하지 않은 자리까지 잡아 원본과 어긋난다.
MOVE_EN=["monotonically increases","monotonically","increases","increase","consistent","preserved",
"compensate","offsets","retract","raise","rises","domed","lower","flattened","advance",
"saturation","reversal","negligible"]
REGION_KO=["front","mid","back","전방","중간","후방"]
REGION_EN=["front","mid","back"]

# 방향/부위 어휘의 표면형 → 원자 개념. 표면형만 비교하면 "단조 증가"→"단조적으로 증가" 같은
# 정당한 자연화까지 탈락하므로 개념 단위로 생존을 본다.
MOVE_CONCEPT_KO={"단조적으로 증가":{"mono","inc"},"단조 증가":{"mono","inc"},"단조":{"mono"},
"전진":{"advance"},"후퇴":{"retract"},"상승":{"elevate"},"하강":{"descend"},
"보존":{"preserve"},"보상":{"compensate"},"상쇄":{"cancel"},"증가":{"inc"},"감소":{"dec"},
"포화":{"saturate"},"역전":{"reverse"},"납작":{"flat"},"아치":{"arch"},"narrow":{"narrow"}}
MOVE_CONCEPT_EN={"monotonically increases":{"mono","inc"},"monotonically":{"mono"},
"increases":{"inc"},"increase":{"inc"},"consistent":{"consistent"},"preserved":{"preserve"},
"compensate":{"compensate"},"offsets":{"cancel"},"retract":{"retract"},"raise":{"elevate"},
"rises":{"elevate"},"domed":{"arch"},"lower":{"descend"},"flattened":{"flat"},
"advance":{"advance"},"saturation":{"saturate"},"reversal":{"reverse"},"negligible":{"negligible"}}
REGION_CONCEPT_KO={"front":{"front"},"전방":{"front"},"mid":{"mid"},"중간":{"mid"},
"back":{"back"},"후방":{"back"}}
REGION_CONCEPT_EN={"front":{"front"},"mid":{"mid"},"back":{"back"}}

# 숫자의 role 추정 문맥 키. 변수명은 언어 공통, 화살표·도밍 표기만 갈린다.
ROLE_COMMON=[("cl_t","cl_t"),("cd_min","cd_min"),("peak_xn","peak_xn"),("peak_z","peak_z"),
             ("tilt","tilt"),("arc_len","arc_len"),("curv_peak","curv_peak"),("vol_ratio","vol_ratio"),
             ("ρ","rho"),("dx","dx_mm"),("dz","dz_mm"),("mm","disp_mm")]
VOCAB={
 "ko":{"names":list(FULL_KO.values()),"move":MOVE_KO,"region":REGION_KO,
       "move_c":MOVE_CONCEPT_KO,"region_c":REGION_CONCEPT_KO,
       "role":[("도밍","doming")]+ROLE_COMMON+[("→","disp_mm")]},
 "en":{"names":list(FULL_EN.values()),"move":MOVE_EN,"region":REGION_EN,
       "move_c":MOVE_CONCEPT_EN,"region_c":REGION_CONCEPT_EN,
       "role":[("doming","doming")]+ROLE_COMMON+[("->","disp_mm")]},
}

def _nums(t): return sorted(x for x in NUMRE.findall(t) if x not in ("","-","."))
def _names(t, names): return sorted(n for n in names if n in t)
def _concepts(t, table):
    c=set()
    for surf,con in table.items():
        if surf in t: c|=con
    return c
def verify(orig, new, lang="ko"):
    v=VOCAB[lang]
    miss_num=list((Counter(_nums(orig))-Counter(_nums(new))).elements())
    miss_name=[x for x in _names(orig,v["names"]) if x not in _names(new,v["names"])]
    miss_move=sorted(_concepts(orig,v["move_c"])-_concepts(new,v["move_c"]))
    miss_region=sorted(_concepts(orig,v["region_c"])-_concepts(new,v["region_c"]))
    ok=not(miss_num or miss_name or miss_move or miss_region)
    return ok, {"missing_numbers":miss_num,"missing_names":miss_name,
                "missing_moves":miss_move,"missing_regions":miss_region}

# ---------------- mask_spans 재계산 (자연화 후 오프셋 갱신) ----------------
def mask_spans(text, lang="ko"):
    v=VOCAB[lang]; sp=[]
    for m in v["names"]:
        i=text.find(m)
        while i>=0: sp.append({"type":"muscle","value":m,"start":i,"end":i+len(m)}); i=text.find(m,i+len(m))
    for w in v["move"]:
        i=text.find(w)
        while i>=0: sp.append({"type":"movement","value":w,"start":i,"end":i+len(w)}); i=text.find(w,i+len(w))
    for w in v["region"]:
        i=text.find(w)
        while i>=0: sp.append({"type":"region","value":w,"start":i,"end":i+len(w)}); i=text.find(w,i+len(w))
    ROLE=v["role"]
    for mt in NUMRE.finditer(text):
        st=mt.start()
        if mt.group() in ("","-","."): continue
        if text[max(0,st-1):st]=="#": continue
        ctx=text[max(0,st-14):st]
        role=next((r for k,r in ROLE if k in ctx),"value")
        sp.append({"type":"number","value":mt.group(),"start":st,"end":mt.end(),"role":role})
    pr={"muscle":0,"region":1,"movement":2,"number":3}
    sp.sort(key=lambda s:(s["start"],pr[s["type"]]))
    out=[];occ=[]
    for s in sp:
        if any(not(s["end"]<=a or s["start"]>=b) for a,b in occ): continue
        out.append(s);occ.append((s["start"],s["end"]))
    return sorted(out,key=lambda s:s["start"])

# ---------------- 프롬프트 / 출력 파싱 ----------------
DEFAULT_PROMPT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"FIXED_PROMPT.txt")
FALLBACK_SYSTEM="숫자·근육 정식명칭 100% 고정, 말투만 자연화. 없는 사실 추가 금지."
FORMAT_SPEC_KO=("\n\n[추가 제약]\n방향어(단조/증가/감소/전진/후퇴/상승/하강/포화/역전/보존/보상/상쇄/납작/아치)와 "
             "부위어(전방/중간/후방, front/mid/back)는 다른 말로 풀어쓰지 말고 그 단어를 그대로 남겨라. "
             "예: '단조 증가'를 '점점 커진다'로 바꾸지 말 것.\n"
             "\n[이 작업의 입출력 형식]\n입력은 항목 하나의 'Q: ...' / 'A: ...' 이다. 자연화한 뒤 "
             "정확히 JSON 한 줄로만 출력하라:\n"
             '{"q": "(자연화 질문)", "a": "(자연화 답변)"}\n'
             "id는 필요 없다. 코드블록·머리말·설명 없이 이 한 줄만.")
FORMAT_SPEC_EN=("\n\nAdditional constraint: keep the direction words (monotonically increases/increase/consistent/"
             "preserved/compensate/offsets/retract/raise/rises/domed/lower/flattened/advance/saturation/"
             "reversal/negligible) "
             "and the region words (front/mid/back) verbatim; do not paraphrase them away "
             "(e.g. do not turn \"monotonically increases\" into \"keeps getting bigger\").\n"
             "\nOutput format: the input is a single item given as 'Q: ...' / 'A: ...'. Rewrite both, then output "
             "exactly one JSON line and nothing else (no id, no prose, no code fences):\n"
             '{"q": "(rewritten question)", "a": "(rewritten answer)"}')
FORMAT_SPEC={"ko":FORMAT_SPEC_KO,"en":FORMAT_SPEC_EN}
def load_system(path=None, lang="ko"):
    if path and not os.path.exists(path):
        sys.exit(f"[ERR] --prompt 파일이 없습니다: {path}")
    p=path or DEFAULT_PROMPT
    if os.path.exists(p):
        print(f"[prompt] {p} (lang={lang})")
        return open(p,encoding="utf-8").read()+FORMAT_SPEC[lang]
    # 규칙 없는 폴백으로 대량 생성하면 verify 실패·원문 폴백이 급증하므로 반드시 알린다
    print(f"[prompt][WARN] FIXED_PROMPT.txt 없음({p}) → 축약 폴백 프롬프트 사용. --prompt 로 경로를 지정하세요.")
    return FALLBACK_SYSTEM+FORMAT_SPEC[lang]
QA_RE=re.compile(r"<Q>\s*(.*?)\s*</Q>\s*<A>\s*(.*?)\s*</A>", re.S)
# variant끼리 프롬프트가 같으면 seed만으로는 말투가 거의 안 갈린다. variant별로 다른 말투를 지시해
# 서로 다른 문장이 나오게 한다. 사실·수치는 그대로여야 하므로 어순·어투만 건드리는 지시로 제한.
STYLES={
 "ko":["짧게 끊어 말하는 구어체로 바꿔라.",
       "친절하게 설명하는 말투로, 절의 순서를 바꿔서 써라.",
       "간결한 전문가 톤으로 써라.",
       "동료에게 브리핑하듯 담백한 평서문으로 써라.",
       "수치를 먼저 제시하고 해석을 뒤에 붙이는 순서로 재배치해라.",
       "질문에 곧바로 답하는 순서로 문장을 재구성해라."],
 "en":["Rewrite it in a conversational tone with shorter sentences.",
       "Rewrite it in a friendly explanatory tone, reordering the clauses.",
       "Rewrite it in a concise expert tone.",
       "Rewrite it as a plain, matter-of-fact briefing to a colleague.",
       "Restructure it so the numbers come first and the interpretation follows.",
       "Restructure it so it answers the question directly from the first clause."],
}
NOECHO={"ko":"반드시 입력과 다른 문장이어야 한다. 입력을 그대로 되풀이하지 마라.",
        "en":"The result must not be the input verbatim; do not echo the input unchanged."}
# 말투 지시가 문장 압축을 유도하면 척도 설명절("0=앞, 1=뒤")이 통째로 날아가 그 안의 숫자까지 사라진다.
# 실측값은 살아 있는데 범례 숫자만 빠져 검증에서 탈락하므로, 절 삭제 금지를 말투 지시 옆에 붙여둔다.
KEEPALL={"ko":"단, 어떤 절도 삭제하지 마라. 숫자가 들어간 절과 척도 설명(예: '0=앞, 1=뒤')은 반드시 남기고, "
              "방향어·부위어도 그 단어 그대로 남겨라.",
         "en":"But do not delete any clause: every clause containing a number must survive, including scale "
              "legends such as \"where 0 is front and 1 is back\". Keep the direction and region words verbatim."}
def build_user(q,a,vi=0,lang="ko"):
    st=STYLES[lang][vi%len(STYLES[lang])]
    return f"[style] {st} {NOECHO[lang]} {KEEPALL[lang]}\n\nQ: {q}\nA: {a}"
def parse_out(text):
    m=QA_RE.search(text)
    if m: return m.group(1).strip(), m.group(2).strip()
    # FIXED_PROMPT는 웹 워크플로용이라 "JSONL로만 출력"을 지시한다. 모델이 그 쪽을 따르는 경우가 많아
    # {"q":...,"a":...} 한 줄도 받아준다 (id는 무시).
    for line in text.strip().splitlines():
        line=line.strip().strip("`").strip()
        if not (line.startswith("{") and line.endswith("}")): continue
        try: o=json.loads(line)
        except Exception: continue
        if isinstance(o,dict) and "q" in o and "a" in o:
            return str(o["q"]).strip(), str(o["a"]).strip()
    return None

# ---------------- 레코드 처리 ----------------
def gpt_turn_indices(conv):
    return [i for i in range(1,len(conv)) if conv[i]["from"]=="gpt" and conv[i-1]["from"]=="human"]

def selftest():
    # 모델 없이 파싱·verify·mask 점검 (가짜 자연화)
    orig_a="이설근 후부(genioglossus posterior) 활성이 커질수록 rest 대비 평균 부위 변위가 단조 증가합니다: 0.15→1.35mm, 0.90→2.02mm (Spearman ρ=1.00)."
    good="이설근 후부(genioglossus posterior)를 세게 켤수록 rest 대비 평균 부위 변위가 단조 증가해요: 0.15→1.35mm, 0.90→2.02mm였고 (Spearman ρ=1.00)."
    para="이설근 후부(genioglossus posterior)를 세게 켤수록 변위가 단조적으로 증가해요: 0.15→1.35mm, 0.90→2.02mm (Spearman ρ=1.00)."  # 표면형만 바뀜 → 통과해야 함
    lost="이설근 후부(genioglossus posterior)를 세게 켤수록 변위가 점점 커져요: 0.15→1.35mm, 0.90→2.02mm (Spearman ρ=1.00)."  # 방향어 소실
    bad ="근육을 켜면 변위가 커집니다: 대략 1.4mm에서 2mm로."  # 숫자·근육명 손실
    out_good=f"<Q>이 근육 세기를 키우면?</Q><A>{good}</A>"
    out_json=json.dumps({"id":1,"q":"이 근육 세기를 키우면?","a":good},ensure_ascii=False)
    print("parse <Q>/<A>:", parse_out(out_good) is not None,
          "| parse JSONL:", parse_out(out_json) is not None,
          "| parse 코드블록JSONL:", parse_out("```json\n"+out_json+"\n```") is not None,
          "| parse 쓰레기:", parse_out("sorry, I cannot") is None)
    print("verify good:", verify(orig_a,good)[0], "| 표면형만 변경:", verify(orig_a,para)[0])
    print("verify 방향어소실:", verify(orig_a,lost))
    print("verify bad:", verify(orig_a,bad))
    for lg in VOCAB:
        v=VOCAB[lg]
        print(f"개념표 커버리지({lg}):", all(w in v["move_c"] for w in v["move"]),
              all(w in v["region_c"] for w in v["region"]))
    sp=mask_spans(good)
    print("mask_spans:", len(sp), "spans; offsets ok:", all(good[s['start']:s['end']]==s['value'] for s in sp))
    # en 경로도 같은 규칙으로 도는지 점검
    en_o="As genioglossus posterior (GGP) activation rises, mean regional displacement from rest monotonically increases: 0.15->1.35mm, 0.90->2.02mm (Spearman rho=1.00), peaking at the back."
    en_g="Once genioglossus posterior (GGP) activation rises, the mean regional displacement from rest monotonically increases: 0.15->1.35mm, 0.90->2.02mm (Spearman rho=1.00), and it peaks at the back."
    en_l="Crank up genioglossus posterior (GGP) and displacement just keeps growing: 0.15->1.35mm, 0.90->2.02mm (Spearman rho=1.00), peaking at the rear."
    print("en verify good:", verify(en_o,en_g,"en")[0], "| en 소실:", verify(en_o,en_l,"en"))
    sp=mask_spans(en_g,"en")
    print("en mask_spans:", len(sp), "spans; offsets ok:", all(en_g[s['start']:s['end']]==s['value'] for s in sp))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src"); ap.add_argument("--tag",default="X")
    ap.add_argument("--model",default="Qwen/Qwen2.5-72B-Instruct")
    ap.add_argument("--tp",type=int,default=2); ap.add_argument("--variants",type=int,default=3)
    ap.add_argument("--out",default="nat_out"); ap.add_argument("--shard-size",type=int,default=2000)
    ap.add_argument("--max-retries",type=int,default=2); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--max-tokens",type=int,default=512); ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--prompt",default=None,help="FIXED_PROMPT.txt 경로 (기본: 스크립트와 같은 폴더)")
    ap.add_argument("--lang",default="ko",choices=sorted(VOCAB),help="데이터 언어. 검증·mask 어휘가 갈린다")
    ap.add_argument("--debug-dump",type=int,default=3,help="시도별로 실패 원문 N개를 out/debug_*.txt 에 기록")
    ap.add_argument("--stride",type=int,default=1,help="원본 파일을 N개마다 하나만 처리(전량의 1/N 규모로 축소)")
    a=ap.parse_args()
    if a.selftest: selftest(); return
    assert a.src, "--src 필요"
    system=load_system(a.prompt,a.lang)
    srcs=sorted(glob.glob(a.src))
    assert srcs, f"--src 패턴에 맞는 파일이 없습니다: {a.src}"
    if a.stride>1:
        srcs=srcs[::a.stride]
        print(f"[{a.tag}] stride={a.stride} -> 원본 파일 {len(srcs)}개만 처리")
    os.makedirs(a.out,exist_ok=True)
    from vllm import LLM, SamplingParams
    llm=LLM(model=a.model, tensor_parallel_size=a.tp, enable_prefix_caching=True, dtype="bfloat16",
            gpu_memory_utilization=0.92, max_model_len=4096)
    tok=llm.get_tokenizer()
    lang=a.lang
    def chat(q,ans,vi):
        msgs=[{"role":"system","content":system},{"role":"user","content":build_user(q,ans,vi,lang)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # 레코드 로드
    recs=[]
    for src in srcs:
        for ln,line in enumerate(open(src,encoding="utf-8")):
            recs.append((os.path.basename(src),ln,json.loads(line)))
            if a.limit and len(recs)>=a.limit: break
        if a.limit and len(recs)>=a.limit: break
    print(f"[{a.tag}] {len(recs)} records, {a.variants} variants each")

    # 며칠짜리 런에서는 "몇 번째 shard인지"만으론 남은 시간을 못 본다. 완료한 shard의 실측 속도로 ETA를 찍는다.
    starts=list(range(0,len(recs),a.shard_size))
    todo=[s for s in starts if not os.path.exists(f"{a.out}/nat_{a.tag}_{s//a.shard_size:05d}.jsonl")]
    print(f"[{a.tag}] shard {len(starts)}개 중 {len(todo)}개 남음 (완료분은 resume으로 건너뜀)", flush=True)
    t_run=time.time(); done_sh=0
    for sh0 in starts:
        outp=f"{a.out}/nat_{a.tag}_{sh0//a.shard_size:05d}.jsonl"
        if os.path.exists(outp): continue
        t_sh=time.time()
        chunk=recs[sh0:sh0+a.shard_size]
        # (rec_i, turn_i, variant) 단위 작업 생성
        jobs=[]  # (key, q, a)
        for ri,(sfn,ln,rec) in enumerate(chunk):
            for ti in gpt_turn_indices(rec["conversations"]):
                q=rec["conversations"][ti-1]["value"]; a_=rec["conversations"][ti]["value"]
                for vi in range(a.variants):
                    jobs.append(((ri,ti,vi),q,a_))
        # 자연화 결과 저장소: [ri][vi][ti] = (q2,a2,ok)
        res={}
        pending=list(range(len(jobs)))
        for attempt in range(a.max_retries+1):
            if not pending: break
            temp=0.7+0.15*attempt
            # variant끼리 프롬프트가 완전히 같으므로 seed를 배치에 하나만 주면 3 variant가 동일해진다.
            # job별 seed를 줘서 증식 효과를 살리고, 값은 (attempt, job)에 의해 결정적이라 재현도 된다.
            prompts=[chat(jobs[j][1],jobs[j][2],jobs[j][0][2]) for j in pending]
            sps=[SamplingParams(temperature=temp, top_p=0.9, max_tokens=a.max_tokens,
                                seed=1000+attempt*10_000_019+j) for j in pending]
            outs=llm.generate(prompts, sps)
            still=[]; dbg=[]
            for j,o in zip(pending,outs):
                raw=o.outputs[0].text
                pr=parse_out(raw)
                (ri,ti,vi)=jobs[j][0]; orig_a=jobs[j][2]; orig_q=jobs[j][1]
                if pr:
                    q2,a2=pr; ok,info=verify(orig_a,a2,a.lang)
                    # 질문도 숫자/근육명 보존 확인(있으면)
                    okq,infoq=verify(orig_q,q2,a.lang)
                    if ok and okq:
                        res[(ri,ti,vi)]=(q2,a2,True); continue
                    if len(dbg)<a.debug_dump: dbg.append(("verify",raw,{"a":info,"q":infoq}))
                elif len(dbg)<a.debug_dump: dbg.append(("parse",raw,None))
                still.append(j)
            pending=still
            # 통과율이 0에 가까우면 프롬프트/형식 문제이므로 원문 출력을 남겨 원인을 볼 수 있게 한다
            if dbg:
                dp=f"{a.out}/debug_{a.tag}_sh{sh0//a.shard_size:05d}_att{attempt}.txt"
                with open(dp,"w",encoding="utf-8") as fd:
                    for why,raw,info in dbg:
                        fd.write(f"===== fail={why} info={info}\n{raw}\n\n")
                print(f"  [debug] {len(dbg)} failing outputs -> {dp}")
            print(f"  shard {sh0//a.shard_size}: attempt {attempt} temp{temp:.2f} -> {len(res)}/{len(jobs)} ok, {len(pending)} retry")
        # 실패분 원문 폴백
        for j in pending:
            (ri,ti,vi)=jobs[j][0]; res[(ri,ti,vi)]=(jobs[j][1],jobs[j][2],False)
        # 레코드 재구성
        n_out=0
        with open(outp,"w") as fo:
            for ri,(sfn,ln,rec) in enumerate(chunk):
                gt=gpt_turn_indices(rec["conversations"])
                for vi in range(a.variants):
                    conv=[dict(t) for t in rec["conversations"]]
                    all_ok=True
                    for ti in gt:
                        q2,a2,ok=res[(ri,ti,vi)]
                        conv[ti-1]={"from":"human","value":q2}
                        conv[ti]={"from":"gpt","value":a2,"mask_spans":mask_spans(a2,a.lang)}
                        all_ok=all_ok and ok
                    nr=dict(rec); nr["conversations"]=conv
                    nr["variant"]=vi; nr["naturalized"]=True; nr["nat_all_faithful"]=all_ok
                    nr["src_ref"]={"file":sfn,"line":ln}
                    fo.write(json.dumps(nr,ensure_ascii=False)+"\n"); n_out+=1
        done_sh+=1
        el=time.time()-t_sh; rate=(time.time()-t_run)/done_sh; left=len(todo)-done_sh
        nok=len(jobs)-len(pending)
        print(f"  -> {outp}: {n_out} recs, faithful {nok}/{len(jobs)} "
              f"({100*nok/len(jobs):.1f}%), {el/60:.1f}분 | 남은 shard {left}개, "
              f"ETA {left*rate/3600:.1f}시간", flush=True)
    print("DONE")

if __name__=="__main__":
    main()