# -*- coding: utf-8 -*-
"""11-muscle inventory (order is FIXED everywhere) + ground-truth anatomical functions.

MUS order matches the ArtiSynth Badin tongue model's exciter list and every
pool / meta / npy column layout in the pipeline. Do not reorder.
"""

MUS = ["GGP", "GGM", "GGA", "STY", "GH", "MH", "HG", "VERT", "TRANS", "IL", "SL"]
MI = {m: i for i, m in enumerate(MUS)}
D = len(MUS)

# Ground-truth anatomy (drives QA muscle attribution and physics->articulation reasoning)
FUNC = {
"GGP":{"name":"genioglossus posterior","ko":"설근 후방","action_en":"elevates & advances the tongue dorsum/body","action_ko":"설배를 상승·전진","dir":["advance","elevate_back"]},
"GGM":{"name":"genioglossus medial","ko":"설근 중부","action_en":"elevates the tongue body","action_ko":"설체를 상승","dir":["elevate"]},
"GGA":{"name":"genioglossus anterior","ko":"설근 전부","action_en":"lowers the anterior tongue and grooves/advances the tip","action_ko":"전방을 낮추고 홈을 만들며 설첨을 전진","dir":["advance","lower_front"]},
"STY":{"name":"styloglossus","ko":"경돌설근","action_en":"retracts and elevates the back of the tongue","action_ko":"설배를 후퇴·거상","dir":["retract","elevate_back"]},
"GH":{"name":"geniohyoid","ko":"이설골근","action_en":"draws the hyoid up and forward","action_ko":"설골을 전상방으로 당김","dir":["advance"]},
"MH":{"name":"mylohyoid","ko":"악설골근","action_en":"elevates the floor of the mouth / tongue body","action_ko":"구강저·설체를 거상","dir":["elevate"]},
"HG":{"name":"hyoglossus","ko":"설골설근","action_en":"depresses and retracts the tongue body","action_ko":"설체를 하강·후퇴","dir":["depress","retract"]},
"VERT":{"name":"verticalis","ko":"수직근","action_en":"flattens/thins the tongue vertically","action_ko":"혀를 수직으로 눌러 납작하게","dir":["flatten"]},
"TRANS":{"name":"transversus","ko":"횡근","action_en":"narrows the tongue side-to-side, elongating it up/forward","action_ko":"좌우로 좁혀 상방·전방으로 늘림","dir":["narrow","advance"]},
"IL":{"name":"inferior longitudinal","ko":"하종근","action_en":"lowers the tongue tip and shortens/retracts","action_ko":"설첨을 낮추고 혀를 단축·후퇴","dir":["tip_down","retract"]},
"SL":{"name":"superior longitudinal","ko":"상종근","action_en":"raises the tongue tip and shortens","action_ko":"설첨을 올리고 혀를 단축","dir":["tip_up"]},
}

# Full display names used verbatim inside QA text (surface-form contract:
# the faithfulness check requires these exact strings to survive naturalization).
FULL_KO = {
    "GGP": "이설근 후부(genioglossus posterior)", "GGM": "이설근 중부(genioglossus medius)",
    "GGA": "이설근 전부(genioglossus anterior)", "STY": "경돌설근(styloglossus)",
    "GH": "이설골근(geniohyoid)", "MH": "악설골근(mylohyoid)", "HG": "설골설근(hyoglossus)",
    "VERT": "수직근(verticalis)", "TRANS": "횡근(transversus)",
    "IL": "하종설근(inferior longitudinal)", "SL": "상종설근(superior longitudinal)",
}
FULL_EN = {
    "GGP": "genioglossus posterior (GGP)", "GGM": "genioglossus medius (GGM)",
    "GGA": "genioglossus anterior (GGA)", "STY": "styloglossus (STY)",
    "GH": "geniohyoid (GH)", "MH": "mylohyoid (MH)", "HG": "hyoglossus (HG)",
    "VERT": "verticalis (VERT)", "TRANS": "transversus (TRANS)",
    "IL": "inferior longitudinal (IL)", "SL": "superior longitudinal (SL)",
}
