import os
import json
import pandas as pd
from typing import List, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
import re
from types import SimpleNamespace  # ← เพิ่ม

load_dotenv()


def get_client():
    k = os.getenv("OPENAI_API_KEY")
    b = os.getenv("OPENAI_BASE_URL") or None
    if not k:
        raise RuntimeError("OPENAI_API_KEY missing")
    return OpenAI(api_key=k, base_url=b) if b else OpenAI(api_key=k)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def dumpj(p: str, o: Any):
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)


# --- Prompts ---
SYS_A = (
    "You are an ontology labeler. Given a SKILL term and a closed set of OCCUPATION labels, "
    "assign ALL occupations from the list that could plausibly require the skill. "
    "Be generous: if a skill might be relevant to multiple occupations, include them all. "
    "Return strictly a JSON array of strings from the list. Never return an empty array."
)
SYS_B = (
    "You are an ontology taxonomy assistant. Given two OCCUPATION labels, answer YES if the "
    "first is-a specialization/subclass of the second; otherwise NO. Return only YES or NO."
)
SYS_C = (
    "You are an ontology relation classifier. Given two OCCUPATION labels and a closed set of non-taxonomic "
    "relations (e.g., collaborates_with, depends_on, related_to), return exactly ONE label."
)

# Stronger auto taxonomy prompt (JSON-only)
AUTO_SYS_B = (
    "You design occupation taxonomies.\n"
    "Given a list of OCCUPATION labels, create HIGH-LEVEL PARENT categories and map each child to exactly ONE parent.\n"
    "OUTPUT STRICTLY JSON ONLY: an array of [child, parent] pairs. No commentary."
)


def chat(client: OpenAI, model: str, sys: str, usr: str, mt: int = 512) -> str:
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0,
        max_tokens=mt,
    )
    return r.choices[0].message.content.strip()


def pick_col(df: pd.DataFrame, prefs: List[str]) -> str:
    # normalize ชื่อคอลัมน์
    cols_map = {c.lower().strip(): c for c in df.columns}
    for p in prefs:
        key = p.lower().strip()
        if key in cols_map:
            return cols_map[key]
    objs = [c for c in df.columns if df[c].dtype == object]
    return objs[0] if objs else df.columns[0]


def _extract_json_array(txt: str):
    """
    ดึง JSON array ตัวแรกจากข้อความทั้งหมดแบบ tolerant
    คืนค่า list หรือ None
    """
    m = re.search(r"\[.*?\]", txt, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else [arr]
    except Exception:
        return None


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", s).strip().replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    if not s:
        s = "unnamed"
    if s[0].isdigit():
        s = "_" + s
    return s


# --- Task A (refactored): supports NOO with 3-round majority + scoring; lower cyclomatic complexity ---
def taskA_map_skills_to_occ(skills_csv: str, occ_csv: str, out_dir: str, model: str) -> List[str]:
    df_s = pd.read_csv(skills_csv)
    df_o = pd.read_csv(occ_csv)

    # normalize header
    df_s.columns = df_s.columns.str.strip()
    df_o.columns = df_o.columns.str.strip()

    s_col = pick_col(df_s, ["skill", "skills, ", "skills", "name", "title"])
    o_col = pick_col(df_o, ["occupation", "job", "title", "name"])

    # หา NOO column แบบ robust (ไม่เปลี่ยนตรรกะการใช้ NOO)
    noo_col = next((c for c in df_o.columns if c.strip().lower() == "noo"), None)
    has_noo = noo_col is not None

    skills = _dedupe_str(df_s[s_col])
    occup = _dedupe_str(df_o[o_col])

    ensure_dir(out_dir)

    if not has_noo:
        occ2skills = _taskA_fallback_original(skills, occup, out_dir, model)
        _persist_occ2skills(out_dir, occ2skills, occup)
        _persist_entities_stats(out_dir, occ2skills, occup)
        return occup

    client = get_client()
    df_noo = df_o[[o_col, noo_col]].copy()
    df_noo[noo_col] = pd.to_numeric(df_noo[noo_col], errors="coerce").fillna(0).astype(int).clip(lower=0)

    # log จะอยู่ใต้ TaskA/<kb>/_debug_logs/
    log_dir = os.path.join(out_dir, "_debug_logs")
    ensure_dir(log_dir)

    occ2skills = {o: [] for o in occup}
    for occ_name, noo in tqdm(df_noo.itertuples(index=False), total=len(df_noo), desc="TaskA: occupation→skills"):
        if noo <= 0:
            continue
        selected = _select_skills_for_occ(occ_name, noo, skills, client, model, log_dir)
        occ2skills[occ_name] = selected

    _persist_occ2skills(out_dir, occ2skills, occup)
    _persist_templates_noo(out_dir)
    _persist_entities_stats(out_dir, occ2skills, occup)
    return occup


# ---------- helpers (small, single-purpose) ----------

def _dedupe_str(series: pd.Series) -> List[str]:
    vals = [str(x).strip() for x in series.dropna().tolist() if str(x).strip()]
    return list(dict.fromkeys(vals))


def _safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)[:80]


def _taskA_fallback_original(skills: List[str], occup: List[str], out_dir: str, model: str) -> dict:
    client = get_client()
    lbls = ", ".join(occup)
    tmpl = (
        "Skill: {skill}\n"
        "Occupation labels: {labels}\n"
        "Answer with ALL matching occupations as a JSON array of strings. "
        "If uncertain, include multiple plausible occupations."
    )
    data, dbg_dir = [], os.path.join(out_dir, "_debug_raw")
    ensure_dir(dbg_dir)

    for s in tqdm(skills, desc="TaskA: skills→occupations"):
        u = tmpl.format(skill=s, labels=lbls)
        txt = chat(client, model, SYS_A, u, 512)
        labs = _extract_json_array(txt) or []
        labs = [lab for lab in labs if lab in occup]
        _safe_debug_write(os.path.join(dbg_dir, f"{_safe_name(s)}.txt"), txt)
        for lab in labs:
            data.append({"text": s, "label": lab})

    dumpj(f"{out_dir}/data.json", data)
    dumpj(f"{out_dir}/label_mapper.json", {str(i): l for i, l in enumerate(occup)})
    dumpj(
        f"{out_dir}/templates.json",
        [
            "Classify the SKILL into one or more OCCUPATION labels. Term: {text}. "
            "Labels: {labels}. Answer with a JSON array of labels."
        ],
    )
    occ2skills = {occ: [] for occ in occup}
    for r in data:
        occ2skills[r["label"]].append(r["text"])
    return {occ: sorted(dict.fromkeys(v)) for occ, v in occ2skills.items()}


def _safe_debug_write(path: str, txt: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt if isinstance(txt, str) else str(txt))
    except Exception:
        pass


def _dump_log(log_dir: str, occ_name: str, logs: dict):
    ensure_dir(log_dir)
    fname = _safe_name(occ_name) + ".json"
    dumpj(os.path.join(log_dir, fname), logs)


def _select_skills_for_occ(occ_name: str, noo: int, skills: List[str], client, model: str, log_dir: str) -> List[str]:
    target = min(noo, len(skills))
    logs = {"occupation": occ_name, "target_NOO": noo, "step1": {}, "step2": [], "step3": []}

    # Step 1: Majority vote (≥2/3)
    rounds = _run_rounds(occ_name, skills, client, model, n_rounds=3)
    logs["step1"]["rounds"] = rounds
    majority = _majority_at_least(rounds, k=2)
    selected = _ordered_intersection(skills, majority)
    logs["step1"]["majority_selected"] = selected.copy()
    if len(selected) >= target:
        _dump_log(log_dir, occ_name, logs)
        return selected[:target]

    # Step 2: เติมจาก union + ให้คะแนน
    union = _ordered_union(rounds)
    extras_from_union = [s for s in union if s not in selected]
    if extras_from_union:
        for sc, sk in _score_and_sort(occ_name, extras_from_union, client, model):
            if len(selected) >= target:
                break
            selected.append(sk)
            logs["step2"].append({"skill": sk, "score": sc})
    if len(selected) >= target:
        _dump_log(log_dir, occ_name, logs)
        return selected[:target]

    # Step 3: เติมจากที่เหลือทั้งหมด + ให้คะแนน
    remaining = [s for s in skills if s not in selected]
    if remaining:
        for sc, sk in _score_and_sort(occ_name, remaining, client, model):
            if len(selected) >= target:
                break
            selected.append(sk)
            logs["step3"].append({"skill": sk, "score": sc})

    _dump_log(log_dir, occ_name, logs)
    return selected[:target]


def _run_rounds(occ_name: str, skills: List[str], client, model: str, n_rounds: int) -> List[List[str]]:
    sys_prompt = (
        "You are a skills extractor. Given ONE OCCUPATION and a CLOSED SET of SKILL labels, "
        "return ONLY the relevant skills for that occupation. "
        "Return STRICTLY a JSON array of strings. Use ONLY skills from the provided set."
    )
    usr_prefix = f"Occupation: {occ_name}\nClosed skill set ({len(skills)} items):\n" + "\n".join(f"- {s}" for s in skills) + "\nAnswer with a JSON array containing only items from the set."
    out = []
    for _ in range(n_rounds):
        txt = chat(client, model, sys_prompt, usr_prefix, mt=1024)
        arr = _extract_json_array(txt) or []
        out.append([s for s in arr if s in skills])
    return out


def _majority_at_least(rounds: List[List[str]], k: int) -> List[str]:
    freq: dict = {}
    for arr in rounds:
        for s in set(arr):
            freq[s] = freq.get(s, 0) + 1
    return [s for s, c in freq.items() if c >= k]


def _ordered_union(rounds: List[List[str]]) -> List[str]:
    seen, out = set(), []
    for arr in rounds:
        for s in arr:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _ordered_intersection(reference: List[str], subset: List[str]) -> List[str]:
    subset_set = set(subset)
    return [s for s in reference if s in subset_set]


def _score_and_sort(occ_name: str, skills: List[str], client, model: str) -> List[tuple]:
    sys_prompt = "You are a judge. Given an OCCUPATION and ONE SKILL, output an integer relevance score 0-100. 0 means unrelated. 100 means essential. Return ONLY the integer."

    def _score(skill_name: str) -> int:
        txt = chat(client, model, sys_prompt, f"Occupation: {occ_name}\nSkill: {skill_name}\nScore 0-100:", mt=8)
        m = re.search(r"\d{1,3}", txt or "")
        v = int(m.group(0)) if m else 0
        return max(0, min(100, v))
    scored = [(_score(sk), sk) for sk in skills]
    scored.sort(reverse=True)
    return scored


def _persist_occ2skills(out_dir: str, occ2skills: dict, occup: List[str]) -> None:
    dumpj(f"{out_dir}/occ2skills.json", [{"occupation": occ, "skills": sks} for occ, sks in occ2skills.items()])
    dumpj(f"{out_dir}/data.json", [{"text": sk, "label": occ} for occ, sks in occ2skills.items() for sk in sks])
    dumpj(f"{out_dir}/label_mapper.json", {str(i): l for i, l in enumerate(occup)})


def _persist_templates_noo(out_dir: str) -> None:
    dumpj(
        f"{out_dir}/templates.json",
        [
            "Select all relevant SKILLs for an OCCUPATION from a CLOSED SET. "
            "Return a JSON array of strings chosen only from the provided set."
        ],
    )


def _persist_entities_stats(out_dir: str, occ2skills: dict, occup: List[str]) -> None:
    ents = [{"entity": sk, "type": occ} for occ, sks in occ2skills.items() for sk in sks]
    per = {occ: len(occ2skills.get(occ, [])) for occ in occup}
    dumpj(f"{os.path.dirname(out_dir)}/jobskillsset_entities.json", ents)
    dumpj(f"{os.path.dirname(out_dir)}/stats.json", {"kb_name": "JobSkillsSet", "n_entities": len(ents), "n_types": len(occup), "per_type": per})


# --- Helpers for Task B auto seeds ---
def _safe_json_array(txt: str):
    m = re.search(r"\[.*?\]", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_propose_taxonomy_seeds(client: OpenAI, model: str, occ_types: List[str]) -> List[List[str]]:
    usr = "Occupations:\n" + "\n".join(f"- {o}" for o in occ_types)

    # try LLM up to 2 attempts
    for _ in range(2):
        txt = chat(client, model, AUTO_SYS_B, usr, mt=1024)
        arr = _safe_json_array(txt)
        ok = []
        if isinstance(arr, list):
            for p in arr:
                if isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) and x.strip() for x in p):
                    ok.append([p[0].strip(), p[1].strip()])
        if ok:
            return ok

    # heuristic fallback: simple buckets
    buckets = {
        "Software & Engineering": ("Engineer", "Developer", "Programmer", "Architect", "Software"),
        "Analytics": ("Analyst", "Science", "Scientist", "BI", "Intelligence"),
        "IT Operations": ("Administrator", "Network", "Systems"),
        "Management": ("Manager", "Project", "Program", "Product"),
    }
    pairs = []
    for o in occ_types:
        parent = None
        low = o.lower()
        for b, kws in buckets.items():
            if any(k.lower() in low for k in kws):
                parent = b
                break
        if not parent:
            parent = "Other"
        pairs.append([o, parent])
    return pairs


# --- Task B (robust): accept seeds directly; if empty, map to generic parent ---
def taskB_occ_taxonomy(
    occ_types: List[str], out_dir: str, model: str, seeds: Optional[List[List[str]]] = None
):
    pairs = []
    if seeds:
        pairs = [{"child": c, "parent": p} for c, p in seeds]
    else:
        pairs = [{"child": c, "parent": "Occupation"} for c in occ_types]

    dumpj(f"{out_dir}/pairs.json", pairs)
    dumpj(f"{out_dir}/label_mapper.json", {"0": "is-a"})
    dumpj(
        f"{out_dir}/template.json",
        ["Decide if child is-a parent. Answer only 'yes' or 'no'.\nChild: {child}\nParent: {parent}\nAnswer:"],
    )


# --- Task C ---
def taskC_occ_relations(
    occ_types: List[str], out_dir: str, model: str, rel_labels: List[str], max_pairs: int = 120
):
    client = get_client()
    lbls = ", ".join(rel_labels)
    tmpl = "Head OCCUPATION: {h}\nTail OCCUPATION: {t}\nLabels: {labels}\nAnswer with one label only."
    pairs = []
    for i in range(len(occ_types)):
        for j in range(i + 1, len(occ_types)):
            pairs.append((occ_types[i], occ_types[j]))
    pairs = pairs[:max_pairs]
    out = []
    for h, t in tqdm(pairs, desc="TaskC: relations"):
        lab = chat(client, model, SYS_C, tmpl.format(h=h, t=t, labels=lbls), 8)
        if lab not in rel_labels:
            lab = rel_labels[0]
        out.append({"head": h, "tail": t, "label": lab})
    dumpj(f"{out_dir}/pairs.json", out)
    dumpj(f"{out_dir}/label_mapper.json", {str(i): l for i, l in enumerate(rel_labels)})
    dumpj(
        f"{out_dir}/templates.json",
        [
            "Classify the relation between Head and Tail strictly as one of: {labels}.\nHead: {head}\nTail: {tail}\nAnswer:"
        ],
    )


# --- Task D.1: skill ROOT skill (YES/NO) ---
def taskD_skill_root(skills_csv: str, out_dir: str, model: str, max_pairs: int = 400):
    df_s = pd.read_csv(skills_csv)
    s_col = pick_col(df_s, ["skill", "skills", "name", "title"])
    skills = [str(x).strip() for x in df_s[s_col].dropna().tolist() if str(x).strip()]
    skills = list(dict.fromkeys(skills))

    client = get_client()
    ensure_dir(out_dir)

    SYS = (
        "You are a skills ontology classifier. Given two SKILL labels, "
        "answer YES if HEAD is a ROOT or more foundational version of TAIL. "
        "Return only YES or NO."
    )
    USR_T = "HEAD SKILL: {h}\nTAIL SKILL: {t}\nAnswer YES or NO only."

    pairs = []
    for i in range(len(skills)):
        for j in range(len(skills)):
            if i == j:
                continue
            pairs.append((skills[i], skills[j]))
    pairs = pairs[:max_pairs]

    out = []
    for h, t in tqdm(pairs, desc="TaskD1: skill-root-skill"):
        lbl = chat(client, model, SYS, USR_T.format(h=h, t=t), 8).upper()
        lbl = "YES" if "YES" in lbl else "NO"
        out.append({"head": h, "tail": t, "label": lbl})

    dumpj(f"{out_dir}/pairs.json", out)
    dumpj(f"{out_dir}/label_mapper.json", {"0": "NO", "1": "YES"})
    dumpj(
        f"{out_dir}/templates.json",
        ["Decide if HEAD is a ROOT of TAIL. Head: {head}  Tail: {tail}  Answer YES/NO."]
    )


# --- Task D.2: skill COMBINATION skill (YES/NO) ---
def taskD_skill_combination(skills_csv: str, out_dir: str, model: str, max_pairs: int = 400):
    df_s = pd.read_csv(skills_csv)
    s_col = pick_col(df_s, ["skill", "skills", "name", "title"])
    skills = [str(x).strip() for x in df_s[s_col].dropna().tolist() if str(x).strip()]
    skills = list(dict.fromkeys(skills))

    client = get_client()
    ensure_dir(out_dir)

    SYS = (
        "You are a skills relations judge. Given two SKILL labels, "
        "answer YES if HEAD commonly COMBINES WITH TAIL as a pair used together in real tasks. "
        "Return only YES or NO."
    )
    USR_T = "HEAD SKILL: {h}\nTAIL SKILL: {t}\nAnswer YES or NO only."

    pairs = []
    for i in range(len(skills)):
        for j in range(len(skills)):
            if i == j:
                continue
            pairs.append((skills[i], skills[j]))
    pairs = pairs[:max_pairs]

    out = []
    for h, t in tqdm(pairs, desc="TaskD2: skill-combination-skill"):
        lbl = chat(client, model, SYS, USR_T.format(h=h, t=t), 8).upper()
        lbl = "YES" if "YES" in lbl else "NO"
        out.append({"head": h, "tail": t, "label": lbl})

    dumpj(f"{out_dir}/pairs.json", out)
    dumpj(f"{out_dir}/label_mapper.json", {"0": "NO", "1": "YES"})
    dumpj(
        f"{out_dir}/templates.json",
        ["Decide if HEAD COMBINES WITH TAIL in practice. Head: {head}  Tail: {tail}  Answer YES/NO."]
    )


# --- Task D.3: skill COMPOSITION skill (YES/NO) ---
def taskD_skill_composition(skills_csv: str, out_dir: str, model: str, max_pairs: int = 400):
    df_s = pd.read_csv(skills_csv)
    s_col = pick_col(df_s, ["skill", "skills", "name", "title"])
    skills = [str(x).strip() for x in df_s[s_col].dropna().tolist() if str(x).strip()]
    skills = list(dict.fromkeys(skills))

    client = get_client()
    ensure_dir(out_dir)

    SYS = (
        "You are a skills decomposition analyst. Given two SKILL labels, "
        "answer YES if HEAD is a COMPONENT composing TAIL (i.e., TAIL is built from HEAD and others). "
        "Return only YES or NO."
    )
    USR_T = "HEAD SKILL: {h}\nTAIL SKILL: {t}\nAnswer YES or NO only."

    pairs = []
    for i in range(len(skills)):
        for j in range(len(skills)):
            if i == j:
                continue
            pairs.append((skills[i], skills[j]))
    pairs = pairs[:max_pairs]

    out = []
    for h, t in tqdm(pairs, desc="TaskD3: skill-composition-skill"):
        lbl = chat(client, model, SYS, USR_T.format(h=h, t=t), 8).upper()
        lbl = "YES" if "YES" in lbl else "NO"
        out.append({"head": h, "tail": t, "label": lbl})

    dumpj(f"{out_dir}/pairs.json", out)
    dumpj(f"{out_dir}/label_mapper.json", {"0": "NO", "1": "YES"})
    dumpj(
        f"{out_dir}/templates.json",
        ["Decide if HEAD is a COMPONENT of TAIL. Head: {head}  Tail: {tail}  Answer YES/NO."]
    )


# --- CLI ---
def main():
    args_dict = {
        "occupations_csv": "occ_update.csv",
        "skills_csv": "skill_update.csv",
        "out_root": "Occupations_Skills_Mapping",
        "model": "gpt-4.1",
        "taxonomy_seeds": 'auto',
        "taxonomy_mode": "auto",
        "relation_labels": ["related_to", "collaborates_with", "depends_on"],
    }
    # แปลง dict -> namespace เพื่อให้ใช้ dot-access ได้
    args = SimpleNamespace(**args_dict)

    # Task A
    A_dir = f"{args.out_root}/TaskA/JobSkillsSet"
    occ_types = taskA_map_skills_to_occ(args.skills_csv, args.occupations_csv, A_dir, args.model)

    import json as _json
    auto = False
    if args.taxonomy_mode == "auto" or (
        isinstance(args.taxonomy_seeds, str) and args.taxonomy_seeds.strip().lower() == "auto"
    ):
        auto = True

    # Task B seeds
    if auto:
        client = get_client()
        seeds = llm_propose_taxonomy_seeds(client, args.model, occ_types)
    else:
        try:
            seeds = _json.loads(args.taxonomy_seeds)
        except Exception:
            seeds = []

    # Debug log for seeds
    dbg_dir = f"{args.out_root}/TaskB/_debug"
    ensure_dir(dbg_dir)
    dumpj(f"{dbg_dir}/auto_seeds.json", seeds or [])

    # Task B
    B_dir = f"{args.out_root}/TaskB/Occupations"
    taskB_occ_taxonomy(occ_types, B_dir, args.model, seeds)

    # Task C
    C_dir = f"{args.out_root}/TaskC/Occupations"
    taskC_occ_relations(occ_types, C_dir, args.model, args.relation_labels)

    # Task D
    D_root = f"{args.out_root}/TaskD"
    taskD_skill_root(args.skills_csv, f"{D_root}/SkillsRoot", args.model)
    taskD_skill_combination(args.skills_csv, f"{D_root}/SkillsCombination", args.model)
    taskD_skill_composition(args.skills_csv, f"{D_root}/SkillsComposition", args.model)

    print(f"Done → {args.out_root}")


if __name__ == "__main__":
    main()
