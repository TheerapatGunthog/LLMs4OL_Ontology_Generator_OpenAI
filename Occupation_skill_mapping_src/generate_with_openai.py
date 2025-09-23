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
    for c in prefs:
        if c in df.columns:
            return c
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


# --- Task A: skill -> multiple occupations, strict from LLM only (no backfill) ---
def taskA_map_skills_to_occ(skills_csv: str, occ_csv: str, out_dir: str, model: str) -> List[str]:
    df_s = pd.read_csv(skills_csv)
    df_o = pd.read_csv(occ_csv)
    s_col = pick_col(df_s, ["skill", "skills", "name", "title"])
    o_col = pick_col(df_o, ["occupation", "job", "title", "name"])

    skills = [str(x).strip() for x in df_s[s_col].dropna().tolist() if str(x).strip()]
    occup = list(dict.fromkeys([str(x).strip() for x in df_o[o_col].dropna().tolist() if str(x).strip()]))

    client = get_client()
    lbls = ", ".join(occup)
    tmpl = (
        "Skill: {skill}\n"
        "Occupation labels: {labels}\n"
        "Answer with ALL matching occupations as a JSON array of strings. "
        "If uncertain, include multiple plausible occupations."
    )

    data = []
    dbg_dir = os.path.join(out_dir, "_debug_raw")
    ensure_dir(dbg_dir)

    for s in tqdm(skills, desc="TaskA: skills→occupations"):
        u = tmpl.format(skill=s, labels=lbls)
        txt = ""
        try:
            txt = chat(client, model, SYS_A, u, 512)
            labs = _extract_json_array(txt)
        except Exception:
            labs = None

        # debug raw
        try:
            with open(os.path.join(dbg_dir, f"{_slug(s)[:80]}.txt"), "w", encoding="utf-8") as f:
                f.write(txt if isinstance(txt, str) else str(txt))
        except Exception:
            pass

        if not labs:
            # strict: ไม่ backfill อัตโนมัติ
            labs = []

        # keep only valid occupations
        labs = [lab for lab in labs if lab in occup]

        # collect strictly what LLM gave
        for lab in labs:
            data.append({"text": s, "label": lab})

    # legacy TaskA artifacts
    dumpj(f"{out_dir}/data.json", data)
    dumpj(f"{out_dir}/label_mapper.json", {str(i): l for i, l in enumerate(occup)})
    dumpj(
        f"{out_dir}/templates.json",
        [
            "Classify the SKILL into one or more OCCUPATION labels. Term: {text}. "
            "Labels: {labels}. Answer with a JSON array of labels."
        ],
    )

    # Build occ2skills, dedupe per occupation, allow cross-occupation duplicates
    occ2skills = {occ: [] for occ in occup}
    for r in data:
        occ2skills[r["label"]].append(r["text"])
    occ2skills = {occ: sorted(dict.fromkeys(sk_list)) for occ, sk_list in occ2skills.items()}

    dumpj(f"{out_dir}/occ2skills.json", [{"occupation": occ, "skills": sks} for occ, sks in occ2skills.items()])

    # Entities and stats (strictly as mapped)
    ents = [{"entity": sk, "type": occ} for occ, sks in occ2skills.items() for sk in sks]
    per = {occ: len(occ2skills[occ]) for occ in occup}
    dumpj(f"{os.path.dirname(out_dir)}/jobskillsset_entities.json", ents)
    dumpj(
        f"{os.path.dirname(out_dir)}/stats.json",
        {
            "kb_name": "JobSkillsSet",
            "n_entities": len(ents),
            "n_types": len(occup),
            "per_type": per
        },
    )

    return occup


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


# --- CLI ---
def main():
    args_dict = {
        "occupations_csv": "occ_update.csv",
        "skills_csv": "skill_update.csv",
        "out_root": "Occupations_Skills_Mapping",
        "model": "gpt-4o-mini",
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

    print(f"Done → {args.out_root}")


if __name__ == "__main__":
    main()
