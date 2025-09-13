
import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Any

# OpenAI SDK v1
from openai import OpenAI

load_dotenv()


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: str, s: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


# ------------------------
# Prompt helpers
# ------------------------
SYS_A = "You are an ontology labeler. Given a term, assign exactly ONE type label from the provided closed set. Return ONLY the label string."
SYS_B = "You are an ontology taxonomy assistant. Given two TYPE labels, answer YES if the first is-a child of the second; otherwise NO. Return only YES or NO."
SYS_C = "You are an ontology relation classifier. Given two TYPE labels and a closed set of non-taxonomic relations, return exactly ONE label from the set."


def chat_once(client: OpenAI, model: str, system: str, user: str, max_tokens: int = 16) -> str:
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content.strip()

# ------------------------
# Task A generation (entity -> type)
# ------------------------


def gen_taskA_from_csv(csv_path: str, name_pref: List[str], label_space: List[str], out_dir: str, model: str):
    df = pd.read_csv(csv_path)
    # pick a name column
    col = None
    for c in name_pref:
        if c in df.columns:
            col = c
            break
    if col is None:
        # fallback: first object column
        candidates = [c for c in df.columns if df[c].dtype == object]
        col = candidates[0] if candidates else df.columns[0]

    terms = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    client = get_client()
    labels_str = ", ".join(label_space)
    tmpl = "Term: {term}\nType labels: {labels}\nAnswer with one label only."
    out_data = []
    for t in tqdm(terms, desc=f"A:{os.path.basename(csv_path)}"):
        user = tmpl.format(term=t, labels=labels_str)
        try:
            lab = chat_once(client, model, SYS_A, user, max_tokens=8)
        except Exception :
            time.sleep(2)
            lab = chat_once(client, model, SYS_A, user, max_tokens=8)
        if lab not in label_space:
            lab = label_space[0]
        out_data.append({"text": t, "label": lab})

    write_json(os.path.join(out_dir, "data.json"), out_data)
    write_json(os.path.join(out_dir, "label_mapper.json"), {str(i): l for i, l in enumerate(label_space)})
    write_json(os.path.join(out_dir, "templates.json"), [
        "Classify the term into one of the labels. Term: {text}. Labels: {labels}. Answer with one label only."
    ])

# ------------------------
# Task B generation (type -> type, is-a)
# ------------------------


def gen_taskB(types: List[str], seed_edges: List[List[str]], out_dir: str, model: str):
    """Confirm or expand is-a edges among provided types using seeds as candidates."""
    client = get_client()
    tmpl = "Child TYPE: {c}\nParent TYPE: {p}\nAnswer only YES or NO."
    confirmed = []
    # confirm seeds
    for c, p in tqdm(seed_edges, desc="B:seeds"):
        user = tmpl.format(c=c, p=p)
        ans = chat_once(client, model, SYS_B, user, max_tokens=2).upper()
        if ans.startswith("Y"):
            confirmed.append([c, p])

    # Optional: try more pairs by sampling cross product limited
    # Keep it simple: only use confirmed seeds
    write_json(os.path.join(out_dir, "pairs.json"), [{"child": c, "parent": p} for c, p in confirmed])
    write_json(os.path.join(out_dir, "label_mapper.json"), {"0": "is-a"})
    write_json(os.path.join(out_dir, "template.json"), [
        "Decide if child is-a parent. Answer only 'yes' or 'no'.\nChild: {child}\nParent: {parent}\nAnswer:"
    ])

# ------------------------
# Task C generation (type -> type, non-taxonomic)
# ------------------------


def gen_taskC(types: List[str], relation_labels: List[str], out_dir: str, model: str, max_pairs: int = 100):
    client = get_client()
    tmpl = "Head TYPE: {h}\nTail TYPE: {t}\nLabels: {labels}\nAnswer with one label only."
    labels_str = ", ".join(relation_labels)

    # build candidate pairs (upper triangle)
    cand = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            cand.append((types[i], types[j]))
    cand = cand[:max_pairs]

    results = []
    for h, t in tqdm(cand, desc="C:pairs"):
        user = tmpl.format(h=h, t=t, labels=labels_str)
        lab = chat_once(client, model, SYS_C, user, max_tokens=8)
        if lab not in relation_labels:
            lab = relation_labels[0]
        results.append({"head": h, "tail": t, "label": lab})

    write_json(os.path.join(out_dir, "pairs.json"), results)
    write_json(os.path.join(out_dir, "label_mapper.json"), {str(i): l for i, l in enumerate(relation_labels)})
    write_json(os.path.join(out_dir, "templates.json"), [
        "Classify the relation between Head and Tail strictly as one of: {labels}.\nHead: {head}\nTail: {tail}\nAnswer:"
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects_csv", required=True)
    ap.add_argument("--skills_csv", required=True)
    ap.add_argument("--out_root", default="datasets")
    ap.add_argument("--model", default="gpt-4o-mini")
    # closed label spaces (editable)
    ap.add_argument("--subject_labels", nargs="+",
                    default=["Core CS", "Applied CS", "Software Eng", "Networks", "Databases", "HCI", "Math", "Other"])
    ap.add_argument("--skill_labels", nargs="+",
                    default=["Programming Language", "Framework/Library", "Tool/Platform", "Soft Skill", "Domain Skill", "Other"])
    ap.add_argument("--subject_seed_edges", type=str,
                    default='[["Applied CS","Core CS"],["Software Eng","Applied CS"],["Networks","Applied CS"],["Databases","Applied CS"],["HCI","Applied CS"],["Math","Core CS"]]')
    ap.add_argument("--skill_seed_edges", type=str,
                    default='[["Framework/Library","Programming Language"],["Domain Skill","Applied CS"],["Tool/Platform","Software Eng"]]')
    ap.add_argument("--subject_rel_labels", nargs="+",
                    default=["related_to", "prerequisite_of", "applies_to"])
    ap.add_argument("--skill_rel_labels", nargs="+",
                    default=["used_for", "part_of", "related_to"])
    args = ap.parse_args()

    out_root = args.out_root
    # --- Task A for subjects ---
    subj_A_dir = os.path.join(out_root, "SubjectsSet", "TaskA")
    gen_taskA_from_csv(args.subjects_csv, ["subject_name_en", "subject", "course", "name", "title"], args.subject_labels, subj_A_dir, args.model)
    # load generated A to get the discovered type set
    subj_data = json.load(open(os.path.join(subj_A_dir, "data.json"), "r", encoding="utf-8"))
    subj_types = sorted(set([d["label"] for d in subj_data if d["label"] != "Other"]))

    # --- Task B for subjects ---
    subj_seed = json.loads(args.subject_seed_edges)
    subj_B_dir = os.path.join(out_root, "SubjectsSet", "TaskB")
    gen_taskB(subj_types, subj_seed, subj_B_dir, args.model)

    # --- Task C for subjects ---
    subj_C_dir = os.path.join(out_root, "SubjectsSet", "TaskC")
    gen_taskC(subj_types, args.subject_rel_labels, subj_C_dir, args.model)

    # --- Task A for skills ---
    skills_A_dir = os.path.join(out_root, "JobSkillsSet", "TaskA")
    gen_taskA_from_csv(args.skills_csv, ["skill", "skills", "name", "title"], args.skill_labels, skills_A_dir, args.model)
    skills_data = json.load(open(os.path.join(skills_A_dir, "data.json"), "r", encoding="utf-8"))
    skill_types = sorted(set([d["label"] for d in skills_data if d["label"] != "Other"]))

    # --- Task B for skills ---
    skill_seed = json.loads(args.skill_seed_edges)
    skills_B_dir = os.path.join(out_root, "JobSkillsSet", "TaskB")
    gen_taskB(skill_types, skill_seed, skills_B_dir, args.model)

    # --- Task C for skills ---
    skills_C_dir = os.path.join(out_root, "JobSkillsSet", "TaskC")
    gen_taskC(skill_types, args.skill_rel_labels, skills_C_dir, args.model)

    print(f"Done. Wrote datasets under: {out_root}/SubjectsSet and {out_root}/JobSkillsSet")


if __name__ == "__main__":
    main()
