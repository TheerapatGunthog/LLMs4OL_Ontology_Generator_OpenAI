# generate_relations_json_fixed.py
# CSV → JSON (force 100% coverage of CSV skills)
# Requirements: pip install openai python-dotenv
# Run example:
#   echo "OPENAI_API_KEY=sk-..." > .env
#   python generate_relations_json_fixed.py --csv skill_update.csv --out out/relations.json

import os
import csv
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# ========== ENV ==========
load_dotenv()
k = os.getenv("OPENAI_API_KEY")
if not k:
    raise SystemExit("OPENAI_API_KEY missing in environment/.env")
client = OpenAI(api_key=k)

# ========== IO ==========


def read_skills(csv_path: Path) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if "skill" not in rdr.fieldnames:
            raise SystemExit("CSV must contain column 'skill'")
        vals = [(r["skill"] or "").strip() for r in rdr if (r.get("skill") or "").strip()]
    seen = {}
    for s in vals:
        seen.setdefault(s.lower(), s)
    skills = list(seen.values())
    if not skills:
        raise SystemExit("No skills found in CSV")
    return skills


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ========== JSON helper ==========


def parse_json_block(txt: str) -> Dict[str, Any]:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", txt)
    if m:
        txt = m.group(1).strip()
    s = txt.find("{")
    e = txt.rfind("}")
    if s != -1 and e != -1 and e > s:
        txt = txt[s:e + 1]
    return json.loads(txt)


# ========== Stage 1 ==========
SYS1 = """You will receive a NUMBERED, CLOSED list of skills.

RULES:
- Select 5–20 items as 'superclasses' by INDEX only.
- NEVER invent or rename. Use only provided indices.
- For each selected index, provide:
  {"idx": <int>, "reason": "<one sentence>", "max_sub_per_super": <int in [5,25]>}

Return ONLY JSON:
{ "superclasses": [ {"idx":..., "reason":"...", "max_sub_per_super":...}, ... ] }"""

USR1_TMPL = """skills_numbered = {skills_numbered}
Return JSON only."""


def stage1_propose(skills: List[str], model="gpt-4o", temperature=0.0) -> Dict[str, Any]:
    skills_numbered = [{"idx": i, "name": s} for i, s in enumerate(skills)]
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYS1},
            {"role": "user", "content": USR1_TMPL.format(
                skills_numbered=json.dumps(skills_numbered, ensure_ascii=False)
            )}
        ]
    )
    return parse_json_block(resp.choices[0].message.content)


def validate_stage1(payload: Dict[str, Any], skills_list: List[str]) -> List[Dict[str, Any]]:
    items = payload.get("superclasses")
    if not isinstance(items, list) or not (5 <= len(items) <= 20):
        raise SystemExit("Stage1 invalid: require 5–20 superclasses")
    seen_idx, out = set(), []
    for it in items:
        idx = int(it["idx"])
        if not (0 <= idx < len(skills_list)):
            raise SystemExit(f"Stage1 idx out of range: {idx}")
        if idx in seen_idx:
            raise SystemExit(f"Stage1 duplicate idx: {idx}")
        seen_idx.add(idx)
        name = skills_list[idx]
        cap = int(it.get("max_sub_per_super", 10))
        cap = max(5, min(25, cap))
        out.append({"name": name, "reason": it.get("reason", ""), "max_sub_per_super": cap})
    return out


# ========== Stage 2A ==========
SYS2A = """Group related subclasses from a CLOSED vocabulary.

INPUT:
- allowed_subclass_pool: CLOSED list of skills (strings).

RULES:
- Use ONLY items verbatim from allowed_subclass_pool.
- Partition them into logical groups; each group has one-sentence 'reason'.
- No invention/renaming. No duplicates within a group.
- EVERY skill in allowed_subclass_pool MUST appear IN AT LEAST ONE group's 'subclasses' (no omissions).
- If a skill doesn't fit, create a singleton group.

OUTPUT JSON ONLY:
{ "groups": [ {"reason":"...", "subclasses":["...","..."]} ] }"""

USR2A_TMPL = """allowed_subclass_pool = {sub_pool}

Instructions:
- Make 5–25 groups depending on diversity.
- Each group's 'subclasses' must be DISTINCT and drawn only from allowed_subclass_pool.
- Ensure EVERY skill appears at least once (use singleton groups if needed).
Return valid JSON only."""


def stage2_assign_groups(sub_pool: List[str], model="gpt-4o", temperature=0.0):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYS2A},
            {"role": "user", "content": USR2A_TMPL.format(
                sub_pool=json.dumps(sub_pool, ensure_ascii=False)
            )}
        ]
    )
    return parse_json_block(resp.choices[0].message.content)


# ========== Stage 2B ==========
SYS2B = """Create subclass↔subclass links over a CLOSED set.

INPUT:
- allowed_link_pool: list of subclasses already assigned to some group. CLOSED set.

RULES:
- Links connect ONLY items from allowed_link_pool.
- Cross-group links allowed.
- No self-loops. Treat {{a,b}} as unordered. Keep links UNIQUE.

OUTPUT JSON ONLY:
{ "subclass_links": [ {"a":"...","b":"...","reason":"..."} ] }"""

USR2B_TMPL = """allowed_link_pool = {link_pool}

Instructions:
- Propose up to 200 UNIQUE, meaningful links among allowed_link_pool only.
- No self-links. Treat {{a,b}} as unordered.
Return valid JSON only."""


def stage2_build_links(assigned_subs: List[str], model="gpt-4o", temperature=0.0):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYS2B},
            {"role": "user", "content": USR2B_TMPL.format(
                link_pool=json.dumps(sorted(assigned_subs), ensure_ascii=False)
            )}
        ]
    )
    return parse_json_block(resp.choices[0].message.content)

# ========== Packing ==========


def pack_subclasses_into_superclasses(supers, groups_only, subclass_pool):
    """Distribute all subclasses into all superclasses (cover 100%)."""
    sup_names = [s["name"] for s in supers]
    flat = []
    for g in groups_only:
        for s in g.get("subclasses", []):
            if s not in flat:
                flat.append(s)
    for s in subclass_pool:
        if s not in flat:
            flat.append(s)

    out_groups = {n: {"superclass": n, "reason": s["reason"], "subclasses": []} for s, n in zip(supers, sup_names)}
    idx = 0
    for sub in flat:
        out_groups[sup_names[idx]]["subclasses"].append(sub)
        idx = (idx + 1) % len(sup_names)
    return list(out_groups.values())


def dedup_links(payload, allowed_nodes):
    out, seen = [], set()
    for l in payload.get("subclass_links", []):
        a = l.get("a", "").strip()
        b = l.get("b", "").strip()
        if not a or not b or a == b:
            continue
        if a not in allowed_nodes or b not in allowed_nodes:
            continue
        key = tuple(sorted([a.lower(), b.lower()]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"a": a, "b": b, "reason": l.get("reason", "")})
    return out

# ========== MAIN ==========


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    skills = read_skills(args.csv)

    # Stage1
    st1 = stage1_propose(skills, model=args.model, temperature=args.temperature)
    supers = validate_stage1(st1, skills)
    with open("out_debug/stage1_superclasses.json", "w", encoding="utf-8") as f:
        json.dump(supers, f, ensure_ascii=False, indent=2)

    sup_names = [s["name"] for s in supers]
    subclass_pool = sorted(set(skills) - set(sup_names))

    # Stage2A
    st2a = stage2_assign_groups(subclass_pool, model=args.model, temperature=args.temperature)
    groups_only = st2a["groups"]

    # Packing ensures full coverage
    groups_final = pack_subclasses_into_superclasses(supers, groups_only, subclass_pool)
    assigned_final = sorted({s for g in groups_final for s in g["subclasses"]})

    # Stage2B
    st2b = stage2_build_links(assigned_final, model=args.model, temperature=args.temperature)
    links_final = dedup_links(st2b, set(assigned_final))

    ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"groups": groups_final, "subclass_links": links_final}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
