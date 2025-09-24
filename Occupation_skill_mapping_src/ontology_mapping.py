# ontology_mapping.py
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from types import SimpleNamespace


# ---------- IO helpers ----------
def loadj(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dumpj(obj: Any, p: Path):
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- Task readers ----------
def read_entities_A(path: Path, kb: str) -> List[Dict[str, str]]:
    """
    นอร์มัลไลซ์ Task A เป็น list[{entity: skill, type: occupation}]
    รองรับ 3 รูปแบบ:
      1) occ2skills.json           -> [{"occupation": str, "skills": [str]}]
      2) data.json (legacy)        -> [{"text": <skill>, "label": <occupation>}]
      3) <kb>_entities.json (ref)  -> [{"entity": <skill>, "type": <occupation>}]
    """
    if path.name == "occ2skills.json":
        rows = loadj(path)
        ents = []
        for r in rows:
            occ = r.get("occupation")
            for sk in r.get("skills", []):
                ents.append({"entity": sk, "type": occ})
        return ents
    if path.name == "data.json":
        rows = loadj(path)
        return [{"entity": r["text"], "type": r["label"]} for r in rows]
    if path.name.endswith("_entities.json"):
        return loadj(path)
    raise FileNotFoundError(f"Unsupported TaskA file: {path}")


def read_taskB_pairs(p: Path) -> List[Dict[str, str]]:
    return loadj(p)  # [{"child":..., "parent":...}]


def read_taskC_relations(p: Path) -> List[Dict[str, str]]:
    return loadj(p)  # [{"head":..., "tail":..., "label":...}]


# ---------- Core build ----------
def build_ontology(datasets_root: Path, kb_name: str) -> Dict[str, Any]:
    """
    datasets_root: Path('Occupations_Skills_Mapping')
    kb_name: e.g., 'JobSkillsSet'
    """
    # Task A inputs live at: <root>/TaskA/<kb>/{occ2skills.json|data.json|<kb>_entities.json}
    a_new = datasets_root / "TaskA" / kb_name / "occ2skills.json"
    a_legacy = datasets_root / "TaskA" / kb_name / "data.json"
    a_ref = datasets_root / "TaskA" / kb_name / f"{kb_name.lower()}_entities.json"

    # entities
    if a_new.exists():
        ents = read_entities_A(a_new, kb_name)
    elif a_legacy.exists():
        ents = read_entities_A(a_legacy, kb_name)
    elif a_ref.exists():
        ents = read_entities_A(a_ref, kb_name)
    else:
        raise FileNotFoundError("Task A entities not found in any supported format.")

    # occupations
    if a_new.exists():
        rows = loadj(a_new)  # [{"occupation": str, "skills": [str]}]
        occupations = sorted({r.get("occupation") for r in rows if r.get("occupation")})
    else:
        occupations = sorted({e["type"] for e in ents if e.get("type")})

    # skills
    skills = sorted({e["entity"] for e in ents if e.get("entity") is not None})

    # requires edges
    requires = [
        {"occupation": e["type"], "skill": e["entity"]}
        for e in ents
        if e.get("entity") is not None and e.get("type") in occupations
    ]

    # Task B and Task C inputs live at: <root>/TaskB/Occupations/pairs.json and <root>/TaskC/Occupations/pairs.json
    b_pairs = read_taskB_pairs(datasets_root / "TaskB" / "Occupations" / "pairs.json")
    c_rels = read_taskC_relations(datasets_root / "TaskC" / "Occupations" / "pairs.json")
    isa = [{"child": r["child"], "parent": r["parent"]} for r in b_pairs]
    rels = [{"head": r["head"], "tail": r["tail"], "label": r["label"]} for r in c_rels]

    return {
        "kb_name": kb_name,
        "nodes": {"occupations": occupations, "skills": skills},
        "edges": {"requiresSkill": requires, "isA": isa, "relations": rels},
        "counts": {
            "n_occupations": len(occupations),
            "n_skills": len(skills),
            "n_requires": len(requires),
            "n_isA": len(isa),
            "n_relations": len(rels),
        },
    }


# ---------- TTL writer (Protégé-friendly) ----------
def _slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\- ]+", "", s)      # keep word, dash, space
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    if not s:
        s = "Unnamed"
    if s[0].isdigit():
        s = "_" + s
    return s


def write_ttl(onto: Dict[str, Any], out_path: Path, base_iri: str = "http://example.org/occ-skills#") -> None:
    occs = onto["nodes"]["occupations"]
    skills = onto["nodes"]["skills"]
    requires = onto["edges"]["requiresSkill"]
    isa = onto["edges"]["isA"]
    rels = onto["edges"]["relations"]
    rel_props = sorted({r["label"] for r in rels}) if rels else []

    lines: List[str] = []
    w = lines.append

    # prefixes
    w(f"@prefix ex: <{base_iri}> .")
    w("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    w("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    w("@prefix owl: <http://www.w3.org/2002/07/owl#> .")
    w("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
    w("")
    w("ex: a owl:Ontology .")
    w("")

    # classes
    w("ex:Occupation a owl:Class ; rdfs:subClassOf owl:Thing .")
    w("ex:Skill a owl:Class ; rdfs:subClassOf owl:Thing .")
    w("")

    # properties
    w("ex:requiresSkill a owl:ObjectProperty ; rdfs:domain ex:Occupation ; rdfs:range ex:Skill .")
    w("ex:isAOccupationOf a owl:ObjectProperty ; rdfs:domain ex:Occupation ; rdfs:range ex:Occupation .")
    for p in rel_props:
        pslug = _slug(p)
        w(f"ex:{pslug} a owl:ObjectProperty ; rdfs:domain ex:Occupation ; rdfs:range ex:Occupation .")
    w("")

    # occupation individuals
    for o in occs:
        oslug = _slug(o)
        w(f"ex:{oslug} a ex:Occupation ; rdfs:label \"{o}\"^^xsd:string .")
    w("")

    # skill individuals
    for s in skills:
        sslug = _slug(s)
        w(f"ex:{sslug} a ex:Skill ; rdfs:label \"{s}\"^^xsd:string .")
    w("")

    # isA edges
    occ_set = {o: _slug(o) for o in occs}
    for e in isa:
        c = e["child"]
        p = e["parent"]
        if c in occ_set and p in occ_set:
            w(f"ex:{occ_set[c]} ex:isAOccupationOf ex:{occ_set[p]} .")
    w("")

    # requiresSkill edges
    sk_set = {s: _slug(s) for s in skills}
    for e in requires:
        o = e["occupation"]
        s = e["skill"]
        if o in occ_set and s in sk_set:
            w(f"ex:{occ_set[o]} ex:requiresSkill ex:{sk_set[s]} .")
    w("")

    # non-taxonomic relations
    for r in rels:
        prop = _slug(r["label"])
        h = r["head"]
        t = r["tail"]
        if h in occ_set and t in occ_set:
            w(f"ex:{occ_set[h]} ex:{prop} ex:{occ_set[t]} .")
    w("")

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------- CLI ----------
def main():
    args = SimpleNamespace(
        datasets_root="Occupations_Skills_Mapping",
        sets=["JobSkillsSet"],
        out_name="ontology.json",
        out_ttl="ontology.ttl",
        base_iri="http://example.org/occ-skills#",
    )

    root = Path(args.datasets_root)

    for kb_name in args.sets:
        # build ontology from <root>/Task*
        onto = build_ontology(root, kb_name)

        # outputs under <root>/<kb>/ontology/
        out_json = root / kb_name / "ontology" / args.out_name
        dumpj(onto, out_json)

        out_ttl = root / kb_name / "ontology" / args.out_ttl
        write_ttl(onto, out_ttl, base_iri=args.base_iri)

        print(f"wrote {out_json}")
        print(f"wrote {out_ttl}")


if __name__ == "__main__":
    main()
