import os
import json
import time
from pathlib import Path
from types import SimpleNamespace

# --- config ---
TASKA_SUBDIR = "JobSkillsSet"
TASKB_SUBDIR = "Occupations"
TASKC_SUBDIR = "Occupations"


def loadj(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dumpj(obj, p: Path):
    os.makedirs(p.parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_taskA(set_root: Path):
    # Prefer new format: occ2skills.json (occupation -> [skills]); fallback to legacy data.json
    kb = set_root.name
    base_in = set_root / "TaskA" / TASKA_SUBDIR
    kb_out = set_root.parent / "TaskA" / kb
    os.makedirs(kb_out, exist_ok=True)

    ents = []
    per_type = {}
    unassigned = []

    occ2skills_path = base_in / "occ2skills.json"
    if occ2skills_path.exists():
        rows = loadj(occ2skills_path)  # [{"occupation": "...", "skills": [...]}]
        all_occs = [r.get("occupation") for r in rows]
        # init per_type เป็นศูนย์ก่อน
        per_type = {occ: 0 for occ in all_occs}

        for r in rows:
            occ = r.get("occupation")
            sks = r.get("skills", [])
            # เพิ่มคู่จริง
            for sk in sks:
                ents.append({"entity": sk, "type": occ})
                per_type[occ] += 1
            # ถ้าไม่มีสกิล ให้เพิ่ม row พิเศษ entity=None
            if not sks:
                unassigned.append(occ)
                ents.append({"entity": None, "type": occ})

    else:
        # legacy fallback
        a = loadj(base_in / "data.json")  # [{"text": <skill>, "label": <occupation>}]
        ents = [{"entity": r["text"], "type": r["label"]} for r in a]
        occs = sorted({r["label"] for r in a})
        per_type = {occ: 0 for occ in occs}
        for r in a:
            per_type[r["label"]] += 1
        # ไม่มีข้อมูล unassigned จากรูปแบบ legacy

    # stats
    stats = {
        "kb_name": kb,
        "n_entities": len([e for e in ents if e["entity"] is not None]),
        "n_types": len(per_type),
        "per_type": per_type,
    }
    if unassigned:
        stats["unassigned_occupations"] = unassigned

    dumpj(ents, kb_out / f"{kb.lower()}_entities.json")
    dumpj(stats, kb_out / "stats.json")

    # copy meta
    lm_src = base_in / "label_mapper.json"
    if lm_src.exists():
        dumpj(loadj(lm_src), kb_out / "label_mapper.json")
    tpl_src = base_in / "templates.json"
    if tpl_src.exists():
        dumpj(loadj(tpl_src), kb_out / "templates.json")


def export_taskB(set_root: Path, model: str = "gpt4", template_name: str = "template-1"):
    kb = set_root.name
    b_pairs = loadj(set_root / "TaskB" / TASKB_SUBDIR / "pairs.json")  # [{"child": <Occ>, "parent": <Occ>}]
    t = time.strftime("%Y-%m-%d %H:%M:%S")

    res_dir = set_root.parent / "TaskB" / "results" / kb / model
    os.makedirs(res_dir, exist_ok=True)
    dumpj(
        {
            "kb_name": kb,
            "model": model,
            "template": template_name,
            "time": t,
            "pairs": b_pairs,
            "metric": "n/a",
        },
        res_dir / f"output-{model}-{template_name}-{t}.json",
    )
    dumpj(
        {
            "kb_name": kb,
            "model": model,
            "template": template_name,
            "time": t,
            "counts": {"pairs": len(b_pairs)},
        },
        res_dir / f"report-{model}-{template_name}-{t}.json",
    )

    # copy meta
    kb_dir = set_root.parent / "TaskB" / kb
    os.makedirs(kb_dir, exist_ok=True)
    lm = set_root / "TaskB" / TASKB_SUBDIR / "label_mapper.json"
    if lm.exists():
        dumpj(loadj(lm), kb_dir / "label_mapper.json")
    tpl = set_root / "TaskB" / TASKB_SUBDIR / "template.json"
    if tpl.exists():
        dumpj(loadj(tpl), kb_dir / "templates.json")


def export_taskC(set_root: Path, model: str = "gpt4", template_name: str = "template-1"):
    kb = set_root.name
    c_pairs = loadj(set_root / "TaskC" / TASKC_SUBDIR / "pairs.json")  # [{"head": <Occ>, "tail": <Occ>, "label": <rel>}]
    t = time.strftime("%Y-%m-%d %H:%M:%S")

    res_dir = set_root.parent / "TaskC" / "results" / kb / model
    os.makedirs(res_dir, exist_ok=True)
    dumpj(
        {
            "kb_name": kb,
            "model": model,
            "template": template_name,
            "time": t,
            "relations": c_pairs,
            "metric": "n/a",
        },
        res_dir / f"output-{model}-{template_name}-{t}.json",
    )
    relset = sorted({r["label"] for r in c_pairs})
    dumpj(
        {
            "kb_name": kb,
            "model": model,
            "template": template_name,
            "time": t,
            "counts": {"relations": len(c_pairs)},
            "labels": relset,
        },
        res_dir / f"report-{model}-{template_name}-{t}.json",
    )

    kb_dir = set_root.parent / "TaskC" / kb
    os.makedirs(kb_dir, exist_ok=True)
    lm = set_root / "TaskC" / TASKC_SUBDIR / "label_mapper.json"
    if lm.exists():
        dumpj(loadj(lm), kb_dir / "label_mapper.json")
    tpl = set_root / "TaskC" / TASKC_SUBDIR / "templates.json"
    if tpl.exists():
        dumpj(loadj(tpl), kb_dir / "templates.json")


def main():
    args_dict = {
        "datasets_root": "Occupations_Skills_Mapping",
        "sets": ["JobSkillsSet"],
        "model": "gpt-4o-mini",
        "template": "template-1",
    }

    args = SimpleNamespace(**args_dict)

    root = Path(args.datasets_root)
    for s in args.sets:
        set_root = root / s
        required_legacy = [
            set_root / "TaskA" / TASKA_SUBDIR / "data.json",
            set_root / "TaskB" / TASKB_SUBDIR / "pairs.json",
            set_root / "TaskC" / TASKC_SUBDIR / "pairs.json",
        ]
        required_new = [
            set_root / "TaskA" / TASKA_SUBDIR / "occ2skills.json",
            set_root / "TaskB" / TASKB_SUBDIR / "pairs.json",
            set_root / "TaskC" / TASKC_SUBDIR / "pairs.json",
        ]
        if not (required_new[0].exists() or required_legacy[0].exists()):
            raise FileNotFoundError(
                f"Missing TaskA inputs: either {required_new[0]} or {required_legacy[0]}"
            )
        missing_b_c = [str(p) for p in required_new[1:] if not p.exists()]
        if missing_b_c:
            raise FileNotFoundError("Missing inputs: " + " | ".join(missing_b_c))

        export_taskA(set_root)
        export_taskB(set_root, model=args.model, template_name=args.template)
        export_taskC(set_root, model=args.model, template_name=args.template)
    print("done.")


if __name__ == "__main__":
    main()
