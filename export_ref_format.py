# export_ref_format.py
import os
import json
import argparse
import time
from pathlib import Path


def loadj(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dumpj(obj, p):
    os.makedirs(Path(p).parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_taskA(set_root: Path):
    # from LLMs4OL_Generator_OpenAI
    a = loadj(set_root / "TaskA" / "data.json")  # [{"text":..., "label":...}]
    kb = set_root.name
    out_dir = set_root.parent.parent / "TaskA" / kb
    os.makedirs(out_dir, exist_ok=True)

    # entities file (แบบโปรเจคใช้ *_entities.json)
    ents = [{"entity": r["text"], "type": r["label"]} for r in a]
    dumpj(ents, out_dir / f"{kb.lower()}_entities.json")

    # stats.json
    per_type = {}
    for r in a:
        per_type[r["label"]] = per_type.get(r["label"], 0) + 1
    stats = {
        "kb_name": kb,
        "n_entities": len(a),
        "n_types": len(per_type),
        "per_type": per_type,
    }
    dumpj(stats, out_dir / "stats.json")

    # label mapper 그대로 복사
    lm_src = set_root / "TaskA" / "label_mapper.json"
    if lm_src.exists():
        dumpj(loadj(lm_src), out_dir / "label_mapper.json")
    # templates 그대로 복사
    tpl_src = set_root / "TaskA" / "templates.json"
    if tpl_src.exists():
        dumpj(loadj(tpl_src), out_dir / "templates.json")


def export_taskB(set_root: Path, model: str = "gpt4", template_name: str = "template-1"):
    kb = set_root.name
    b_pairs = loadj(set_root / "TaskB" / "pairs.json")  # [{"child":T1,"parent":T2}]
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    res_dir = set_root.parent.parent / "TaskB" / "results" / kb / model
    os.makedirs(res_dir, exist_ok=True)
    # output-* ตามสไตล์โปรเจค
    output = {
        "kb_name": kb,
        "model": model,
        "template": template_name,
        "time": t,
        "pairs": b_pairs,           # confirmed is-a pairs
        "metric": "n/a"
    }
    dumpj(output, res_dir / f"output-{model}-{template_name}-{t}.json")
    # report-*
    report = {
        "kb_name": kb,
        "model": model,
        "template": template_name,
        "time": t,
        "counts": {"pairs": len(b_pairs)}
    }
    dumpj(report, res_dir / f"report-{model}-{template_name}-{t}.json")
    # copy label_mapper / template
    lm = set_root / "TaskB" / "label_mapper.json"
    if lm.exists():
        dumpj(loadj(lm), set_root.parent.parent / "TaskB" / kb / "label_mapper.json")
    tpl = set_root / "TaskB" / "template.json"
    if tpl.exists():
        dumpj(loadj(tpl), set_root.parent.parent / "TaskB" / kb / "templates.json")


def export_taskC(set_root: Path, model: str = "gpt4", template_name: str = "template-1"):
    kb = set_root.name
    c_pairs = loadj(set_root / "TaskC" / "pairs.json")  # [{"head":T1,"tail":T2,"label":rel}]
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    res_dir = set_root.parent.parent / "TaskC" / "results" / kb / model
    os.makedirs(res_dir, exist_ok=True)
    # output-*
    output = {
        "kb_name": kb,
        "model": model,
        "template": template_name,
        "time": t,
        "relations": c_pairs,       # typed non-taxonomic relations
        "metric": "n/a"
    }
    dumpj(output, res_dir / f"output-{model}-{template_name}-{t}.json")
    # report-*
    relset = sorted({r["label"] for r in c_pairs})
    report = {
        "kb_name": kb,
        "model": model,
        "template": template_name,
        "time": t,
        "counts": {"relations": len(c_pairs)},
        "labels": relset
    }
    dumpj(report, res_dir / f"report-{model}-{template_name}-{t}.json")
    # copy label_mapper / templates
    lm = set_root / "TaskC" / "label_mapper.json"
    if lm.exists():
        dumpj(loadj(lm), set_root.parent.parent / "TaskC" / kb / "label_mapper.json")
    tpl = set_root / "TaskC" / "templates.json"
    if tpl.exists():
        dumpj(loadj(tpl), set_root.parent.parent / "TaskC" / kb / "templates.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", default="datasets")
    ap.add_argument("--sets", nargs="+", required=True)  # e.g. SubjectsSet JobSkillsSet
    ap.add_argument("--model", default="gpt4")
    ap.add_argument("--template", default="template-1")
    args = ap.parse_args()

    root = Path(args.datasets_root)
    for s in args.sets:
        set_root = root / s
        export_taskA(set_root)
        export_taskB(set_root, model=args.model, template_name=args.template)
        export_taskC(set_root, model=args.model, template_name=args.template)
    print("done.")


if __name__ == "__main__":
    main()
