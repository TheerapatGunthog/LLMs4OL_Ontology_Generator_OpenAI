# การสร้างและติดตั้ง environment

```bash
python -m venv <name_your_env>
source <name_your_env>/bin/activate   # บน Linux/Mac
venv\Scripts\activate      # บน Windows

pip install -r requirements.txt

```

ตั้ง .env ที่ root project

```
OPENAI_API_KEY="your_api_key"
```

---

# Usage

cd เข้า floder `Occupation_skill_mapping_src/`

```bash
cd Occupation_skill_mapping_src/
```

## 1 Generate Tasks (A, B, C)

cd เปิด `generate_with_openai.py` แล้วแก้ตัวแปร `args = { ... }` หรือ class `_Args` ให้ตรงกับไฟล์ CSV

```python
args = {
    "occupations_csv": "occ_update.csv",
    "skills_csv": "skill_update.csv",
    "out_root": "Occupations_Skills_Mapping",
    "model": "gpt-4o-mini",
    "taxonomy_seeds": "auto",     # หรือ JSON string [["child","parent"],...]
    "taxonomy_mode": "auto",      # "manual" หรือ "auto"
    "relation_labels": ["related_to", "collaborates_with", "depends_on"],
}

```

จากนั้นรัน

```bash
python generate_with_openai.py
```

ผลลัพธ์จะอยู่ในโฟลเดอร์ `Occupations_Skills_Mapping/TaskA`, `TaskB`, `TaskC`.

## 2 Export Reference Format

เปิด `export_ref_format` แล้วแก้ค่าตัวแปร

```python
args = {
    "datasets_root": "Occupations_Skills_Mapping",
    "sets": ["JobSkillsSet"],
    "model": "gpt-4o-mini",
    "template": "template-1",
}
```

```bash
python export_ref_format.py
```

จะสร้างไฟล์ `entities.json`, `stats.json`, `templates.json` ใน TaskA/TaskB/TaskC.

## Build Ontology + Export TTL

เปิด `ontology_mapping.py` แล้วแก้ค่าตัวแปร

```python
args = {
    "datasets_root": "Occupations_Skills_Mapping",
    "sets": ["JobSkillsSet"],
    "out_name": "ontology.json",
    "out_ttl": "ontology.ttl",
    "base_iri": "http://example.org/occ-skills#",
}
```

```bash
python ontology_mapping.py
```

จะได้

- `ontology.json`
- `ontology.ttl` (สำหรับโหลดเข้า Protégé หรือระบบ ontology อื่น ๆ)
