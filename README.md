# 1. การสร้างและติดตั้ง environment

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

# 2 การใช้ generate_with_openai.py

เข้า floder `Occupation_skill_mapping_src/`
การ run code

```bash
python generate_with_openai.py \
  --occupations_csv occ_update.csv \
  --skills_csv skill_update.csv \
  --out_root my_output_dir \
  --model gpt-4o-mini \
  --taxonomy_mode auto \
  --relation_labels related_to collaborates_with depends_on

```

Arguments

- --occupations_csv (required) : ไฟล์ CSV รายการอาชีพ
- --skills_csv (required) : ไฟล์ CSV รายการสกิล
- --out_root (default=Occupations_Skills_Mapping) : โฟลเดอร์ผลลัพธ์
- --model (default=gpt-4o-mini) : เลือก LLM
- --taxonomy_seeds : JSON string [[child, parent], ...] หรือ "auto"
- --taxonomy_mode : "manual" ใช้ seeds ที่ให้มา หรือ "auto" ให้ LLM สร้างเอง
- --relation_labels (default=related_to collaborates_with depends_on) : ชุด label ของ relation
  ผลลัพธ์ที่สร้างจะอยู่ใน out_root/TaskA, out_root/TaskB, out_root/TaskC.

# 3 การใช้ export_ref_format.py

สคริปต์นี้ export ข้อมูลที่ได้ให้อยู่ใน format มาตรฐาน (เช่น \*\_entities.json, stats.json, templates.json)

```bash
python export_ref_format.py \
  --datasets_root my_output_dir \
  --sets JobSkillsSet \
  --model gpt-4o-mini \
  --template template-1
```

Arguments

- --datasets_root (default=datasets) : root directory ของชุดข้อมูล
- --sets (required) : รายชื่อ dataset (เช่น JobSkillsSet)
- --model (default=gpt-4o-mini) : model สำหรับ metadata
- --template (default=template-1) : template name

# 4 การใช้ ontology_mapping.py

สคริปต์นี้รวมผล Task A, B, C ให้เป็น ontology object และ export ออกเป็น Turtle (TTL)

```bash
from pathlib import Path
from ontology_mapping import build_ontology, write_ttl

onto = build_ontology(Path("my_output_dir"), "JobSkillsSet")
write_ttl(onto, Path("ontology.ttl"))
```
