# OpenAI Dataset Generator for LLMs4OL (Tasks A/B/C)

Generates datasets **with OpenAI** (no heuristics).

## Usage
```bash
conda activate llms4olAPIonly
pip install -r requirements.txt
cp .env.example .env   # put OPENAI_API_KEY=...
python generate_with_openai.py   --subjects_csv /path/to/cleaned_Subject.csv   --skills_csv   /path/to/cleaned_Job_Skill.csv   --out_root     datasets   --model        gpt-4o-mini
```
Outputs:
```
datasets/
  SubjectsSet/TaskA|TaskB|TaskC/...
  JobSkillsSet/TaskA|TaskB|TaskC/...
```
