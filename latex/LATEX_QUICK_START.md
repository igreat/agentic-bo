# Quick Start: Generate LaTeX Reports from BO Results

### Step 1: Run a complete BO optimization

```bash
# Initialize BO run
uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42

# Build surrogate oracle
uv run python -m bo_workflow.cli build-oracle --run-id <RUN_ID>

# Run full BO loop
uv run python -m bo_workflow.cli run-proxy --run-id <RUN_ID> --iterations 20
```
### Step 2: Generate Written Sections for Report

```bash
uv run python -m bo_workflow.cli report --run-id <RUN_ID>
uv run python -m bo_workflow.report_writer_cli <RUN_ID>
```

### Step 3: Generate LaTeX report

```bash
uv run python latex/generate_latex_report.py <RUN_ID>
uv run python latex/generate_poster.py <RUN_ID>
```

### Step 3: Compile to PDF

```bash
# Run from the project root so LaTeX can find the included style file 
#REMEMBER TO RUN TWICE
xelatex -output-directory="runs/<RUN_ID>" "runs/<RUN_ID>/report.tex"
xelatex -output-directory="runs/<RUN_ID>" "runs/<RUN_ID>/poster.tex"
```

**Output:** `runs/<RUN_ID>/report.pdf`
**Output:** `runs/<RUN_ID>/poster.pdf`

## 🎓 Learning Resources

- **LaTeX Wikibook:** https://en.wikibooks.org/wiki/LaTeX
- **Scientific Writing Skill Docs:** See `LATEX_REPORT_GUIDE.md` references section
- **claude-scientific-skills:** https://github.com/K-Dense-AI/claude-scientific-skills
