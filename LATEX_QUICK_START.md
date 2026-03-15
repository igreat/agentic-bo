# Quick Start: Generate LaTeX Reports from BO Results

## ✅ What You Now Have

1. **LATEX_REPORT_GUIDE.md** — Complete step-by-step guide (comprehensive)
2. **scripts/generate_latex_report.py** — Automated report generation script
3. **runs/vivid-heron-3397/report.tex** — Example LaTeX report (already generated!)

---

## 🚀 TL;DR - Use It Now

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

### Step 2: Generate LaTeX report

```bash
uv run python scripts/generate_latex_report.py <RUN_ID>
```

### Step 3: Compile to PDF

```bash
cd runs/<RUN_ID>
xelatex report.tex
```

**Output:** `report.pdf` ← Ready for sharing!

---

## 📋 Report Contents

Each generated LaTeX report includes:

✅ **Title Page** — Professional cover with run metadata
✅ **Executive Summary** — Key findings at a glance  
✅ **Methodology** — BO engine, objective, dimensions explained
✅ **Results** — Convergence metrics, best values, improvement %
✅ **Top Candidates** — Table of best suggestions for validation
✅ **Oracle Analysis** — Model fidelity (CV RMSE), feature importance
✅ **Discussion** — Findings, recommendations, limitations
✅ **Appendices** — Full iteration logs, run metadata

---

## 🎨 Customize the Report

### Option A: Modify the Default Template

Edit the template in `scripts/generate_latex_report.py` (lines ~290-450) to add:
- Your institution logo
- Custom color scheme
- Additional sections
- Branded headers/footers

### Option B: Use a Custom Template

Create your own LaTeX template (e.g., `my_template.tex`):

```bash
uv run python scripts/generate_latex_report.py <RUN_ID> --template my_template.tex
```

The script will fill in placeholders like:
- `INSERT_BEST_Y` → best observed value
- `INSERT_IMPROVEMENT_PERCENT` → % improvement
- `INSERT_RUN_ID` → run ID
- (Full list in LATEX_REPORT_GUIDE.md)

---

## 🔧 Add to Your Workflow

### Option 1: Manual (What You're Doing Now)

```bash
# After run-proxy completes
uv run python scripts/generate_latex_report.py <RUN_ID>
xelatex runs/<RUN_ID>/report.tex
```

### Option 2: Automatic (Make It Seamless)

Modify `bo_workflow/cli.py` to add a new command:

```python
# In bo_workflow/cli.py
subparsers.add_parser("report-latex", ...).set_defaults(
    handler=lambda args: subprocess.run([
        "python", "scripts/generate_latex_report.py", args.run_id
    ])
)
```

Then use:
```bash
uv run python -m bo_workflow.cli report-latex --run-id <RUN_ID>
```

---

## 📊 Example: Full End-to-End Workflow

```bash
# 1. Initialize BO
uv run python -m bo_workflow.cli init \
  --dataset data/HER_virtual_data.csv \
  --target Target --objective max --seed 42 > init_result.json

# Extract RUN_ID from JSON output or use:
export RUN_ID=$(cat init_result.json | grep -o '"run_id":"[^"]*' | cut -d'"' -f4)

# 2. Build oracle
uv run python -m bo_workflow.cli build-oracle --run-id $RUN_ID

# 3. Run full BO
uv run python -m bo_workflow.cli run-proxy --run-id $RUN_ID --iterations 20

# 4. Generate both JSON and LaTeX reports
uv run python -m bo_workflow.cli report --run-id $RUN_ID
uv run python scripts/generate_latex_report.py $RUN_ID

# 5. Compile LaTeX to PDF
cd runs/$RUN_ID
xelatex report.tex
cd ../..

# 6. View results
open runs/$RUN_ID/report.pdf          # macOS
xdg-open runs/$RUN_ID/report.pdf      # Linux
start runs/$RUN_ID/report.pdf         # Windows
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pandas` | Use `uv run python` instead of just `python` |
| LaTeX compilation fails | Ensure you have XeLaTeX installed: `xelatex --version` |
| Missing `scientific_report.sty` | Download from [claude-scientific-skills repo](https://github.com/K-Dense-AI/claude-scientific-skills/blob/main/scientific-skills/scientific-writing/assets/scientific_report.sty) and place in `runs/<RUN_ID>/` |
| Placeholders not filled | Check column names in suggestions/observations JSONLs match the script |
| PDF is blank | Run: `xelatex report.tex` twice or use `latexmk -xelatex report.tex` |

---

## 📚 Full Documentation

For complete details, see **LATEX_REPORT_GUIDE.md** in the workspace root:
- Step-by-step installation
- Advanced customization
- Template modification
- Integration patterns
- Real-world examples

---

## 💡 Pro Tips

1. **Keep run metadata** — Store the `state.json` and `oracle_meta.json` for reproducibility
2. **Batch reports** — Generate reports for multiple runs:
   ```bash
   for run in runs/*/; do
     uv run python scripts/generate_latex_report.py $(basename $run)
   done
   ```
3. **Version control** — Git-track your `.tex` files for collaboration
4. **Archive PDFs** — Keep compiled PDFs alongside `.tex` for immutability
5. **Custom styles** — Copy `scientific_report.sty` locally and modify for branding

---

## 🎓 Learning Resources

- **LaTeX Wikibook:** https://en.wikibooks.org/wiki/LaTeX
- **Scientific Writing Skill Docs:** See `LATEX_REPORT_GUIDE.md` references section
- **claude-scientific-skills:** https://github.com/K-Dense-AI/claude-scientific-skills
