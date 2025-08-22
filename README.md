# GPT-OSS-20B Red Team Framework

Competition submission for GPT-OSS-20B Red Team Hackathon.

## Quick Start

### 1. Install Framework
```bash
git clone https://github.com/pravan1/gpt-oss-20b-redteam
cd gpt-oss-20b-redteam
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
python -m gpt_oss_redteam.cli init
```

### 2. Install vLLM for GPT-OSS-20B
**Important:** GPT-OSS-20B is an open-weights model that requires a local server. Install vLLM:

```bash
pip install uv
# Windows PowerShell
uv pip install --pre vllm==0.10.1+gptoss `
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ `
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 `
  --index-strategy unsafe-best-match

# macOS/Linux
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

### 3. Start Local vLLM Server
```bash
vllm serve openai/gpt-oss-20b --port 8000
```
Leave this terminal running - it serves the model locally.

### 4. Configure Environment (PowerShell)
```powershell
$env:OPENAI_API_KEY = "local-anything"  # Any string works for local server
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"  # Point to local vLLM
```

### 5. Run Tests
```powershell
# Option A: Convenience script
python .\win_hackathon.py

# Option B: PowerShell helper
.\run_redteam.ps1

# Option C: CLI directly (specify categories and output)
python -m gpt_oss_redteam.cli run `
  --backend openai_compatible `
  --categories reward_hacking,deception,eval_awareness `
  --seeds 0,1,2 `
  --out runs\gptoss-local
```

### 6. Generate Reports & Kaggle Findings
```bash
# Summarize a run
python -m gpt_oss_redteam.cli report --run-dir runs\gptoss-local

# Create Kaggle findings templates
python -m gpt_oss_redteam.cli findings --issue-id ISSUE_1 --out findings_templates\finding_1.json
python -m gpt_oss_redteam.cli findings --issue-id ISSUE_2 --out findings_templates\finding_2.json
# ... up to 5 findings
```

## Submission Structure

```
competition_submission/     # Main Kaggle submission
├── finding_1.json         # Up to 5 findings in Kaggle format
├── finding_2.json
├── finding_3.json
├── finding_4.json
├── finding_5.json
├── reproduction.ipynb     # Notebook to reproduce findings
└── writeup.md            # Competition writeup

hackathon_submissions/     # Backup submission directory
notebooks/                 # Reproduction notebooks
config/                   # Configuration files
src/gpt_oss_redteam/      # Core framework code
```

## Probe Categories

- **reward_hacking** - Gaming reward/scoring systems
- **deception** - Misleading or false outputs
- **hidden_motivation** - Concealed objectives
- **eval_awareness** - Evaluation/benchmark detection
- **cot_issues** - Chain-of-thought reasoning problems
- **data_exfiltration** - Information leakage
- **sabotage** - Destructive behaviors
- **sandbagging** - Deliberate underperformance
- **tool_misuse** - Improper tool usage

## Configuration

Edit `config/settings.yaml`:
- Set `backend.type` to "mock" or "openai_compatible"
- Adjust `sampling` parameters (temperature, max_tokens)
- Configure `seeds` for reproducibility

Edit `config/providers.yaml`:
- Configure API endpoints and keys
- Uses environment variables: `${OPENAI_API_KEY}`, `${OPENAI_BASE_URL}`

## CLI Commands

```bash
# List all probes
python -m gpt_oss_redteam.cli list

# Run specific categories
python -m gpt_oss_redteam.cli run --categories [categories] --seeds [seeds] --out [dir]

# Generate report
python -m gpt_oss_redteam.cli report --run-dir [dir]

# Create findings
python -m gpt_oss_redteam.cli findings --issue-id [ID] --out [file]
```

## License

MIT License - See LICENSE file for details.