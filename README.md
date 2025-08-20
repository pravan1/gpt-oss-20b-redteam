# GPT-OSS-20B Red Team Framework

Competition submission for GPT-OSS-20B Red Team Hackathon.

## Quick Start

### 1. Install
```bash
pip install -e .
python -m gpt_oss_redteam.cli init
```

### 2. Set API Credentials (PowerShell)
```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
```

### 3. Run Tests
```powershell
# Use the convenience script
.\win_hackathon.py

# Or use PowerShell runner
.\run_redteam.ps1

# Or CLI directly
python -m gpt_oss_redteam.cli run --backend openai_compatible --categories reward_hacking,deception --seeds 0,1,2 --out runs/realapi
```

### 4. Generate Kaggle Findings
```bash
python -m gpt_oss_redteam.cli findings --issue-id ISSUE_1 --out findings_templates/finding_1.json
python -m gpt_oss_redteam.cli findings --issue-id ISSUE_2 --out findings_templates/finding_2.json
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