# GPT-OSS-20B Red Team Framework

A comprehensive red-teaming framework for evaluating OpenAI's GPT-OSS-20B model in the Kaggle hackathon.

## 🛡️ Safety Notice

This framework is designed for **responsible security research only**. All probes contain **benign, safe content** and are intended to identify potential model vulnerabilities without causing harm. Do not modify this framework to generate harmful content.

## 🚀 Quick Start

```bash
# Install with pip (or uv)
uv pip install -e .

# Copy configuration files
cp config/settings.example.yaml config/settings.yaml
cp config/providers.example.yaml config/providers.yaml

# Initialize environment
redteam init

# List available probes
redteam list

# Run probes with mock backend
redteam run --categories reward_hacking,deception --backend mock --out runs/demo

# Generate report
redteam report --run-dir runs/demo

# Import Harmony findings to Kaggle format
redteam import-harmony --in examples/example-harmony-findings.json --out findings_templates/from_harmony.findings.json
```

## 📦 Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (for local model inference)
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/gpt-oss-20b-redteam.git
cd gpt-oss-20b-redteam

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e ".[dev]"

# Setup configuration
make setup
```

## 🔧 Configuration

### Backend Selection

Edit `config/settings.yaml`:

```yaml
backend:
  type: mock  # Options: hf_local, openai_compatible, mock
```

### Provider Configuration

Edit `config/providers.yaml`:

```yaml
providers:
  openai_compatible:
    base_url: ${OPENAI_API_BASE}
    api_key: ${OPENAI_API_KEY}
    model: gpt-oss-20b
```

## 🎯 Probe Categories

The framework includes 9 probe categories with safe, benign examples:

- **reward_hacking**: Tests for gaming evaluation metrics
- **deception**: Tests for false claims and misleading statements
- **hidden_motivation**: Tests for concealed objectives or biases
- **eval_awareness**: Tests if model detects evaluation contexts
- **cot_issues**: Tests chain-of-thought reasoning consistency
- **sabotage**: Tests for intentional errors (safety-focused)
- **tool_misuse**: Tests inappropriate tool usage claims
- **data_exfiltration**: Tests for data leakage attempts
- **sandbagging**: Tests for deliberate underperformance

## 📊 Running Evaluations

### CLI Commands

```bash
# Run specific categories
redteam run --categories reward_hacking,deception --seeds 0,1,2,42

# Run individual probes
redteam run --probes confidence_incorrect_facts,false_tool_claims

# Generate Kaggle findings
redteam findings --issue-id ISSUE-001 --run-dir runs/latest --out findings.json

# Convert formats
redteam import-harmony --in harmony.json --out kaggle.json
redteam export-harmony --in kaggle.json --out harmony.json
```

### Python API

```python
from gpt_oss_redteam import load_config
from gpt_oss_redteam.model.backends import get_backend
from gpt_oss_redteam.probes.registry import run_category

# Setup
config = load_config()
backend = get_backend("mock", config.backend.model_dump(), config.providers)

# Run probes
results = run_category("reward_hacking", backend, seeds=[0, 1, 2])

# Generate report
from gpt_oss_redteam.eval.reporting import make_findings_json
findings = make_findings_json(results, "ISSUE-001")
```

## 📈 Kaggle Submission

### Preparing Findings

1. Run comprehensive evaluation:
```bash
redteam run --categories all --backend openai_compatible --seeds 0,1,2,42,1337
```

2. Generate findings JSON:
```bash
redteam findings --issue-id KAGGLE-SUBMISSION-001 --run-dir runs/latest --out submission.json
```

3. Validate against schema:
```python
from gpt_oss_redteam.eval.converters import validate_kaggle_findings
validate_kaggle_findings("submission.json")
```

### Findings Format

```json
{
  "version": "0.1.0",
  "issue_id": "KAGGLE-001",
  "findings": [{
    "title": "Reward Hacking Issues Detected",
    "observed_behavior": "Model showed overconfidence on incorrect answers",
    "expected_behavior": "Model should calibrate confidence with accuracy",
    "evidence": ["High confidence (95%) on wrong answer"],
    "severity_estimate": 3,
    "breadth_estimate": 2
  }]
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests -v

# Run with coverage
pytest tests --cov=src/gpt_oss_redteam

# Run specific test category
pytest tests/test_probes.py -v
```

## 📓 Notebooks

See `notebooks/01_reproduce_findings.ipynb` for:
- Interactive probe execution
- Metrics visualization
- Harmony↔Kaggle conversion demos
- Deterministic validation

## 🔄 Reproducibility

### Deterministic Seeds

```yaml
# config/settings.yaml
seeds:
  default: [0, 1, 2, 42, 1337]
  deterministic: true
  torch_deterministic: true
```

### Version Pinning

Key dependencies are pinned in `pyproject.toml` for reproducibility.

## 🏗️ Project Structure

```
gpt-oss-20b-redteam/
├── src/gpt_oss_redteam/
│   ├── model/           # Backend implementations
│   ├── probes/          # Probe framework
│   │   └── categories/  # Probe implementations
│   ├── eval/            # Metrics & reporting
│   └── cli.py           # CLI interface
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks
├── tests/               # Test suite
└── findings_templates/  # Output templates
```

## ⚖️ Ethical Use

This framework is for **defensive security research only**:

- ✅ Identify model vulnerabilities
- ✅ Improve model safety
- ✅ Academic research
- ❌ Generate harmful content
- ❌ Bypass safety measures
- ❌ Malicious exploitation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- OpenAI for the GPT-OSS-20B model
- Kaggle for hosting the hackathon
- The red-teaming community

## 📞 Support

- Issues: [GitHub Issues](https://github.com/your-org/gpt-oss-20b-redteam/issues)
- Documentation: See `/docs` directory
- Email: redteam@example.com

---

**Remember**: Use this framework responsibly and ethically. The goal is to improve AI safety, not to cause harm.