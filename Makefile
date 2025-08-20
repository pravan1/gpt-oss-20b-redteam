.PHONY: setup format lint test run-demo clean

setup:
	pip install -e ".[dev]"
	cp config/settings.example.yaml config/settings.yaml 2>/dev/null || true
	cp config/providers.example.yaml config/providers.yaml 2>/dev/null || true
	redteam init

format:
	black src tests
	ruff check --fix src tests

lint:
	ruff check src tests
	mypy src

test:
	pytest tests -v --cov=src/gpt_oss_redteam --cov-report=term-missing

run-demo:
	redteam run --categories reward_hacking,deception --backend mock --out runs/demo
	redteam report --run-dir runs/demo

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage .mypy_cache
	rm -rf runs/* 2>/dev/null || true