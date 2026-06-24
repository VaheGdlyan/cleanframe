# Contributing to Cleanframe

Thank you for your interest in contributing to Cleanframe! We are building the most reliable, hardware-optimized data intelligence framework, and we'd love your help.

## Our Philosophy

1. **Do No Harm:** Rules must never mutate data implicitly. By default, they must act as auditors and raise warnings. Explicit flags (like `action="drop"`) must be required to execute changes.
2. **Strict Typing:** Cleanframe relies on strict type continuity. Every signature must pass strict static analysis. 

## Local Environment Setup

Cleanframe relies on `uv` for blazing-fast dependency management and environment isolation.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vahe-gdlyan/cleanframe.git
   cd cleanframe
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

## Development Workflow

Before submitting a Pull Request, you must ensure your code passes our quality gates.

1. **Run the Test Suite:**
   We have exactly 51 unit tests acting as our regression baseline.
   ```bash
   uv run pytest
   ```

2. **Run Static Analysis:**
   We maintain zero static analysis warnings and 100% strict type coverage.
   ```bash
   uv run ruff check .
   uv run mypy --strict .
   ```

## Pull Request Process

1. Create a feature branch from `main`.
2. Implement your feature or fix. If adding a new rule, ensure it implements `RuleProtocol`.
3. Add corresponding unit tests in `tests/`.
4. Ensure `uv run pytest`, `ruff`, and `mypy` pass perfectly.
5. Submit a Pull Request outlining the problem solved and the architectural impact.

We review all PRs promptly and appreciate your efforts!
