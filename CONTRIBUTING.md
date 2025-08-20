# Contributing to Translation Evaluation Framework

We welcome contributions to the Translation Evaluation Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Relevant log output or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear, descriptive title
- Detailed description of the proposed enhancement
- Use cases and benefits
- Any relevant examples or mockups

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as necessary
5. Ensure all tests pass
6. Update documentation if needed
7. Commit your changes with clear, descriptive messages
8. Push to your branch
9. Create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/translation-evaluation-framework.git
cd translation-evaluation-framework

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/

# Check types
mypy src/

# Lint code
flake8 src/
```

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Maintain test coverage above 80%
- Use meaningful variable and function names
- Keep functions focused and concise

## Testing

- Write unit tests for all new functionality
- Update existing tests when modifying code
- Ensure all tests pass before submitting PRs
- Include integration tests for major features

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all new functions and classes
- Update type hints and examples
- Consider adding usage examples

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
