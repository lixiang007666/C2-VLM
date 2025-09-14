# Contributing to C2-VLM

We welcome contributions to C2-VLM! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/C2-VLM.git
   cd C2-VLM
   ```
3. **Set up the development environment**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Include docstrings for all public functions and classes
- Keep line length under 100 characters

### Code Formatting

We use the following tools for code formatting and linting:

```bash
# Format code
black src/ scripts/ examples/ tests/

# Sort imports
isort src/ scripts/ examples/ tests/

# Lint code
flake8 src/ scripts/ examples/ tests/
```

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting a PR
- Aim for good test coverage

Run tests with:
```bash
pytest tests/ -v
```

### Documentation

- Update documentation for any new features
- Include examples for new functionality
- Update the README if necessary

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and platform
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

For feature requests, please provide:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach

### Code Contributions

#### Pull Request Process

1. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Add tests** for your changes

4. **Run the test suite** to ensure everything works:
   ```bash
   pytest tests/
   ```

5. **Format your code**:
   ```bash
   black src/ scripts/ examples/ tests/
   isort src/ scripts/ examples/ tests/
   ```

6. **Commit your changes** with clear commit messages:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

#### Pull Request Guidelines

- **Clear title and description**: Explain what your PR does and why
- **Reference issues**: Link to any relevant issues
- **Small, focused changes**: Keep PRs manageable in size
- **Update documentation**: Include any necessary documentation updates
- **Add tests**: Ensure your changes are tested

### Commit Message Guidelines

Use clear and descriptive commit messages:

```
type(scope): brief description

Longer description if necessary

- List any breaking changes
- Reference any issues closed
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add attention visualization functionality
fix(training): resolve memory leak in data loader
docs(readme): update installation instructions
```

## Development Setup

### Environment Setup

1. **Python version**: Use Python 3.8 or higher
2. **Virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Development Dependencies

For development, install additional dependencies:
```bash
pip install -r requirements.txt
pip install pytest black flake8 isort
```

### Pre-commit Hooks

Consider setting up pre-commit hooks to automatically format code:

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
C2-VLM/
├── src/                    # Main source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and processing
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics and scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── examples/              # Usage examples
├── tests/                 # Test files
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## Community Guidelines

- **Be respectful**: Treat all community members with respect
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Remember that everyone is volunteering their time
- **Ask for help**: Don't hesitate to ask questions if you're stuck

## Questions and Support

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Email**: Contact the maintainers directly for sensitive issues

## License

By contributing to C2-VLM, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to C2-VLM! 🚀