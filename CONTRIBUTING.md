# Contribution Guidelines

## Welcome
We welcome contributions to BreedAI. To keep the framework fair, reproducible, and consistent, please follow these guidelines.

## How to Contribute
1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes with a clear message
4. Push and open a Pull Request

## What We Accept
- Bug fixes, documentation improvements, and new tests
- Model additions that follow the standardized preprocessing pipeline
- Performance improvements that preserve reproducibility

## Required Standards
- Reproducibility first: keep seeds, splits, and preprocessing identical to the framework standard
- No breaking changes without discussion
- Clear documentation for any new behavior or parameter
- Respect data integrity (do not modify benchmark data or public datasets)

## Code Style
- Follow existing formatting and file organization
- Keep functions small and reusable
- Add docstrings or comments when logic is non-trivial

## Tests and Reports
- Run the pipeline if you change core logic
- Update report notebooks if results or outputs change

## Pull Request Checklist
- [ ] Changes are reproducible
- [ ] Documentation updated (README / notebooks if needed)
- [ ] No breaking changes without prior discussion
- [ ] All tests/pipeline runs pass

## Questions
For major changes (new models, pipeline changes, etc.), open an issue first.