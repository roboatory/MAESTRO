# Personal Coding Standards

## Language & Tooling
- Primary language: Python 3
- Project scaffolding and environment management: uv (init, add, run, sync)
- Linting and formatting: ruff + ruff-format (configured in .pre-commit-config.yaml)

## Naming Conventions
- Spell out variable and function names fully — avoid abbreviations
- Exception: mathematical or statistical implementations may use conventional notation (e.g., `mu`, `sigma`, `x`, `y`, `n`, `p`, `alpha`)
- Prioritize names that express intent over names that describe type or structure

## Commenting Policy
- Only add comments to complex or non-obvious blocks of code
- Do not comment self-explanatory code — clear naming is preferred
- Prefer expressive code over compensatory comments

## Design Priorities (applied in order)
1. **Simplicity** — prefer the straightforward solution; avoid cleverness unless it is the more performant option
2. **Readability** — code is read far more than it is written
3. **Performance** — optimize where it measurably matters

## Python Conventions
- Type hints on all public functions and methods
- When writing function signatures (in your own code), put each argument on its own line
- For function invocations, use whatever call formatting is considered best practice by the Python community (typically, short calls on one line; longer calls may use multiple lines as needed)
- Prefer `pathlib.Path` over `os.path` for all file system operations
- Prefer f-strings over `.format()` or `%` formatting
- Prefer list/dict/set comprehensions over for-loops where the result is readable
- Raise specific exceptions — not bare `Exception`
- No wildcard imports (`from x import *`)

## Git Conventions

### Commit Prefixes
Use one of the following prefixes for all commits:
- `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `style`

### Commit Messages
- Format: `<prefix>: short_description`, where `short_description` is a single short line, all lowercase, no punctuation, in imperative mood
- Add a body only when the why is not obvious; same style rules apply
- Add `closes #<number>` on its own line when closing a GitHub issue
- Do not add co-author trailers to commit messages

### Branch Naming
- Format: `<prefix>/<short_description>`, where `short_description` is a single short line, all lowercase, no punctuation, in imperative mood
- Use underscores to separate words; always lowercase

### Pull Requests
- Create as draft by default
- Title follows commit message format
- Body includes: summary, changes (logical bullets), testing notes, and issue references

## Project Layout
- After project creation, treat `main.py` at the repository root as a **driver only**: parse CLI arguments or load config, perform minimal wiring, and delegate to code under `src/`. It should not hold domain logic, substantial control flow, or architectural structure
- Put all implementation that defines the codebase and its architecture in `src/` (the installable package tree: modules, subpackages, services, domain models, adapters, and similar); new functionality belongs there, not in a growing root-level script
- Place all project tests under the `tests` folder; keep tooling config and documentation at the project root; reserve `main.py` for the entry point and `src/` for everything the application actually does

## Docstrings
- Use single-line docstrings for all functions; rely on type annotations for type information
- Only expand to a multi-line docstring when behavior is non-obvious and cannot be captured in one sentence
- Never use google, numpy, or sphinx/rst docstring sections (no `args:`, `returns:`, `:param`, etc.)

## uv
- `uv init` to scaffold new projects
- `uv add <package>` to add runtime dependencies
- `uv add --dev <package>` to add development dependencies
- `uv run <command>` to execute in the project environment
- `uv sync` to install all dependencies from lockfile
