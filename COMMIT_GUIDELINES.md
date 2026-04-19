# Commit Guidelines

This project uses a simple commit message convention to keep the Git history readable, consistent, and easy to understand.

## Format

```text
type(scope): short summary

Types:

feat - new functionality
fix - bug fixes
chore - setup, tooling, or maintenance changes
docs - documentation-only changes
refactor - code improvements without changing behavior
test - adding or updating tests

Guidelines:

Keep the summary short and specific
Use the imperative mood, such as add, fix, or update
Prefer one logical change per commit when practical
Use a scope that identifies the area being changed

Examples:

- chore(init): bootstrap RAG search project structure
- docs(readme): add setup instructions
- feat(search): load movie dataset
- fix(loader): handle missing JSON file
- refactor(parser): simplify JSON handling
