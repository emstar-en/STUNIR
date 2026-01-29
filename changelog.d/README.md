# Changelog Fragments

This directory contains changelog fragments for the next release.

## How to Use

When making changes, create a file in this directory:

```
<issue-number>.<type>.md
```

### Types

| Type | Description |
|------|-------------|
| `feature` | New features |
| `bugfix` | Bug fixes |
| `breaking` | Breaking changes |
| `deprecation` | Deprecated features |
| `doc` | Documentation changes |
| `misc` | Other changes |

### Example

For issue #123 adding a new feature:

```bash
echo "Added support for WebAssembly target" > changelog.d/123.feature.md
```

### Templates

See the `_templates/` directory for fragment templates.

## Generating Changelog

```bash
# Using git-cliff
git cliff --unreleased

# Or using the release script
./scripts/release.sh changelog
```

## Notes

- One fragment per issue/PR
- Keep messages concise but descriptive
- Include migration notes for breaking changes
- Fragments are consumed during release
