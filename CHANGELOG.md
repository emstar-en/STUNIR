# Changelog

All notable changes to STUNIR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Interactive examples for Python, Rust, and Haskell
- Jupyter notebook tutorials
- Video tutorial scripts with ASCII cinema recordings
- Automated changelog generation with git-cliff
- Comprehensive glossary of terms
- Expanded FAQ with 50+ questions
- Migration guides for version upgrades
- Release notes automation

### Changed
- Improved documentation structure
- Enhanced error messages with actionable suggestions

### Fixed
- Various documentation typos and inconsistencies

## [2.0.0] - 2026-01-28

### Added
- **Core**: Complete STUNIR implementation with deterministic IR generation
- **Targets**: Support for Python, Rust, C89, C99, x86, and ARM targets
- **Manifests**: Comprehensive manifest system for artifact tracking
  - IR manifests
  - Targets manifests
  - Receipts manifests
  - Pipeline manifests
- **Native Tools**: Haskell-native toolchain for manifest generation
- **Provenance**: C provenance header generation with extended macros
- **Verification**: Strict verification mode with hash validation
- **Documentation**: Complete documentation suite
  - User guide
  - API reference
  - Security policy
  - Contributing guidelines
- **Testing**: Comprehensive test suite with 100% pass rate
- **CI/CD**: GitHub Actions workflows for testing and releases

### Changed
- Restructured project for better modularity
- Improved canonical JSON output for RFC 8785 compliance
- Enhanced error handling with detailed messages

### Security
- Added security policy and vulnerability reporting
- Implemented secure hash verification

## [1.0.0] - 2025-06-15

### Added
- Initial STUNIR release
- Basic spec-to-IR conversion
- Python target generation
- Simple receipt generation
- Basic documentation

### Known Issues
- Limited target support (Python only)
- No manifest verification
- Basic error handling

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | 2026-01-28 | Complete rewrite, multi-target support, manifest system |
| 1.0.0 | 2025-06-15 | Initial release |

## Upgrade Notes

See [Migration Guides](docs/migration/) for detailed upgrade instructions.

## Links

- [GitHub Releases](https://github.com/stunir/stunir/releases)
- [Documentation](https://stunir.dev/docs)
- [Migration Guides](docs/migration/)
