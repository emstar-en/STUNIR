# Migration Guides

This directory contains migration guides for upgrading between STUNIR versions.

## Available Guides

| From | To | Guide | Difficulty |
|------|-----|-------|------------|
| 1.x | 2.x | [v1_to_v2.md](v1_to_v2.md) | Moderate |
| 2.x | 3.x | [v2_to_v3.md](v2_to_v3.md) | TBD |

## Quick Migration Checklist

### Upgrading to v2.0

- [ ] Update Python to 3.8+
- [ ] Regenerate all manifests
- [ ] Update spec files for new schema
- [ ] Run verification to ensure compatibility

## Migration Process

1. **Backup** your current installation
2. **Read** the migration guide for your version
3. **Test** in a development environment first
4. **Update** code and configurations
5. **Verify** all outputs match expected results

## Getting Help

If you encounter issues during migration:

- Check the [FAQ](../FAQ.md)
- Search [GitHub Issues](https://github.com/stunir/stunir/issues)
- Open a new issue with the `migration` label

## Contributing

Help improve migration guides:

1. Report unclear instructions
2. Suggest additional migration paths
3. Share your migration experience
