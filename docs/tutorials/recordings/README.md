# STUNIR Tutorial Recordings

This directory contains ASCII cinema recordings of the tutorial sessions.

## Available Recordings

| File | Tutorial | Duration |
|------|----------|----------|
| `01_getting_started.cast` | Getting Started | ~10 min |
| `02_basic_workflow.cast` | Basic Workflow | ~15 min |
| `03_advanced_features.cast` | Advanced Features | ~20 min |
| `04_troubleshooting.cast` | Troubleshooting | ~15 min |

## Playback

### Local Playback

```bash
# Install asciinema
pip install asciinema

# Play a recording
asciinema play 01_getting_started.cast

# Play at 2x speed
asciinema play -s 2 01_getting_started.cast
```

### Web Playback

Recordings are also available at:
- [asciinema.org/~stunir](https://asciinema.org/~stunir) (placeholder)

## Recording New Sessions

```bash
# Start recording
asciinema rec -t "STUNIR Tutorial" output.cast

# End recording
exit  # or Ctrl+D

# Upload (optional)
asciinema upload output.cast
```

## Notes

- Recordings are terminal-only (no GUI)
- Best viewed in a terminal with 80+ columns
- Pause with Space, quit with q during playback
