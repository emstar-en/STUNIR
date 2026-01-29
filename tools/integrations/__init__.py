"""STUNIR External Tool Integrations.

Provides integrations with:
- Git (commit tracking, branch detection, metadata)
- CI/CD (GitHub Actions, GitLab CI, environment detection)
- Package managers (Cargo, pip/poetry, Cabal/Stack)

Usage:
    from tools.integrations import GitIntegration, CIIntegration
    from tools.integrations import get_git_info, detect_ci_environment
"""

from .git_integration import (
    GitIntegration,
    get_git_info,
    get_commit_hash,
    get_branch_name,
    is_dirty,
    get_git_metadata,
)
from .ci_integration import (
    CIIntegration,
    detect_ci_environment,
    is_ci,
    get_ci_metadata,
    generate_github_actions_workflow,
    generate_gitlab_ci_config,
)
from .package_integration import (
    CargoIntegration,
    PipIntegration,
    CabalIntegration,
    detect_package_manager,
)

__all__ = [
    # Git integration
    'GitIntegration',
    'get_git_info',
    'get_commit_hash',
    'get_branch_name',
    'is_dirty',
    'get_git_metadata',
    # CI integration
    'CIIntegration',
    'detect_ci_environment',
    'is_ci',
    'get_ci_metadata',
    'generate_github_actions_workflow',
    'generate_gitlab_ci_config',
    # Package managers
    'CargoIntegration',
    'PipIntegration',
    'CabalIntegration',
    'detect_package_manager',
]
