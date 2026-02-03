"""STUNIR CI/CD Integration.

Provides CI/CD platform integration:
- GitHub Actions helpers
- GitLab CI helpers
- Environment detection
- CI metadata for builds
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class CIPlatform(Enum):
    """Supported CI/CD platforms."""
    GITHUB_ACTIONS = 'github_actions'
    GITLAB_CI = 'gitlab_ci'
    JENKINS = 'jenkins'
    CIRCLECI = 'circleci'
    TRAVIS = 'travis'
    AZURE_PIPELINES = 'azure_pipelines'
    BITBUCKET = 'bitbucket'
    LOCAL = 'local'


@dataclass
class CIMetadata:
    """CI/CD environment metadata."""
    platform: CIPlatform
    is_ci: bool
    build_number: Optional[str]
    build_url: Optional[str]
    branch: Optional[str]
    commit: Optional[str]
    pull_request: Optional[str]
    actor: Optional[str]
    repository: Optional[str]
    event_name: Optional[str]


class CIIntegration:
    """CI/CD platform integration."""
    
    def __init__(self):
        """Initialize CI integration."""
        self._platform: Optional[CIPlatform] = None
        self._metadata: Optional[CIMetadata] = None
    
    def detect_platform(self) -> CIPlatform:
        """Detect the current CI/CD platform.
        
        Returns:
            Detected CI platform
        """
        if self._platform:
            return self._platform
        
        # GitHub Actions
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            self._platform = CIPlatform.GITHUB_ACTIONS
        # GitLab CI
        elif os.environ.get('GITLAB_CI') == 'true':
            self._platform = CIPlatform.GITLAB_CI
        # Jenkins
        elif os.environ.get('JENKINS_URL'):
            self._platform = CIPlatform.JENKINS
        # CircleCI
        elif os.environ.get('CIRCLECI') == 'true':
            self._platform = CIPlatform.CIRCLECI
        # Travis CI
        elif os.environ.get('TRAVIS') == 'true':
            self._platform = CIPlatform.TRAVIS
        # Azure Pipelines
        elif os.environ.get('TF_BUILD') == 'True':
            self._platform = CIPlatform.AZURE_PIPELINES
        # Bitbucket Pipelines
        elif os.environ.get('BITBUCKET_BUILD_NUMBER'):
            self._platform = CIPlatform.BITBUCKET
        else:
            self._platform = CIPlatform.LOCAL
        
        return self._platform
    
    def is_ci(self) -> bool:
        """Check if running in a CI environment."""
        return self.detect_platform() != CIPlatform.LOCAL
    
    def get_metadata(self) -> CIMetadata:
        """Get CI environment metadata.
        
        Returns:
            CIMetadata object with platform-specific information
        """
        if self._metadata:
            return self._metadata
        
        platform = self.detect_platform()
        
        if platform == CIPlatform.GITHUB_ACTIONS:
            self._metadata = self._get_github_metadata()
        elif platform == CIPlatform.GITLAB_CI:
            self._metadata = self._get_gitlab_metadata()
        elif platform == CIPlatform.JENKINS:
            self._metadata = self._get_jenkins_metadata()
        elif platform == CIPlatform.CIRCLECI:
            self._metadata = self._get_circleci_metadata()
        else:
            self._metadata = CIMetadata(
                platform=platform,
                is_ci=platform != CIPlatform.LOCAL,
                build_number=None,
                build_url=None,
                branch=None,
                commit=None,
                pull_request=None,
                actor=None,
                repository=None,
                event_name=None,
            )
        
        return self._metadata
    
    def _get_github_metadata(self) -> CIMetadata:
        """Get GitHub Actions metadata."""
        return CIMetadata(
            platform=CIPlatform.GITHUB_ACTIONS,
            is_ci=True,
            build_number=os.environ.get('GITHUB_RUN_NUMBER'),
            build_url=f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}",
            branch=os.environ.get('GITHUB_REF_NAME'),
            commit=os.environ.get('GITHUB_SHA'),
            pull_request=os.environ.get('GITHUB_EVENT_NAME') == 'pull_request' and os.environ.get('GITHUB_REF', '').split('/')[-2] or None,
            actor=os.environ.get('GITHUB_ACTOR'),
            repository=os.environ.get('GITHUB_REPOSITORY'),
            event_name=os.environ.get('GITHUB_EVENT_NAME'),
        )
    
    def _get_gitlab_metadata(self) -> CIMetadata:
        """Get GitLab CI metadata."""
        return CIMetadata(
            platform=CIPlatform.GITLAB_CI,
            is_ci=True,
            build_number=os.environ.get('CI_PIPELINE_IID'),
            build_url=os.environ.get('CI_PIPELINE_URL'),
            branch=os.environ.get('CI_COMMIT_REF_NAME'),
            commit=os.environ.get('CI_COMMIT_SHA'),
            pull_request=os.environ.get('CI_MERGE_REQUEST_IID'),
            actor=os.environ.get('GITLAB_USER_LOGIN'),
            repository=os.environ.get('CI_PROJECT_PATH'),
            event_name=os.environ.get('CI_PIPELINE_SOURCE'),
        )
    
    def _get_jenkins_metadata(self) -> CIMetadata:
        """Get Jenkins metadata."""
        return CIMetadata(
            platform=CIPlatform.JENKINS,
            is_ci=True,
            build_number=os.environ.get('BUILD_NUMBER'),
            build_url=os.environ.get('BUILD_URL'),
            branch=os.environ.get('GIT_BRANCH') or os.environ.get('BRANCH_NAME'),
            commit=os.environ.get('GIT_COMMIT'),
            pull_request=os.environ.get('CHANGE_ID'),
            actor=os.environ.get('BUILD_USER'),
            repository=os.environ.get('GIT_URL'),
            event_name=None,
        )
    
    def _get_circleci_metadata(self) -> CIMetadata:
        """Get CircleCI metadata."""
        return CIMetadata(
            platform=CIPlatform.CIRCLECI,
            is_ci=True,
            build_number=os.environ.get('CIRCLE_BUILD_NUM'),
            build_url=os.environ.get('CIRCLE_BUILD_URL'),
            branch=os.environ.get('CIRCLE_BRANCH'),
            commit=os.environ.get('CIRCLE_SHA1'),
            pull_request=os.environ.get('CIRCLE_PULL_REQUEST', '').split('/')[-1] or None,
            actor=os.environ.get('CIRCLE_USERNAME'),
            repository=os.environ.get('CIRCLE_REPOSITORY_URL'),
            event_name=None,
        )
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get CI metadata as dictionary for receipts."""
        meta = self.get_metadata()
        return {
            'ci_platform': meta.platform.value,
            'ci_build_number': meta.build_number,
            'ci_build_url': meta.build_url,
            'ci_branch': meta.branch,
            'ci_commit': meta.commit,
            'ci_pull_request': meta.pull_request,
            'ci_actor': meta.actor,
            'ci_repository': meta.repository,
        }


def generate_github_actions_workflow(name: str = 'STUNIR Build',
                                     python_version: str = '3.11',
                                     runs_on: str = 'ubuntu-latest') -> str:
    """Generate a GitHub Actions workflow file.
    
    Args:
        name: Workflow name
        python_version: Python version to use
        runs_on: Runner type
        
    Returns:
        YAML workflow content
    """
    return f"""name: {name}

on:
  push:
    branches: [ main, devsite ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: {runs_on}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '{python_version}'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --tb=short
    
    - name: Verify manifests
      run: |
        python -m tools.manifest.verify_manifest receipts/
    
    - name: Build
      run: |
        ./scripts/build.sh native
"""


def generate_gitlab_ci_config(python_version: str = '3.11') -> str:
    """Generate a GitLab CI configuration file.
    
    Args:
        python_version: Python version to use
        
    Returns:
        YAML configuration content
    """
    return f"""image: python:{python_version}

stages:
  - test
  - build
  - verify

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

before_script:
  - python -m pip install --upgrade pip
  - pip install -e .

test:
  stage: test
  script:
    - pip install pytest pytest-cov
    - pytest tests/ -v --tb=short --junitxml=report.xml
  artifacts:
    reports:
      junit: report.xml

build:
  stage: build
  script:
    - ./scripts/build.sh native
  artifacts:
    paths:
      - receipts/
      - asm/

verify:
  stage: verify
  script:
    - python -m tools.manifest.verify_manifest receipts/
  needs:
    - build
"""


# Module-level convenience functions
def detect_ci_environment() -> CIPlatform:
    """Detect the current CI/CD platform."""
    return CIIntegration().detect_platform()


def is_ci() -> bool:
    """Check if running in a CI environment."""
    return CIIntegration().is_ci()


def get_ci_metadata() -> Dict[str, Any]:
    """Get CI metadata dictionary."""
    return CIIntegration().get_metadata_dict()
