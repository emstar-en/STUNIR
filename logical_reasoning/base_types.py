"""
Base types and data structures for the logical reasoning system.

Defines common types used across all reasoning engines.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class Severity(Enum):
    """Severity levels for findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class Certainty(Enum):
    """Certainty levels for conclusions."""
    CERTAIN = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    UNCERTAIN = 0.0


@dataclass
class Finding:
    """Base class for analysis findings."""
    id: str
    type: str
    file_path: str
    line_number: Optional[int] = None
    message: str = ""
    severity: Severity = Severity.INFO
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # Which analysis pass generated this
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'message': self.message,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }


@dataclass
class FileMetrics:
    """Metrics for a single file."""
    file_path: str
    lines_of_code: int = 0
    cyclomatic_complexity: float = 0.0
    function_count: int = 0
    test_coverage: float = 0.0
    todo_count: int = 0
    unwrap_count: int = 0
    doc_coverage: float = 0.0
    issue_count: int = 0
    
    # Derived metrics
    @property
    def todo_density(self) -> float:
        """TODOs per 100 lines of code."""
        if self.lines_of_code == 0:
            return 0.0
        return (self.todo_count / self.lines_of_code) * 100
    
    @property
    def unwrap_ratio(self) -> float:
        """Unwrap calls per function."""
        if self.function_count == 0:
            return 0.0
        return self.unwrap_count / self.function_count
    
    @property
    def issue_density(self) -> float:
        """Issues per 100 lines of code."""
        if self.lines_of_code == 0:
            return 0.0
        return (self.issue_count / self.lines_of_code) * 100
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return [
            self.cyclomatic_complexity / 20.0,  # Normalize
            self.test_coverage / 100.0,
            self.todo_density / 10.0,  # Normalize
            self.unwrap_ratio / 5.0,  # Normalize
            self.doc_coverage / 100.0
        ]


@dataclass
class Explanation:
    """Explanation for a conclusion or finding."""
    conclusion: str
    reasoning_chain: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    method: str = ""  # Which reasoning method produced this
    
    def format(self) -> str:
        """Format explanation as readable text."""
        lines = [
            f"CONCLUSION: {self.conclusion}",
            f"CONFIDENCE: {self.confidence:.2%}",
            f"METHOD: {self.method}",
            "",
            "REASONING CHAIN:"
        ]
        for i, step in enumerate(self.reasoning_chain, 1):
            lines.append(f"  {i}. {step}")
        
        lines.extend(["", "EVIDENCE:"])
        for item in self.evidence:
            lines.append(f"  - {item}")
        
        return "\n".join(lines)


@dataclass
class ActionItem:
    """Actionable recommendation."""
    priority: int
    description: str
    effort: str  # "Low", "Medium", "High"
    impact: str  # "Low", "Medium", "High"
    reasoning: str
    confidence: float
    affected_files: List[str] = field(default_factory=list)
    
    @property
    def priority_score(self) -> float:
        """Calculate priority score based on multiple factors."""
        effort_map = {"Low": 1.0, "Medium": 0.5, "High": 0.25}
        impact_map = {"Low": 0.33, "Medium": 0.66, "High": 1.0}
        
        return (self.confidence * 
                effort_map.get(self.effort, 0.5) * 
                impact_map.get(self.impact, 0.5) * 
                (11 - self.priority))  # Invert priority (1 = highest)


class ReasoningEngine:
    """Base class for all reasoning engines."""
    
    def __init__(self, name: str):
        self.name = name
        self.findings: List[Finding] = []
        self.conclusions: List[Any] = []
    
    def process(self, findings: List[Finding]) -> List[Any]:
        """Process findings and return conclusions."""
        raise NotImplementedError
    
    def explain(self, conclusion_id: str) -> Explanation:
        """Generate explanation for a conclusion."""
        raise NotImplementedError
    
    def reset(self):
        """Reset engine state."""
        self.findings = []
        self.conclusions = []
