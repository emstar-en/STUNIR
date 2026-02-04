"""
Deductive Reasoning Engine for STUNIR.

Validates findings against formal rules using forward chaining.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
from .base_types import Finding, Severity, Explanation
from .knowledge_graph import KnowledgeGraph, Entity, EntityType, Relationship, RelationshipType


class RuleCategory(Enum):
    IR_FORMAT = "ir_format"
    CODE_GENERATION = "code_generation"
    SECURITY = "security"
    VERSION_CONSISTENCY = "version_consistency"
    NAMING = "naming"
    DOCUMENTATION = "documentation"


@dataclass
class Rule:
    """A deductive rule."""
    id: str
    name: str
    category: RuleCategory
    severity: Severity
    description: str
    premise: Callable[[Finding], bool]
    condition: Callable[[Finding], bool]
    message_template: str
    recommendation: str
    
    def evaluate(self, finding: Finding) -> Optional['Conclusion']:
        """Evaluate rule against a finding."""
        if not self.premise(finding):
            return None
        
        if self.condition(finding):
            return None  # Rule satisfied
        
        return Conclusion(
            rule_id=self.id,
            rule_name=self.name,
            severity=self.severity,
            message=self.message_template.format(**finding.metadata),
            recommendation=self.recommendation,
            finding=finding,
            certainty=1.0
        )


@dataclass
class Conclusion:
    """Result of rule evaluation."""
    rule_id: str
    rule_name: str
    severity: Severity
    message: str
    recommendation: str
    finding: Finding
    certainty: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'recommendation': self.recommendation,
            'file_path': self.finding.file_path,
            'line_number': self.finding.line_number,
            'certainty': self.certainty
        }


class DeductiveEngine:
    """
    Deductive reasoning engine using forward chaining.
    """
    
    def __init__(self):
        self.rules: List[Rule] = []
        self.conclusions: List[Conclusion] = []
        self.facts: Dict[str, Any] = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load the default rule set."""
        
        # IR Format Rules
        self.add_rule(Rule(
            id="D-IR-001",
            name="IR Version Required",
            category=RuleCategory.IR_FORMAT,
            severity=Severity.ERROR,
            description="All .stunir files must have ir_version field",
            premise=lambda f: f.file_path.endswith('.stunir'),
            condition=lambda f: f.metadata.get('has_ir_version', False),
            message_template="IR file '{file_path}' missing required 'ir_version' field",
            recommendation="Add 'ir_version' field with value '1.0'"
        ))
        
        self.add_rule(Rule(
            id="D-IR-002",
            name="Valid IR Version",
            category=RuleCategory.IR_FORMAT,
            severity=Severity.WARNING,
            description="IR version must be valid",
            premise=lambda f: f.file_path.endswith('.stunir') and f.metadata.get('has_ir_version', False),
            condition=lambda f: f.metadata.get('ir_version') in ['1.0', '2.0', '3.0'],
            message_template="Invalid IR version '{ir_version}' in '{file_path}'",
            recommendation="Use valid version: 1.0, 2.0, or 3.0"
        ))
        
        self.add_rule(Rule(
            id="D-IR-003",
            name="Function Structure Complete",
            category=RuleCategory.IR_FORMAT,
            severity=Severity.ERROR,
            description="Functions must have name, return_type, and parameters",
            premise=lambda f: f.type == 'ir_function',
            condition=lambda f: all(k in f.metadata for k in ['has_name', 'has_return_type', 'has_parameters']),
            message_template="Function in '{file_path}' missing required fields",
            recommendation="Ensure all functions have name, return_type, and parameters"
        ))
        
        # Security Rules
        self.add_rule(Rule(
            id="D-SEC-001",
            name="Unsafe Block Documentation",
            category=RuleCategory.SECURITY,
            severity=Severity.WARNING,
            description="Unsafe blocks must be documented",
            premise=lambda f: f.type == 'unsafe_block',
            condition=lambda f: f.metadata.get('has_safety_comment', False),
            message_template="Unsafe block at line {line_number} lacks safety documentation",
            recommendation="Add '// SAFETY: explanation' comment before unsafe block"
        ))
        
        self.add_rule(Rule(
            id="D-SEC-002",
            name="No Eval Usage",
            category=RuleCategory.SECURITY,
            severity=Severity.ERROR,
            description="Avoid eval() and exec() for security",
            premise=lambda f: f.type == 'security_risk',
            condition=lambda f: f.metadata.get('risk_type') not in ['eval', 'exec'],
            message_template="Dangerous {risk_type} usage at line {line_number}",
            recommendation="Replace with safer alternatives or validate input thoroughly"
        ))
        
        self.add_rule(Rule(
            id="D-SEC-003",
            name="Input Validation",
            category=RuleCategory.SECURITY,
            severity=Severity.ERROR,
            description="User input must be validated",
            premise=lambda f: f.type == 'user_input',
            condition=lambda f: f.metadata.get('is_validated', False),
            message_template="Unvalidated user input at line {line_number}",
            recommendation="Add input validation before processing"
        ))
        
        # Version Consistency Rules
        self.add_rule(Rule(
            id="D-VER-001",
            name="Cargo.toml Version Consistency",
            category=RuleCategory.VERSION_CONSISTENCY,
            severity=Severity.ERROR,
            description="Cargo.toml version must match CHANGELOG.md",
            premise=lambda f: f.file_path == 'Cargo.toml',
            condition=lambda f: f.metadata.get('version_matches_changelog', False),
            message_template="Version mismatch: Cargo.toml ({cargo_version}) vs CHANGELOG.md ({changelog_version})",
            recommendation="Update versions to match before release"
        ))
        
        self.add_rule(Rule(
            id="D-VER-002",
            name="Version Reference Consistency",
            category=RuleCategory.VERSION_CONSISTENCY,
            severity=Severity.WARNING,
            description="All version references should be consistent",
            premise=lambda f: f.type == 'version_reference',
            condition=lambda f: f.metadata.get('is_consistent', False),
            message_template="Inconsistent version reference: {version}",
            recommendation="Standardize version references across files"
        ))
        
        # Code Generation Rules
        self.add_rule(Rule(
            id="D-CG-001",
            name="C Function Prototypes",
            category=RuleCategory.CODE_GENERATION,
            severity=Severity.ERROR,
            description="C requires function prototypes",
            premise=lambda f: f.metadata.get('target_language') == 'c',
            condition=lambda f: f.metadata.get('has_prototype', False),
            message_template="Function '{function_name}' missing prototype",
            recommendation="Add function prototype before use"
        ))
        
        self.add_rule(Rule(
            id="D-CG-002",
            name="Rust Unsafe Documentation",
            category=RuleCategory.CODE_GENERATION,
            severity=Severity.ERROR,
            description="Rust unsafe blocks need documentation",
            premise=lambda f: f.metadata.get('target_language') == 'rust',
            condition=lambda f: not f.metadata.get('needs_unsafe', False) or f.metadata.get('has_unsafe_doc', False),
            message_template="Unsafe operation without documentation",
            recommendation="Document safety invariants for unsafe code"
        ))
        
        # Naming Rules
        self.add_rule(Rule(
            id="D-NAM-001",
            name="Consistent Naming Convention",
            category=RuleCategory.NAMING,
            severity=Severity.WARNING,
            description="Follow project naming conventions",
            premise=lambda f: f.type == 'naming_violation',
            condition=lambda f: False,  # Always trigger if premise matches
            message_template="Naming violation: '{name}' does not follow {convention}",
            recommendation="Use consistent naming convention throughout project"
        ))
        
        # Documentation Rules
        self.add_rule(Rule(
            id="D-DOC-001",
            name="Public API Documentation",
            category=RuleCategory.DOCUMENTATION,
            severity=Severity.WARNING,
            description="Public APIs should be documented",
            premise=lambda f: f.metadata.get('is_public', False),
            condition=lambda f: f.metadata.get('has_documentation', False),
            message_template="Public {item_type} '{name}' lacks documentation",
            recommendation="Add docstring or documentation comment"
        ))
    
    def add_rule(self, rule: Rule):
        """Add a rule to the engine."""
        self.rules.append(rule)
    
    def infer(self, findings: List[Finding]) -> List[Conclusion]:
        """
        Apply all rules to findings using forward chaining.
        """
        self.conclusions = []
        
        for finding in findings:
            for rule in self.rules:
                conclusion = rule.evaluate(finding)
                if conclusion:
                    self.conclusions.append(conclusion)
        
        return self.conclusions
    
    def get_conclusions_by_category(self, category: RuleCategory) -> List[Conclusion]:
        """Get conclusions filtered by category."""
        return [c for c in self.conclusions 
                if any(r.category == category for r in self.rules if r.id == c.rule_id)]
    
    def get_conclusions_by_severity(self, severity: Severity) -> List[Conclusion]:
        """Get conclusions filtered by severity."""
        return [c for c in self.conclusions if c.severity == severity]
    
    def explain(self, conclusion: Conclusion) -> Explanation:
        """Generate explanation for a conclusion."""
        return Explanation(
            conclusion=conclusion.message,
            reasoning_chain=[
                f"Finding detected: {conclusion.finding.type}",
                f"Rule applied: {conclusion.rule_name} ({conclusion.rule_id})",
                f"Premise satisfied: {conclusion.finding.file_path}",
                f"Condition violated: {conclusion.message}"
            ],
            evidence=[
                f"File: {conclusion.finding.file_path}",
                f"Line: {conclusion.finding.line_number}",
                f"Severity: {conclusion.severity.value}",
                f"Certainty: {conclusion.certainty:.0%}"
            ],
            confidence=conclusion.certainty,
            method="deductive"
        )
    
    def to_knowledge_graph(self, graph: Optional[KnowledgeGraph] = None) -> KnowledgeGraph:
        """Convert conclusions to knowledge graph."""
        if graph is None:
            graph = KnowledgeGraph()
        
        for conclusion in self.conclusions:
            # Create finding entity
            finding_entity = Entity(
                id=f"finding_{conclusion.finding.id}",
                type=EntityType.FINDING,
                properties=conclusion.finding.to_dict(),
                confidence=conclusion.finding.confidence,
                source="deductive"
            )
            graph.add_entity(finding_entity)
            
            # Create rule entity
            rule_entity = Entity(
                id=f"rule_{conclusion.rule_id}",
                type=EntityType.RULE,
                properties={
                    'name': conclusion.rule_name,
                    'category': conclusion.rule_id.split('-')[1].lower()
                },
                confidence=1.0,
                source="deductive"
            )
            graph.add_entity(rule_entity)
            
            # Create relationship
            graph.add_relationship(Relationship(
                source=rule_entity,
                target=finding_entity,
                type=RelationshipType.EXPLAINS,
                weight=conclusion.certainty
            ))
        
        return graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'total_rules': len(self.rules),
            'total_conclusions': len(self.conclusions),
            'by_severity': {
                sev.value: len(self.get_conclusions_by_severity(sev))
                for sev in Severity
            },
            'by_category': {
                cat.value: len([c for c in self.conclusions 
                               if any(r.category == cat for r in self.rules if r.id == c.rule_id)])
                for cat in RuleCategory
            }
        }
