"""
STUNIR Logical Reasoning Analysis System

A comprehensive analysis framework implementing:
- Deductive reasoning (formal rule validation)
- Inductive reasoning (pattern discovery)
- Abductive reasoning (hypothesis generation)
- ANFIS (Adaptive Neuro-Fuzzy Inference System)
- Mamdani and Sugeno fuzzy inference
- Hybrid learning (LSE + Gradient Descent)
- Metaheuristic optimization (PSO + GA)
"""

__version__ = "2.0.0"
__author__ = "STUNIR Analysis Team"

from .knowledge_graph import KnowledgeGraph, Entity, Relationship
from .base_types import Finding, FileMetrics, Severity, Explanation
from .deductive_engine import DeductiveEngine, Rule, Conclusion
from .anfis import ANFIS, GaussianMF

__all__ = [
    'KnowledgeGraph', 'Entity', 'Relationship',
    'Finding', 'FileMetrics', 'Severity', 'Explanation',
    'DeductiveEngine', 'Rule', 'Conclusion',
    'ANFIS', 'GaussianMF',
]
