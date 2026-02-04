#!/usr/bin/env python3
"""
STUNIR Logical Reasoning Analysis - Main Execution Script

Runs the complete logical reasoning analysis pipeline:
1. Loads findings from previous 20-pass scan
2. Applies deductive, inductive, and abductive reasoning
3. Trains ANFIS on codebase metrics
4. Generates comprehensive report
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logical_reasoning.base_types import Finding, FileMetrics, Severity
from logical_reasoning.deductive_engine import DeductiveEngine
from logical_reasoning.anfis import ANFIS


def load_findings(filepath: str) -> List[Finding]:
    """Load findings from the 20-pass analysis report."""
    findings = []
    
    if not os.path.exists(filepath):
        print(f"Warning: Findings file not found: {filepath}")
        return findings
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert findings to Finding objects
    for pass_name, pass_findings in data.get('findings', {}).items():
        for finding_data in pass_findings:
            finding = Finding(
                id=f"{pass_name}_{len(findings)}",
                type=finding_data.get('type', 'unknown'),
                file_path=finding_data.get('file', 'unknown'),
                line_number=finding_data.get('line'),
                message=finding_data.get('message', ''),
                severity=Severity(finding_data.get('severity', 'info').lower()),
                confidence=finding_data.get('confidence', 1.0),
                metadata=finding_data.get('metadata', {}),
                source=pass_name
            )
            findings.append(finding)
    
    return findings


def generate_training_data(findings: List[Finding]) -> tuple:
    """
    Generate training data from codebase findings.
    
    Features:
    - complexity_score: Normalized cyclomatic complexity
    - test_coverage: Percentage of code covered by tests
    - todo_density: TODOs per 100 lines
    - unwrap_ratio: Unwrap calls per function
    - doc_coverage: Documented functions / total functions
    
    Target:
    - quality_score: Derived from issue density (100 - issue_density)
    """
    # Aggregate metrics by file
    file_metrics: Dict[str, FileMetrics] = {}
    
    for finding in findings:
        file_path = finding.file_path
        
        if file_path not in file_metrics:
            file_metrics[file_path] = FileMetrics(file_path=file_path)
        
        metrics = file_metrics[file_path]
        
        # Update metrics based on finding type
        if finding.type == 'complexity':
            metrics.cyclomatic_complexity = finding.metadata.get('complexity', 0)
        elif finding.type == 'todo':
            metrics.todo_count += 1
        elif finding.type == 'unwrap':
            metrics.unwrap_count += 1
        elif finding.type == 'function':
            metrics.function_count += 1
            if finding.metadata.get('has_documentation', False):
                metrics.doc_coverage += 1
        elif finding.type == 'test':
            metrics.test_coverage = finding.metadata.get('coverage', 0)
        elif finding.type == 'issue':
            metrics.issue_count += 1
        
        # Estimate lines of code
        if finding.line_number:
            metrics.lines_of_code = max(metrics.lines_of_code, finding.line_number)
    
    # Normalize doc_coverage
    for metrics in file_metrics.values():
        if metrics.function_count > 0:
            metrics.doc_coverage = (metrics.doc_coverage / metrics.function_count) * 100
    
    # Convert to feature vectors
    X = []
    Y = []
    
    for metrics in file_metrics.values():
        if metrics.lines_of_code > 0:  # Only include files with code
            features = metrics.to_feature_vector()
            target = 100 - metrics.issue_density  # Higher = better quality
            
            X.append(features)
            Y.append(target)
    
    return np.array(X), np.array(Y), list(file_metrics.values())


def run_deductive_analysis(findings: List[Finding]) -> Dict:
    """Run deductive reasoning on findings."""
    print("\n" + "="*60)
    print("DEDUCTIVE REASONING")
    print("="*60)
    
    engine = DeductiveEngine()
    conclusions = engine.infer(findings)
    
    stats = engine.get_statistics()
    print(f"\nTotal Rules: {stats['total_rules']}")
    print(f"Conclusions: {stats['total_conclusions']}")
    print(f"\nBy Severity:")
    for sev, count in stats['by_severity'].items():
        if count > 0:
            print(f"  {sev}: {count}")
    
    print(f"\nBy Category:")
    for cat, count in stats['by_category'].items():
        if count > 0:
            print(f"  {cat}: {count}")
    
    return {
        'conclusions': [c.to_dict() for c in conclusions],
        'statistics': stats
    }


def train_anfis_model(X: np.ndarray, Y: np.ndarray) -> Dict:
    """Train ANFIS model on codebase metrics."""
    print("\n" + "="*60)
    print("ANFIS TRAINING")
    print("="*60)
    
    if len(X) < 10:
        print("Insufficient data for ANFIS training (need >= 10 samples)")
        return {'error': 'Insufficient data'}
    
    # Create and train ANFIS
    n_inputs = X.shape[1]
    n_rules = min(12, len(X) // 3)  # Adaptive rule count
    
    print(f"\nANFIS Configuration:")
    print(f"  Inputs: {n_inputs}")
    print(f"  Rules: {n_rules}")
    print(f"  Training samples: {len(X)}")
    
    anfis = ANFIS(n_inputs=n_inputs, n_rules=n_rules, n_outputs=1)
    
    # Train
    history = anfis.train(
        X=X,
        Y=Y,
        epochs=100,
        learning_rate=0.01,
        early_stopping_patience=15,
        verbose=True
    )
    
    # Evaluate
    metrics = anfis.evaluate(X, Y)
    
    print(f"\nTraining Complete:")
    print(f"  Epochs: {history['epochs_trained']}")
    print(f"  Final Train MSE: {history['final_train_mse']:.4f}")
    print(f"  Final Val MSE: {history['final_val_mse']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    
    # Extract learned rules
    rules = anfis.get_fuzzy_rules()
    print(f"\nLearned Fuzzy Rules (showing first 3):")
    for rule in rules[:3]:
        print(f"  {rule}")
    
    # Save model
    model_path = Path(__file__).parent / 'anfis_model.json'
    anfis.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    return {
        'history': history,
        'metrics': metrics,
        'rules': rules,
        'model_path': str(model_path)
    }


def generate_comprehensive_report(
    deductive_results: Dict,
    anfis_results: Dict,
    file_metrics: List[FileMetrics],
    output_path: str
) -> None:
    """Generate comprehensive analysis report."""
    
    report = {
        'metadata': {
            'analysis_type': 'Logical Reasoning Analysis',
            'version': '2.0.0',
            'methods': ['deductive', 'anfis', 'fuzzy_inference']
        },
        'summary': {
            'total_files_analyzed': len(file_metrics),
            'total_deductive_conclusions': deductive_results['statistics']['total_conclusions'],
            'anfis_r_squared': anfis_results.get('metrics', {}).get('r2', 0),
            'anfis_rmse': anfis_results.get('metrics', {}).get('rmse', 0)
        },
        'deductive_analysis': deductive_results,
        'anfis_model': {
            'training_history': anfis_results.get('history', {}),
            'performance_metrics': anfis_results.get('metrics', {}),
            'fuzzy_rules': anfis_results.get('rules', [])
        },
        'file_metrics': [
            {
                'file': m.file_path,
                'complexity': m.cyclomatic_complexity,
                'test_coverage': m.test_coverage,
                'todo_density': m.todo_density,
                'unwrap_ratio': m.unwrap_ratio,
                'doc_coverage': m.doc_coverage,
                'issue_density': m.issue_density
            }
            for m in file_metrics[:20]  # Top 20 files
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"REPORT SAVED: {output_path}")
    print("="*60)


def main():
    """Main execution function."""
    print("="*60)
    print("STUNIR LOGICAL REASONING ANALYSIS")
    print("="*60)
    print("\nInitializing analysis pipeline...")
    
    # Load findings from previous analysis
    findings_file = Path(__file__).parent.parent / 'analysis_report.json'
    print(f"\nLoading findings from: {findings_file}")
    
    findings = load_findings(str(findings_file))
    print(f"Loaded {len(findings)} findings")
    
    if len(findings) == 0:
        print("\nNo findings to analyze. Creating synthetic data for demonstration...")
        # Create synthetic findings for demonstration
        findings = create_synthetic_findings()
        print(f"Created {len(findings)} synthetic findings")
    
    # Generate training data
    print("\nGenerating training data from codebase metrics...")
    X, Y, file_metrics = generate_training_data(findings)
    print(f"Generated {len(X)} training samples")
    
    # Run deductive analysis
    deductive_results = run_deductive_analysis(findings)
    
    # Train ANFIS
    anfis_results = train_anfis_model(X, Y)
    
    # Generate report
    output_path = Path(__file__).parent.parent / 'logical_reasoning_report.json'
    generate_comprehensive_report(
        deductive_results,
        anfis_results,
        file_metrics,
        str(output_path)
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_path}")


def create_synthetic_findings() -> List[Finding]:
    """Create synthetic findings for demonstration."""
    findings = []
    
    # Create findings for 50 files
    for i in range(50):
        file_path = f"src/file_{i}.rs"
        
        # Complexity finding
        findings.append(Finding(
            id=f"complexity_{i}",
            type='complexity',
            file_path=file_path,
            line_number=10,
            message=f"Cyclomatic complexity: {np.random.randint(1, 30)}",
            severity=Severity.INFO,
            metadata={'complexity': np.random.randint(1, 30)}
        ))
        
        # TODO finding
        if np.random.random() > 0.5:
            for _ in range(np.random.randint(1, 5)):
                findings.append(Finding(
                    id=f"todo_{i}_{len(findings)}",
                    type='todo',
                    file_path=file_path,
                    line_number=np.random.randint(1, 100),
                    message='TODO comment found',
                    severity=Severity.INFO
                ))
        
        # Unwrap finding
        if np.random.random() > 0.3:
            for _ in range(np.random.randint(1, 10)):
                findings.append(Finding(
                    id=f"unwrap_{i}_{len(findings)}",
                    type='unwrap',
                    file_path=file_path,
                    line_number=np.random.randint(1, 100),
                    message='unwrap() usage',
                    severity=Severity.WARNING
                ))
        
        # Function finding
        for _ in range(np.random.randint(1, 20)):
            findings.append(Finding(
                id=f"func_{i}_{len(findings)}",
                type='function',
                file_path=file_path,
                line_number=np.random.randint(1, 100),
                message='Function definition',
                severity=Severity.INFO,
                metadata={'has_documentation': np.random.random() > 0.3}
            ))
        
        # Issue finding
        if np.random.random() > 0.4:
            for _ in range(np.random.randint(1, 5)):
                findings.append(Finding(
                    id=f"issue_{i}_{len(findings)}",
                    type='issue',
                    file_path=file_path,
                    line_number=np.random.randint(1, 100),
                    message='Code issue detected',
                    severity=Severity.WARNING if np.random.random() > 0.5 else Severity.ERROR
                ))
    
    return findings


if __name__ == '__main__':
    main()
