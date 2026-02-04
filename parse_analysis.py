#!/usr/bin/env python3
"""Parse analysis report and show key findings"""
import json
from collections import Counter
from pathlib import Path

summary_path = Path('analysis') / 'analysis_report.summary.json'
legacy_path = Path('analysis_report.json')

if summary_path.exists():
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    data = {
        "summary": summary.get("summary", {}),
        "metadata": summary.get("metadata", {}),
        "by_pass": summary.get("by_pass", {}),
        "by_file": summary.get("by_file", {}),
        "by_type": summary.get("by_type", {}),
        "severity_summary": summary.get("severity_summary", {})
    }
else:
    with open(legacy_path, 'r', encoding='utf-8') as f:
        legacy = json.load(f)
    data = {
        "summary": legacy.get("summary", {}),
        "metadata": legacy.get("metadata", {}),
        "by_pass": legacy.get("summary", {}).get("by_pass", {}),
        "findings": legacy.get("findings", {})
    }

print("=" * 60)
print("STUNIR CODEBASE ANALYSIS - PRE-RELEASE CHECK")
print("=" * 60)
print()
print(f"Total Findings: {data['summary'].get('total_findings', 0):,}")
print(f"Passes Run: {len(data['by_pass']) if 'by_pass' in data else len(data['summary'].get('by_pass', {}))}")
print()

print("=" * 60)
print("FINDINGS BY PASS (Top 10)")
print("=" * 60)
by_pass = [(k, v) for k, v in (data.get('by_pass') or data['summary'].get('by_pass', {})).items()]
by_pass.sort(key=lambda x: x[1], reverse=True)
for pass_name, count in by_pass[:10]:
    print(f"  {pass_name:30s}: {count:5d}")
print()

print("=" * 60)
print("SEVERITY BREAKDOWN")
print("=" * 60)
if 'severity_summary' in data:
    for sev, count in sorted(data['severity_summary'].items(), key=lambda x: -x[1]):
        print(f"  {sev:15s}: {count:5d}")
else:
    severities = Counter()
    for cat, findings in data['findings'].items():
        for f in findings:
            severities[f.get('severity', 'unknown')] += 1
    for sev, count in severities.most_common():
        print(f"  {sev:15s}: {count:5d}")
print()

print("=" * 60)
print("CATEGORY BREAKDOWN (All)")
print("=" * 60)
if 'by_type' in data:
    for cat, count in sorted(data['by_type'].items(), key=lambda x: x[1], reverse=True):
        if count:
            print(f"  {cat:30s}: {count:5d}")
else:
    for cat, findings in sorted(data['findings'].items(), key=lambda x: len(x[1]), reverse=True):
        if findings:
            print(f"  {cat:30s}: {len(findings):5d}")
print()

print("=" * 60)
print("SAMPLE FINDINGS (First 5)")
print("=" * 60)
if 'by_file' in data:
    count = 0
    for file_path, entry in data['by_file'].items():
        if count >= 5:
            break
        types = entry.get('types', {})
        top_type = sorted(types.items(), key=lambda x: -x[1])[0][0] if types else "unknown"
        print(f"\n[SUMMARY] {file_path}")
        print(f"  Count: {entry.get('count', 0)}")
        print(f"  Top Type: {top_type}")
        count += 1
else:
    count = 0
    for cat, findings in data['findings'].items():
        for f in findings[:2]:
            if count >= 5:
                break
            print(f"\n[{f.get('severity', 'unknown').upper()}] {cat}")
            print(f"  File: {f.get('file', 'unknown')}")
            msg = f.get('message', 'N/A')
            print(f"  Message: {msg[:80]}...")
            count += 1
        if count >= 5:
            break
