"""Static analysis helper for the UAV Risk project.
Generates:
 - file -> imports
 - file -> defined functions/classes
 - file -> called function names
 - cross-reference: function -> definition file(s)
 - heuristics for risky external calls (joblib, FAISS, AsyncGroq, hmac, os.getenv, open(index.signature))

Run this script from repo root: python3 scripts/static_analysis.py
"""
# STAGE6_CLEANUP_REVIEW:
# Classification: STAGE6_STATIC_ANALYSIS_TOOLING_KEEP
# Plan lineage: STAGE6_SUPPORT_TOOLING
# Runtime status: Manual static analysis helper; not API/runtime.
# Legacy signal: Heuristic scanner may include old risk keywords and compatibility symbols.
# Replacement: No runtime replacement; use only as cleanup/audit support.
# Action rule: Keep as manual tooling. Do not use as final readiness evidence.
import ast
import os
import json
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), '..')
SRC = os.path.abspath(os.path.join(ROOT, 'src'))

results = {
    'files': {},
    'function_defs': defaultdict(list),
    'call_sites': defaultdict(list),
    'risky_usages': defaultdict(list),
}

RISK_KEYWORDS = [
    'joblib.load', 'FAISS.load_local', 'FAISS.from_documents', 'AsyncGroq', 'groq',
    'hmac.new', 'os.getenv', 'open(', 'index.signature', 'allow_dangerous_deserialization',
    'verify_and_safely_load_faiss', 'safe_load_bundle', 'assemble_feature_vector_from_dict',
    'DataValidator', 'get_imputed_value', 'get_safe_value'
]


def extract_name(node):
    # Try to get a dotted name from ast nodes
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return '.'.join(reversed(parts))
    return None


def analyze_file(path):
    rel = os.path.relpath(path, SRC)
    data = {
        'imports': [],
        'from_imports': [],
        'functions': [],
        'classes': [],
        'calls': [],
        'risky': []
    }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        tree = ast.parse(src, filename=path)
    except Exception as e:
        data['error'] = str(e)
        return rel, data

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                data['imports'].append(n.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for n in node.names:
                data['from_imports'].append((module, n.name))
        elif isinstance(node, ast.FunctionDef):
            data['functions'].append({'name': node.name, 'lineno': node.lineno})
            results['function_defs'][node.name].append(rel)
        elif isinstance(node, ast.ClassDef):
            data['classes'].append({'name': node.name, 'lineno': node.lineno})
            results['function_defs'][node.name].append(rel)
        elif isinstance(node, ast.Call):
            fn = extract_name(node.func)
            if fn:
                data['calls'].append({'name': fn, 'lineno': node.lineno})
                results['call_sites'][fn].append(rel)
                # heuristics risky usage detection
                for kw in RISK_KEYWORDS:
                    if kw in fn or kw in ast.dump(node):
                        data['risky'].append({'kw': kw, 'call': fn, 'lineno': node.lineno})
                        results['risky_usages'][kw].append({'file': rel, 'call': fn, 'lineno': node.lineno})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if 'index.signature' in node.value:
                data['risky'].append({'kw': 'index.signature', 'call': 'literal', 'lineno': node.lineno})
                results['risky_usages']['index.signature'].append({'file': rel, 'call': 'literal', 'lineno': node.lineno})

    return rel, data


def walk_src():
    for root, dirs, files in os.walk(SRC):
        for fname in files:
            if not fname.endswith('.py'):
                continue
            path = os.path.join(root, fname)
            rel, data = analyze_file(path)
            results['files'][rel] = data


if __name__ == '__main__':
    walk_src()
    out_path = os.path.join(ROOT, 'analysis', 'static_analysis.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fo:
        json.dump(results, fo, indent=2, ensure_ascii=False)
    # print concise summary
    print('\nStatic analysis summary:')
    print(f"- scanned files: {len(results['files'])}")
    print(f"- unique function defs found: {len(results['function_defs'])}")
    print(f"- unique call targets found: {len(results['call_sites'])}")
    print('\nTop risky usages detected:')
    for kw, occurrences in results['risky_usages'].items():
        print(f"  - {kw}: {len(occurrences)} occurrences (sample: {occurrences[:3]})")
    print('\nFiles with most calls:')
    files_call_counts = [(f, len(d['calls'])) for f, d in results['files'].items()]
    files_call_counts.sort(key=lambda x: x[1], reverse=True)
    for f, c in files_call_counts[:8]:
        print(f"  - {f}: {c} calls")
    print('\nWrote detailed JSON to:', out_path)
