"""Static analyzer for Python Script/Module Review Checklist (docs §6.2).

The analyzer parses Python source with ``ast`` and reports checklist issues:

- imports_ordering: stdlib → third-party → local (finance_ml/relative)
- global_mutable_state: module-level dict/list/set assignment
- missing_type_hints: functions without annotations (params or return)
- print_statements: use of print instead of logging
- training_return_schema: train* functions returning dict missing required keys
- dataset_prep_return: dataset prep functions not returning 5-tuple or DatasetSplit

Design goals: fast, deterministic, no imports/exec of target code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Issue = Dict[str, Any]


def _root_module(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.split(".")[0]


def _classify_import(root: str, level: int = 0) -> str:
    """Classify import as 'stdlib' | 'third' | 'local'."""
    if level and level > 0:
        return "local"

    # Local package (project)
    if root.startswith("finance_ml") or root in {"finance_ml"}:
        return "local"

    # Try Python's stdlib module names (3.10+)
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if isinstance(stdlib_names, set) and root in stdlib_names:
        return "stdlib"

    # Fallback minimal stdlib set
    fallback_stdlib = {
        "os",
        "sys",
        "pathlib",
        "logging",
        "json",
        "re",
        "math",
        "datetime",
        "itertools",
        "functools",
        "typing",
        "subprocess",
        "argparse",
        "unittest",
        "statistics",
        "collections",
        "dataclasses",
    }
    if root in fallback_stdlib:
        return "stdlib"

    return "third"


def _check_import_order(module: ast.Module) -> Optional[Issue]:
    order_map = {"stdlib": 0, "third": 1, "local": 2}
    seen: List[Tuple[str, int]] = []

    for node in module.body:
        if isinstance(node, ast.Import):
            if not node.names:
                continue
            root = _root_module(node.names[0].name)
            cat = _classify_import(root, level=0)
            seen.append((cat, getattr(node, "lineno", 0)))
        elif isinstance(node, ast.ImportFrom):
            root = _root_module(node.module)
            cat = _classify_import(root, level=getattr(node, "level", 0))
            seen.append((cat, getattr(node, "lineno", 0)))

    highest = -1
    for cat, lineno in seen:
        rank = order_map.get(cat, 99)
        if rank < highest:
            return {
                "code": "imports_ordering",
                "message": "Imports must be grouped stdlib → third-party → local",
                "line": lineno,
            }
        highest = max(highest, rank)
    return None


def _is_mutable_constructor(v: ast.AST) -> bool:
    if isinstance(v, (ast.Dict, ast.List, ast.Set)):
        return True
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in {"dict", "list", "set"}:
        return True
    return False


def _check_global_mutable_state(module: ast.Module) -> Optional[Issue]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            if _is_mutable_constructor(node.value):
                return {
                    "code": "global_mutable_state",
                    "message": "Avoid global mutable state at module level",
                    "line": getattr(node, "lineno", 0),
                }
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and _is_mutable_constructor(node.value):
                return {
                    "code": "global_mutable_state",
                    "message": "Avoid global mutable state at module level",
                    "line": getattr(node, "lineno", 0),
                }
    return None


def _function_has_type_hints(fn: ast.FunctionDef) -> bool:
    if fn.returns is None:
        return False
    for arg in list(fn.args.args) + list(fn.args.kwonlyargs):
        if arg.arg in {"self", "cls"}:
            continue
        if arg.annotation is None:
            return False
    if fn.args.vararg and fn.args.vararg.annotation is None:
        return False
    if fn.args.kwarg and fn.args.kwarg.annotation is None:
        return False
    return True


def _check_missing_type_hints(module: ast.Module) -> Optional[Issue]:
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            if not _function_has_type_hints(node):
                return {
                    "code": "missing_type_hints",
                    "message": f"Function '{node.name}' lacks complete type hints",
                    "line": getattr(node, "lineno", 0),
                }
    return None


def _check_print_statements(module: ast.Module) -> Optional[Issue]:
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "print":
                return {
                    "code": "print_statements",
                    "message": "Use logging instead of print statements",
                    "line": getattr(node, "lineno", 0),
                }
    return None


REQUIRED_TRAIN_KEYS = {"model", "metrics", "y_pred", "y_proba", "artifacts"}


def _literal_dict_keys(d: ast.Dict) -> Optional[set[str]]:
    keys: set[str] = set()
    for k in d.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.add(k.value)
        else:
            return None
    return keys


def _check_training_return_schema(module: ast.Module) -> Optional[Issue]:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("train"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    keys = _literal_dict_keys(sub.value)
                    if keys is None:
                        continue
                    missing = REQUIRED_TRAIN_KEYS - keys
                    if missing:
                        return {
                            "code": "training_return_schema",
                            "message": f"Training function '{node.name}' return dict missing keys: {sorted(missing)}",
                            "line": getattr(sub, "lineno", getattr(node, "lineno", 0)),
                        }
    return None


DATASET_FN_HINTS = ("make_dataset", "prepare_dataset", "dataset")


def _ann_id(node: ast.AST) -> Optional[str]:
    # Extract identifier name from annotation
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _ann_id(node.value)
    return None


def _has_5tuple_return(fn: ast.FunctionDef) -> bool:
    # Heuristic: at least one explicit return of a Tuple literal with 5 elts
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Tuple):
            if len(sub.value.elts) == 5:
                return True
    return False


def _check_dataset_prep_return(module: ast.Module) -> Optional[Issue]:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and any(h in node.name for h in DATASET_FN_HINTS):
            ok = False
            if node.returns is not None:
                ann = _ann_id(node.returns)
                if ann in {"DatasetSplit", "DatasetSplit"}:
                    ok = True
            if not ok and _has_5tuple_return(node):
                ok = True
            if not ok:
                return {
                    "code": "dataset_prep_return",
                    "message": f"Function '{node.name}' should return 5-tuple or DatasetSplit",
                    "line": getattr(node, "lineno", 0),
                }
    return None


def review_python_source(source: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Analyze Python source text and report checklist issues.

    Returns a dict with keys:
    - issues: List[Issue]
    - summary: Dict[str, Any]
    """
    try:
        module = ast.parse(source)
    except SyntaxError as e:
        return {
            "issues": [
                {
                    "code": "syntax_error",
                    "message": f"SyntaxError: {e}",
                    "line": getattr(e, "lineno", 0),
                }
            ],
            "summary": {"filename": filename or "<string>", "functions_checked": 0},
        }

    issues: List[Issue] = []

    for check in (
        _check_import_order,
        _check_global_mutable_state,
        _check_missing_type_hints,
        _check_print_statements,
        _check_training_return_schema,
        _check_dataset_prep_return,
    ):
        issue = check(module)
        if issue:
            issues.append(issue)

    fn_count = sum(1 for n in module.body if isinstance(n, ast.FunctionDef))
    return {
        "issues": issues,
        "summary": {
            "filename": filename or "<string>",
            "functions_checked": fn_count,
            "imports_checked": True,
        },
    }


def review_python_file(path: Path | str) -> Dict[str, Any]:
    """Read a Python file and analyze its contents."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return review_python_source(text, filename=str(p))
