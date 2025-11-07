"""
Documentation Generation Tool for Jupyter Notebooks

Generates comprehensive documentation for notebooks:
1. Extracts docstrings from functions and classes
2. Creates cell-level documentation with descriptions
3. Generates navigation index for notebook structure
4. Produces markdown documentation file

Usage:
    python tools/generate_notebook_docs.py [notebook_path]
    
Example:
    python tools/generate_notebook_docs.py ml_finance_model_main_v10.ipynb
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DocstringExtractor(ast.NodeVisitor):
    """AST visitor to extract docstrings from functions and classes."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        
    def visit_FunctionDef(self, node):
        """Extract function documentation."""
        docstring = ast.get_docstring(node)
        self.functions.append({
            'name': node.name,
            'docstring': docstring or 'No documentation available.',
            'line': node.lineno,
        })
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        """Extract async function documentation."""
        docstring = ast.get_docstring(node)
        self.functions.append({
            'name': node.name,
            'docstring': docstring or 'No documentation available.',
            'line': node.lineno,
            'async': True,
        })
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Extract class documentation."""
        docstring = ast.get_docstring(node)
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_doc = ast.get_docstring(item)
                methods.append({
                    'name': item.name,
                    'docstring': method_doc or 'No documentation available.',
                })
        
        self.classes.append({
            'name': node.name,
            'docstring': docstring or 'No documentation available.',
            'line': node.lineno,
            'methods': methods,
        })
        self.generic_visit(node)


def extract_markdown_headers(source: str) -> List[Tuple[int, str, str]]:
    """Extract markdown headers from cell source.
    
    Args:
        source: Cell source code
        
    Returns:
        List of (level, text, anchor) tuples
    """
    headers = []
    lines = source.split('\n')
    
    for line in lines:
        # Match markdown headers (# Header, ## Header, etc.)
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            # Create anchor (lowercase, replace spaces with hyphens)
            anchor = text.lower().replace(' ', '-').replace('(', '').replace(')', '')
            anchor = re.sub(r'[^a-z0-9-]', '', anchor)
            headers.append((level, text, anchor))
    
    return headers


def extract_cell_purpose(source: str) -> Optional[str]:
    """Extract the purpose/description of a code cell.
    
    Args:
        source: Cell source code
        
    Returns:
        Description string or None
    """
    # Look for comment at the top of the cell
    lines = source.strip().split('\n')
    if not lines:
        return None
    
    # Check first few lines for comments
    comments = []
    for line in lines[:5]:
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('##'):
            comment = stripped.lstrip('#').strip()
            if len(comment) > 10:  # Meaningful comment
                comments.append(comment)
        elif stripped and not stripped.startswith('#'):
            break
    
    if comments:
        return ' '.join(comments)
    
    # Check for triple-quoted strings at the start
    if lines[0].strip().startswith('"""') or lines[0].strip().startswith("'''"):
        for i, line in enumerate(lines):
            if i > 0 and ('"""' in line or "'''" in line):
                doc = '\n'.join(lines[1:i])
                return doc.strip()
    
    return None


def analyze_notebook_structure(notebook_path: str) -> Dict:
    """Analyze notebook structure and extract documentation.
    
    Args:
        notebook_path: Path to .ipynb file
        
    Returns:
        Dict with notebook documentation structure
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    structure = {
        'title': Path(notebook_path).stem.replace('_', ' ').title(),
        'path': notebook_path,
        'cells': [],
        'toc': [],  # Table of contents
        'functions': [],
        'classes': [],
    }
    
    markdown_cell_count = 0
    code_cell_count = 0
    
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell['source'])
        
        if cell_type == 'markdown':
            markdown_cell_count += 1
            
            # Extract headers for TOC
            headers = extract_markdown_headers(source)
            for level, text, anchor in headers:
                structure['toc'].append({
                    'level': level,
                    'text': text,
                    'anchor': anchor,
                    'cell': i,
                })
            
            structure['cells'].append({
                'index': i,
                'type': 'markdown',
                'content': source[:200] + '...' if len(source) > 200 else source,
                'headers': headers,
            })
        
        elif cell_type == 'code':
            code_cell_count += 1
            
            # Extract docstrings
            extractor = DocstringExtractor()
            try:
                tree = ast.parse(source)
                extractor.visit(tree)
                
                structure['functions'].extend([
                    {**func, 'cell': i} for func in extractor.functions
                ])
                structure['classes'].extend([
                    {**cls, 'cell': i} for cls in extractor.classes
                ])
            except SyntaxError:
                pass
            
            # Extract cell purpose
            purpose = extract_cell_purpose(source)
            
            structure['cells'].append({
                'index': i,
                'type': 'code',
                'purpose': purpose,
                'functions': len(extractor.functions),
                'classes': len(extractor.classes),
                'lines': len(source.split('\n')),
            })
    
    structure['stats'] = {
        'total_cells': len(notebook['cells']),
        'markdown_cells': markdown_cell_count,
        'code_cells': code_cell_count,
        'total_functions': len(structure['functions']),
        'total_classes': len(structure['classes']),
    }
    
    return structure


def generate_markdown_documentation(structure: Dict, output_path: str = None) -> str:
    """Generate markdown documentation from notebook structure.
    
    Args:
        structure: Notebook structure from analyze_notebook_structure
        output_path: Optional path to save documentation
        
    Returns:
        Markdown documentation text
    """
    lines = [
        f"# {structure['title']} - Documentation",
        "",
        f"**Source**: `{structure['path']}`",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]
    
    # Generate TOC
    if structure['toc']:
        for entry in structure['toc']:
            indent = "  " * (entry['level'] - 1)
            lines.append(f"{indent}- [{entry['text']}](#{entry['anchor']})")
    else:
        lines.append("*No table of contents available*")
    
    lines.extend([
        "",
        "---",
        "",
        "## Notebook Statistics",
        "",
        f"- **Total Cells**: {structure['stats']['total_cells']}",
        f"- **Markdown Cells**: {structure['stats']['markdown_cells']}",
        f"- **Code Cells**: {structure['stats']['code_cells']}",
        f"- **Functions Defined**: {structure['stats']['total_functions']}",
        f"- **Classes Defined**: {structure['stats']['total_classes']}",
        "",
        "---",
        "",
    ])
    
    # Functions section
    if structure['functions']:
        lines.extend([
            "## Functions",
            "",
            "### Defined Functions",
            "",
        ])
        
        for func in structure['functions']:
            func_name = func['name']
            is_async = func.get('async', False)
            async_prefix = "async " if is_async else ""
            
            lines.extend([
                f"#### `{async_prefix}def {func_name}()`",
                "",
                f"**Defined in**: Cell {func['cell']}",
                "",
                f"**Documentation**:",
                "",
                f"```",
                func['docstring'],
                f"```",
                "",
            ])
    
    # Classes section
    if structure['classes']:
        lines.extend([
            "## Classes",
            "",
            "### Defined Classes",
            "",
        ])
        
        for cls in structure['classes']:
            lines.extend([
                f"#### `class {cls['name']}`",
                "",
                f"**Defined in**: Cell {cls['cell']}",
                "",
                f"**Documentation**:",
                "",
                f"```",
                cls['docstring'],
                f"```",
                "",
            ])
            
            if cls['methods']:
                lines.append("**Methods**:")
                lines.append("")
                for method in cls['methods']:
                    lines.extend([
                        f"##### `{method['name']}()`",
                        "",
                        f"```",
                        method['docstring'],
                        f"```",
                        "",
                    ])
    
    # Cell-by-cell breakdown
    lines.extend([
        "---",
        "",
        "## Cell-by-Cell Breakdown",
        "",
    ])
    
    for cell in structure['cells']:
        cell_idx = cell['index']
        cell_type = cell['type']
        
        if cell_type == 'markdown':
            if cell['headers']:
                header_text = cell['headers'][0][1]  # First header text
                lines.extend([
                    f"### Cell {cell_idx}: Markdown - {header_text}",
                    "",
                ])
            else:
                lines.extend([
                    f"### Cell {cell_idx}: Markdown",
                    "",
                ])
        else:
            lines.extend([
                f"### Cell {cell_idx}: Code",
                "",
            ])
            
            if cell['purpose']:
                lines.extend([
                    f"**Purpose**: {cell['purpose']}",
                    "",
                ])
            
            lines.extend([
                f"**Lines**: {cell['lines']}",
                f"**Functions**: {cell['functions']}",
                f"**Classes**: {cell['classes']}",
                "",
            ])
    
    # Footer
    lines.extend([
        "---",
        "",
        f"*Documentation generated automatically from `{structure['path']}`*",
    ])
    
    doc = "\n".join(lines)
    
    # Save or return
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc)
        print(f"Documentation saved to: {output_path}")
    
    return doc


def generate_report(notebook_path: str, output_path: str = None) -> str:
    """Generate a text-based documentation report.
    
    Args:
        notebook_path: Path to .ipynb file
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    structure = analyze_notebook_structure(notebook_path)
    
    # Build report
    lines = [
        "=" * 80,
        "NOTEBOOK DOCUMENTATION REPORT",
        "=" * 80,
        f"\nNotebook: {notebook_path}",
        "",
        "=" * 80,
        "STATISTICS",
        "=" * 80,
        f"\nTotal Cells:      {structure['stats']['total_cells']}",
        f"Markdown Cells:   {structure['stats']['markdown_cells']}",
        f"Code Cells:       {structure['stats']['code_cells']}",
        f"Functions:        {structure['stats']['total_functions']}",
        f"Classes:          {structure['stats']['total_classes']}",
        "",
    ]
    
    # Table of Contents
    if structure['toc']:
        lines.extend([
            "=" * 80,
            "TABLE OF CONTENTS",
            "=" * 80,
            "",
        ])
        
        for entry in structure['toc']:
            indent = "  " * (entry['level'] - 1)
            lines.append(f"{indent}{entry['text']} (Cell {entry['cell']})")
        
        lines.append("")
    
    # Functions
    if structure['functions']:
        lines.extend([
            "=" * 80,
            f"FUNCTIONS ({len(structure['functions'])} total)",
            "=" * 80,
            "",
        ])
        
        for func in structure['functions'][:20]:  # Limit to 20
            is_async = func.get('async', False)
            async_prefix = "async " if is_async else ""
            lines.extend([
                f"{async_prefix}def {func['name']}() - Cell {func['cell']}",
                f"  {func['docstring'][:100]}..." if len(func['docstring']) > 100 else f"  {func['docstring']}",
                "",
            ])
        
        if len(structure['functions']) > 20:
            lines.append(f"  ... and {len(structure['functions']) - 20} more\n")
    
    # Classes
    if structure['classes']:
        lines.extend([
            "=" * 80,
            f"CLASSES ({len(structure['classes'])} total)",
            "=" * 80,
            "",
        ])
        
        for cls in structure['classes']:
            lines.extend([
                f"class {cls['name']} - Cell {cls['cell']}",
                f"  {cls['docstring'][:100]}..." if len(cls['docstring']) > 100 else f"  {cls['docstring']}",
                f"  Methods: {len(cls['methods'])}",
                "",
            ])
    
    # Recommendations
    lines.extend([
        "=" * 80,
        "RECOMMENDATIONS",
        "=" * 80,
        "",
    ])
    
    undocumented_functions = sum(
        1 for f in structure['functions']
        if f['docstring'] == 'No documentation available.'
    )
    
    if undocumented_functions > 0:
        lines.append(f"- Add docstrings to {undocumented_functions} undocumented functions")
    else:
        lines.append("✓ All functions have docstrings")
    
    if not structure['toc']:
        lines.append("- Add markdown headers to create table of contents")
    else:
        lines.append(f"✓ Table of contents available ({len(structure['toc'])} sections)")
    
    code_without_purpose = sum(
        1 for c in structure['cells']
        if c['type'] == 'code' and not c.get('purpose')
    )
    
    if code_without_purpose > 0:
        lines.append(f"- Add comments to {code_without_purpose} code cells to describe their purpose")
    
    lines.extend([
        "",
        "=" * 80,
    ])
    
    report = "\n".join(lines)
    
    # Save or print
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)
    
    return report


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_notebook_docs.py <notebook_path> [output_path]")
        print("\nExample:")
        print("  python tools/generate_notebook_docs.py ml_finance_model_main_v10.ipynb")
        print("  python tools/generate_notebook_docs.py ml_finance_model_main_v10.ipynb docs/notebook_docs.md")
        print("\nOutput formats:")
        print("  - .txt: Text report")
        print("  - .md: Markdown documentation")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(notebook_path).exists():
        print(f"Error: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    # Analyze structure
    structure = analyze_notebook_structure(notebook_path)
    
    # Generate appropriate output
    if output_path and output_path.endswith('.md'):
        generate_markdown_documentation(structure, output_path)
    else:
        generate_report(notebook_path, output_path)


if __name__ == "__main__":
    main()
