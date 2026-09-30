"""Code Search Tool — AST and symbol search with Keeper ranking."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from isli.tools.base import BaseTool, ToolSchema

IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".isli", ".pytest_cache"}


class CodeSearchTool(BaseTool):
    """AST symbol search across codebase."""

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="code_search",
            description=(
                "Find code symbols (functions, classes) via AST or regex across the workspace."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Symbol name or substring",
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "'function', 'class', or 'any'",
                        "enum": ["function", "class", "any"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search (default '.')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max symbols (default 20)",
                    },
                },
                "required": ["query"],
            },
            requires_approval=False,
        )

    def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").lower()
        if not query:
            return "Error: 'query' parameter is required."

        sym_type = (kwargs.get("symbol_type") or "any").lower()
        search_path_str = kwargs.get("path", ".") or "."
        max_results = kwargs.get("max_results", 20) or 20

        try:
            root_search = self.resolve_path(search_path_str)
        except PermissionError as e:
            return f"Error: {e}"

        symbols: list[dict[str, Any]] = []

        # Non-Python extensions supported via regex
        REGEX_EXTENSIONS = {
            ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".c", ".cpp", ".h", ".hpp", ".java"
        }

        # Discover code files
        if root_search.is_file():
            target_files = [root_search]
        else:
            target_files = []
            for root, dirs, files in os.walk(root_search):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext == ".py" or ext in REGEX_EXTENSIONS:
                        target_files.append(Path(root) / f)

        import re

        FN_REGEX = re.compile(
            r"^\s*(?:export\s+)?(?:async\s+)?(?:function|def|fn|func)\s+([A-Za-z0-9_]+)\s*(\([^)]*\))?",
            re.MULTILINE,
        )
        CLASS_REGEX = re.compile(
            r"^\s*(?:export\s+)?(?:class|struct|interface|trait|type)\s+([A-Za-z0-9_]+)",
            re.MULTILINE,
        )

        for cf in target_files:
            rel = str(cf.relative_to(self.project_root.resolve()))
            try:
                content = cf.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if cf.suffix == ".py":
                try:
                    tree = ast.parse(content, filename=rel)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                            if sym_type in {"function", "any"} and query in node.name.lower():
                                args = [a.arg for a in node.args.args]
                                symbols.append({
                                    "file": rel,
                                    "line": node.lineno,
                                    "kind": "function",
                                    "name": node.name,
                                    "signature": f"def {node.name}({', '.join(args)})",
                                })
                        elif (
                            isinstance(node, ast.ClassDef)
                            and sym_type in {"class", "any"}
                            and query in node.name.lower()
                        ):
                            symbols.append({
                                "file": rel,
                                "line": node.lineno,
                                "kind": "class",
                                "name": node.name,
                                "signature": f"class {node.name}",
                            })
                except Exception:
                    pass
            else:
                # Regex fallback for other languages (JS/TS, Go, Rust, C/C++, Java)
                for line_num, line in enumerate(content.splitlines(), start=1):
                    fn_match = FN_REGEX.match(line)
                    if fn_match and sym_type in {"function", "any"}:
                        name = fn_match.group(1)
                        if query in name.lower():
                            symbols.append({
                                "file": rel,
                                "line": line_num,
                                "kind": "function",
                                "name": name,
                                "signature": line.strip(),
                            })
                    class_match = CLASS_REGEX.match(line)
                    if class_match and sym_type in {"class", "any"}:
                        name = class_match.group(1)
                        if query in name.lower():
                            symbols.append({
                                "file": rel,
                                "line": line_num,
                                "kind": "class",
                                "name": name,
                                "signature": line.strip(),
                            })

            if len(symbols) >= 150:
                break

        if not symbols:
            return f"No code symbols found matching '{query}'."

        if len(symbols) > 10 and self.keeper.available:
            final_symbols = self.keeper.rank_results(symbols, query=query, top_k=max_results)
        else:
            final_symbols = symbols[:max_results]

        lines = [
            f"{s['file']}:{s['line']} [{s['kind']}] {s['signature']}"
            for s in final_symbols
        ]
        return "\n".join(lines)
