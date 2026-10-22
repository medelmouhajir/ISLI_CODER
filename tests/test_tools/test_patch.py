"""Tests for PatchTool and multi-file unified diff engine."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.tools.patch import PatchTool
from isli.utils.patch import apply_file_patch, parse_multi_file_diff


def test_parse_multi_file_diff():
    diff_text = """
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,4 @@
 def foo():
+    x = 1
     return True
--- a/src/bar.py
+++ b/src/bar.py
@@ -5,2 +5,3 @@
 print("hello")
+print("world")
"""
    patches = parse_multi_file_diff(diff_text)
    assert len(patches) == 2
    assert patches[0].new_path == "src/foo.py"
    assert len(patches[0].hunks) == 1
    assert patches[1].new_path == "src/bar.py"
    assert len(patches[1].hunks) == 1


def test_patch_single_file_success(temp_project_dir: Path, mock_keeper: KeeperClient):
    file = temp_project_dir / "code.py"
    file.write_text("x = 10\ny = 20\nz = 30\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    diff = """--- a/code.py
+++ b/code.py
@@ -1,3 +1,3 @@
 x = 10
-y = 20
+y = 42
 z = 30
"""
    res = tool.execute(patch=diff)

    assert "Successfully patched 1 file" in res
    assert "code.py" in res
    assert file.read_text(encoding="utf-8") == "x = 10\ny = 42\nz = 30\n"
    assert tool.last_metadata.get("modified_files") == ["code.py"]


def test_patch_multi_file_success(temp_project_dir: Path, mock_keeper: KeeperClient):
    f1 = temp_project_dir / "f1.txt"
    f2 = temp_project_dir / "f2.txt"
    f1.write_text("alpha\nbeta\n", encoding="utf-8")
    f2.write_text("one\ntwo\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    diff = """--- a/f1.txt
+++ b/f1.txt
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
--- a/f2.txt
+++ b/f2.txt
@@ -1,2 +1,3 @@
 one
+one-and-half
 two
"""
    res = tool.execute(patch=diff)

    assert "Successfully patched 2 file(s)" in res
    assert f1.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert f2.read_text(encoding="utf-8") == "one\none-and-half\ntwo\n"
    assert set(tool.last_metadata.get("modified_files", [])) == {"f1.txt", "f2.txt"}


def test_patch_multi_hunk_offsets(temp_project_dir: Path, mock_keeper: KeeperClient):
    file = temp_project_dir / "lines.txt"
    initial_content = "\n".join(f"line_{i}" for i in range(1, 20)) + "\n"
    file.write_text(initial_content, encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    # Hunk 1 inserts 3 new lines at line 2.
    # Hunk 2 targets line 15, which will have shifted down by 3 lines!
    diff = """--- a/lines.txt
+++ b/lines.txt
@@ -2,2 +2,5 @@
 line_2
+added_a
+added_b
+added_c
 line_3
@@ -14,2 +17,2 @@
 line_14
-line_15
+line_15_modified
"""
    res = tool.execute(patch=diff)

    assert "Successfully patched" in res
    content = file.read_text(encoding="utf-8")
    assert "added_a\nadded_b\nadded_c" in content
    assert "line_15_modified" in content


def test_patch_fuzzy_line_matching(temp_project_dir: Path, mock_keeper: KeeperClient):
    file = temp_project_dir / "fuzzy.py"
    # Content has extra lines at top, making the expected hunk line number slightly off
    file.write_text("# prelude 1\n# prelude 2\nx = 1\ny = 2\nz = 3\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    # Diff says old_start=1, but the context 'x = 1' is actually at line 3
    diff = """--- a/fuzzy.py
+++ b/fuzzy.py
@@ -1,3 +1,3 @@
 x = 1
-y = 2
+y = 99
 z = 3
"""
    res = tool.execute(patch=diff)

    assert "Successfully patched" in res
    assert "y = 99" in file.read_text(encoding="utf-8")


def test_patch_atomic_rollback_on_hunk_failure(
    temp_project_dir: Path, mock_keeper: KeeperClient
):
    f1 = temp_project_dir / "clean1.txt"
    f2 = temp_project_dir / "clean2.txt"
    f1.write_text("hello world\n", encoding="utf-8")
    f2.write_text("goodbye world\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    # Hunk 1 for clean1.txt matches.
    # Hunk 2 for clean2.txt fails because expected context does not exist.
    diff = """--- a/clean1.txt
+++ b/clean1.txt
@@ -1,1 +1,1 @@
-hello world
+hello universe
--- a/clean2.txt
+++ b/clean2.txt
@@ -1,1 +1,1 @@
-nonexistent string
+something else
"""
    res = tool.execute(patch=diff)

    assert "Patch rejected" in res
    assert "No files were modified" in res
    # Ensure f1 was NOT modified!
    assert f1.read_text(encoding="utf-8") == "hello world\n"
    assert f2.read_text(encoding="utf-8") == "goodbye world\n"


def test_patch_python_syntax_error_rejected(
    temp_project_dir: Path, mock_keeper: KeeperClient
):
    file = temp_project_dir / "valid.py"
    file.write_text("def test():\n    return 42\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    diff = """--- a/valid.py
+++ b/valid.py
@@ -1,2 +1,2 @@
 def test():
-    return 42
+    return (broken syntax
"""
    res = tool.execute(patch=diff)

    assert "Patch rejected" in res
    assert "syntax error" in res.lower()
    assert "No files were modified" in res
    assert "def test():\n    return 42\n" == file.read_text(encoding="utf-8")


def test_patch_create_and_delete_file(temp_project_dir: Path, mock_keeper: KeeperClient):
    del_file = temp_project_dir / "to_delete.txt"
    del_file.write_text("temporary content\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    diff = """--- /dev/null
+++ b/new_file.txt
@@ -0,0 +1,2 @@
+line 1
+line 2
--- a/to_delete.txt
+++ /dev/null
@@ -1,1 +0,0 @@
-temporary content
"""
    res = tool.execute(patch=diff)

    assert "Successfully patched" in res
    new_f = temp_project_dir / "new_file.txt"
    assert new_f.exists()
    assert "line 1\nline 2" in new_f.read_text(encoding="utf-8")
    assert not del_file.exists()


def test_patch_strip_markdown_code_fences(
    temp_project_dir: Path, mock_keeper: KeeperClient
):
    file = temp_project_dir / "target.py"
    file.write_text("msg = 'hello'\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    diff = """```diff
--- a/target.py
+++ b/target.py
@@ -1,1 +1,1 @@
-msg = 'hello'
+msg = 'greetings'
```"""
    res = tool.execute(patch=diff)

    assert "Successfully patched" in res
    assert "msg = 'greetings'" in file.read_text(encoding="utf-8")


def test_patch_no_headers_with_path_arg(
    temp_project_dir: Path, mock_keeper: KeeperClient
):
    file = temp_project_dir / "bare.py"
    file.write_text("a = 1\nb = 2\n", encoding="utf-8")

    tool = PatchTool(mock_keeper, temp_project_dir)
    diff = """@@ -1,2 +1,2 @@
 a = 1
-b = 2
+b = 42
"""
    res = tool.execute(path="bare.py", patch=diff)

    assert "Successfully patched" in res
    assert "b = 42" in file.read_text(encoding="utf-8")


def test_patch_keeper_validation_failure(temp_project_dir: Path):
    config = KeeperConfig(enabled=True)
    keeper = KeeperClient(config)
    keeper._loaded = True
    keeper._llm = MagicMock()
    keeper._llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"valid": false, "issues": ["Invalid configuration key"], '
                        '"suggestion": "Use server_name instead"}'
                    )
                }
            }
        ]
    }

    file = temp_project_dir / "config.txt"
    file.write_text("host: localhost\n", encoding="utf-8")

    tool = PatchTool(keeper, temp_project_dir)
    diff = """--- a/config.txt
+++ b/config.txt
@@ -1,1 +1,2 @@
 host: localhost
+invalid_key: true
"""
    res = tool.execute(patch=diff)

    assert "Keeper edit validation failed" in res
    assert "Invalid configuration key" in res
    assert "invalid_key: true" not in file.read_text(encoding="utf-8")
