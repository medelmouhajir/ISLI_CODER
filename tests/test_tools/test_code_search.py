"""Tests for CodeSearchTool."""

from isli.tools.code_search import CodeSearchTool


def test_code_search_functions_and_classes(temp_project_dir, mock_keeper):
    py_code = """
class AuthService:
    def verify_token(self, token):
        pass

def generate_key():
    return 'secret'
"""
    (temp_project_dir / "service.py").write_text(py_code, encoding="utf-8")

    tool = CodeSearchTool(mock_keeper, temp_project_dir)

    # Search for class
    res_class = tool.execute(query="Auth", symbol_type="class")
    assert "service.py" in res_class
    assert "[class] class AuthService" in res_class

    # Search for function
    res_func = tool.execute(query="verify", symbol_type="function")
    assert "service.py" in res_func
    assert "[function] def verify_token(self, token)" in res_func

    # Search any
    res_any = tool.execute(query="generate")
    assert "[function] def generate_key()" in res_any


def test_code_search_multi_language_regex(temp_project_dir, mock_keeper):
    ts_code = """
    export class UserStore {
        constructor() {}
    }
    export async function fetchUsers() {
        return [];
    }
    """
    (temp_project_dir / "users.ts").write_text(ts_code, encoding="utf-8")

    tool = CodeSearchTool(mock_keeper, temp_project_dir)

    res_class = tool.execute(query="UserStore", symbol_type="class")
    assert "users.ts" in res_class
    assert "UserStore" in res_class

    res_func = tool.execute(query="fetchUsers", symbol_type="function")
    assert "users.ts" in res_func
    assert "fetchUsers" in res_func
