"""Tests for CLI and CommandHandler."""


from unittest.mock import patch

from isli.cli import create_components
from isli.commands import CommandHandler
from isli.config import Config, load_config
from isli.engine.agent_client import AgentClient
from isli.engine.keeper_client import KeeperClient
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.utils.permissions import PermissionGate


def test_command_handler(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = Config()
    agent = AgentClient(config.agent)
    keeper = KeeperClient(config.keeper)
    session_mgr = SessionManager(SessionStore(tmp_path / "cmd.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()

    handler = CommandHandler(config, agent, keeper, session_mgr, perm_gate)

    # /help
    assert "Available Slash Commands" in handler.handle("/help")

    # /model
    model_list_res = handler.handle("/model")
    assert "Available Top Models" in model_list_res
    assert "1." in model_list_res

    # Test index switching (e.g. /model 1)
    res_index_switch = handler.handle("/model 1")
    assert "Switched cloud LLM model to" in res_index_switch
    assert "openrouter/" in agent.config.model

    res_switch = handler.handle("/model gpt-4o")
    assert "Switched cloud LLM model to" in res_switch
    assert agent.config.model == "gpt-4o"

    # /keeper
    assert "smollm2-135m" in handler.handle("/keeper")
    with patch.object(KeeperClient, "load", return_value=None):
        res_k_switch = handler.handle("/keeper qwen3-0.6b")
    assert "Switched Keeper model to" in res_k_switch
    assert keeper.config.model_name == "qwen3-0.6b"

    # /cost
    assert "Token" in handler.handle("/cost") and "Cost" in handler.handle("/cost")

    # /pwd and /workspace
    pwd_res = handler.handle("/pwd")
    assert "Current Workspace:" in pwd_res
    assert "Folder:" in pwd_res

    ws_res = handler.handle("/workspace")
    assert "Current Workspace:" in ws_res

    # /clear
    session_mgr.add_message("user", "test message")
    assert len(session_mgr.get_history()) == 1
    clear_res = handler.handle("/clear")
    assert "cleared" in clear_res.lower()
    assert len(session_mgr.get_history()) == 0

    # /unknown
    unknown_res = handler.handle("/unknown_cmd")
    assert "Unknown slash command" in unknown_res

    # /save and /load
    save_res = handler.handle("/save my-work")
    assert "my-work" in save_res

    sessions_res = handler.handle("/sessions")
    assert "my-work" in sessions_res

    load_res = handler.handle("/load my-work")
    assert "Loaded session 'my-work'" in load_res


def test_create_components(tmp_path):
    config = load_config(project_root=tmp_path)
    config.keeper.enabled = False
    loop, commands, session_mgr, mcp_manager = create_components(config, tmp_path)

    assert loop is not None
    assert commands is not None
    assert session_mgr is not None
    assert len(loop.tool_engine.get_definitions()) == 10  # 10 tools registered (including tasks & todo)!


def test_command_handler_tasks(tmp_path, monkeypatch):
    from isli.engine.task_planner import TaskPlanner

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = Config()
    agent = AgentClient(config.agent)
    keeper = KeeperClient(config.keeper)
    planner = TaskPlanner()
    session_mgr = SessionManager(SessionStore(tmp_path / "cmd_tasks.db"), planner=planner)
    session_mgr.start_new()
    perm_gate = PermissionGate()
    handler = CommandHandler(config, agent, keeper, session_mgr, perm_gate, planner=planner)

    # Empty tasks
    res_empty = handler.handle("/tasks")
    assert "No active plan tasks" in res_empty

    # Add task
    res_add = handler.handle("/tasks add Write documentation")
    assert "Added task 1" in res_add
    assert len(planner.get_tasks()) == 1

    # List tasks
    res_list = handler.handle("/tasks")
    assert "Write documentation" in res_list
    assert "Active Plan Tasks" in res_list

    # Mark done
    res_done = handler.handle("/tasks done 1")
    assert "Marked task 1 as completed" in res_done
    assert planner.get_task("1").status == "completed"

    # Clear tasks
    res_clear = handler.handle("/tasks clear")
    assert "cleared" in res_clear
    assert len(planner.get_tasks()) == 0


def test_render_token_bar(tmp_path):
    from isli.cli import render_token_bar

    config = Config()
    agent = AgentClient(config.agent)
    bar = render_token_bar(agent, budget=16000, project_root=tmp_path)
    assert tmp_path.name in bar
    assert "Tokens:" in bar
    assert "Cost:" in bar


def test_cli_vram_and_config_commands(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    config = Config()
    agent = AgentClient(config.agent)
    keeper = KeeperClient(config.keeper)
    session_mgr = SessionManager(SessionStore(tmp_path / "cmd2.db"))
    session_mgr.start_new()
    perm_gate = PermissionGate()
    handler = CommandHandler(config, agent, keeper, session_mgr, perm_gate)

    # /keeper telemetry includes device
    keeper_res = handler.handle("/keeper")
    assert "Device:" in keeper_res

    # /config includes device
    config_res = handler.handle("/config")
    assert "Local Keeper SLM:" in config_res
    assert "VRAM" in config_res

    # Test CLI overrides with VRAM flags
    cfg_gpu = load_config(overrides={"keeper_gpu_layers": -1, "keeper_device": "gpu"})
    assert cfg_gpu.keeper.n_gpu_layers == -1
    assert cfg_gpu.keeper.device == "gpu"

    cfg_cpu = load_config(overrides={"keeper_gpu_layers": 0, "keeper_device": "cpu"})
    assert cfg_cpu.keeper.n_gpu_layers == 0
    assert cfg_cpu.keeper.device == "cpu"
