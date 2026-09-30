"""End-to-end integration tests for ISLI."""

from unittest.mock import MagicMock

from isli.engine.agent_client import AgentClient, AgentResponse
from isli.engine.context_manager import ContextManager
from isli.engine.keeper_client import KeeperClient, KeeperConfig
from isli.engine.prompt_assembler import PromptAssembler
from isli.engine.react_loop import ReActLoop
from isli.engine.tool_engine import ToolEngine
from isli.memory.session import SessionManager
from isli.memory.store import SessionStore
from isli.tools import register_all
from isli.utils.permissions import PermissionGate


def test_full_agentic_workflow(temp_project_dir):
    # 1. Setup Keeper
    keeper_cfg = KeeperConfig(enabled=True)
    keeper = KeeperClient(keeper_cfg)
    keeper._loaded = True
    keeper._llm = MagicMock()
    # Keeper validation returns valid
    keeper._llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"valid": true, "issues": [], "suggestion": null}'}}]
    }

    # 2. Setup ToolEngine with all 8 tools
    engine = ToolEngine(keeper, temp_project_dir)
    register_all(engine, keeper, temp_project_dir)

    # 3. Setup Context & Memory
    (temp_project_dir / "ISLI.md").write_text("Project instructions.", encoding="utf-8")
    ctx_mgr = ContextManager(temp_project_dir)
    prompt_asm = PromptAssembler(engine, ctx_mgr)
    session_store = SessionStore(temp_project_dir / "sessions.db")
    session_mgr = SessionManager(session_store)
    session_mgr.start_new(name="integration-run")

    perm_gate = PermissionGate()
    perm_gate.grant_session("write")
    perm_gate.grant_session("edit")

    # 4. Mock Agent Client to simulate multi-step problem solving:
    # Step 1: Write a python file
    # Step 2: Edit the file
    # Step 3: Read back and confirm
    agent = MagicMock(spec=AgentClient)
    agent.complete.side_effect = [
        AgentResponse(
            content="I will create the calculator module.",
            tool_calls=[{
                "id": "c1",
                "name": "write",
                "arguments": {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"},
            }],
        ),
        AgentResponse(
            content="I noticed a bug in add(). I will edit it.",
            tool_calls=[{
                "id": "c2",
                "name": "edit",
                "arguments": {
                    "path": "calc.py",
                    "target": "return a - b",
                    "replacement": "return a + b",
                },
            }],
        ),
        AgentResponse(
            content="The calculator module has been created and verified.",
            tool_calls=[],
        ),
    ]

    loop = ReActLoop(
        agent=agent,
        tool_engine=engine,
        prompt_assembler=prompt_asm,
        keeper=keeper,
        permission_gate=perm_gate,
        session_manager=session_mgr,
        context_manager=ctx_mgr,
        max_turns=10,
    )

    result = loop.run("Create a calculator module in calc.py")

    assert "calculator module has been created and verified" in result
    calc_file = temp_project_dir / "calc.py"
    assert calc_file.exists()
    assert "return a + b" in calc_file.read_text(encoding="utf-8")

    # Verify session history recorded every step
    history = session_mgr.get_history()
    assert len(history) == 6  # user, assistant, tool, assistant, tool, assistant
