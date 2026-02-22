"""EvoScientist Agent graph construction.

This module creates and exports the compiled agent graph.
Usage:
    from EvoScientist import agent

    # Notebook / programmatic usage
    for state in agent.stream(
        {"messages": [HumanMessage(content="your question")]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        ...
"""

import json
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, CompositeBackend

from .backends import CustomSandboxBackend, MergedReadOnlyBackend
from .config import get_effective_config, apply_config_to_env
from .llm import get_chat_model
from .mcp import load_mcp_tools
from .middleware import create_memory_middleware, ToolErrorHandlerMiddleware
from .prompts import RESEARCHER_INSTRUCTIONS, get_system_prompt
from .utils import load_subagents
from .tools import tavily_search, think_tool, skill_manager
from . import paths as _paths_mod
from .paths import set_active_workspace, set_workspace_root

# =============================================================================
# Configuration
# =============================================================================

# Load configuration from file/env/defaults
_config = get_effective_config()
apply_config_to_env(_config)

# NOTE: We intentionally do NOT call set_workspace_root() at module level.
# The CLI (commands.py) calls set_workspace_root() *before* importing this
# module.  A module-level call here would overwrite the CLI's --workdir
# value with config.default_workdir, violating the priority chain
# (CLI args > config file).  Instead, config.default_workdir is applied
# as a fallback inside create_cli_agent() when no explicit workspace_dir
# is provided.

# Research limits (from config)
MAX_CONCURRENT = _config.max_concurrent
MAX_ITERATIONS = _config.max_iterations

# Workspace settings (defer dir creation to CLI; here we just resolve paths)
# Read from the paths module so values reflect any earlier set_workspace_root().
WORKSPACE_DIR = str(_paths_mod.WORKSPACE_ROOT)
set_active_workspace(WORKSPACE_DIR)
MEMORY_DIR = str(_paths_mod.MEMORY_DIR)  # Shared across sessions (not per-session)
SKILLS_DIR = str(Path(__file__).parent / "skills")
USER_SKILLS_DIR = str(_paths_mod.USER_SKILLS_DIR)
SUBAGENTS_CONFIG = Path(__file__).parent / "subagent.yaml"

# =============================================================================
# Initialization
# =============================================================================


# Generate system prompt with limits
SYSTEM_PROMPT = get_system_prompt(
    max_concurrent=MAX_CONCURRENT,
    max_iterations=MAX_ITERATIONS,
)

# Initialize chat model using the LLM module (respects config settings)
chat_model = get_chat_model(
    model=_config.model,
    provider=_config.provider,
)

# Initialize workspace backend
_workspace_backend = CustomSandboxBackend(
    root_dir=WORKSPACE_DIR,
    virtual_mode=True,
    timeout=300,
)

# Skills backend: merge user-installed (./skills/) and system (package) skills
_skills_backend = MergedReadOnlyBackend(
    primary_dir=USER_SKILLS_DIR,                        # user-installed, takes priority
    secondary_dir=SKILLS_DIR,                           # package built-in, fallback
)

# Memory backend: persistent filesystem for long-term memory (shared across sessions)
_memory_backend = FilesystemBackend(
    root_dir=MEMORY_DIR,
    virtual_mode=True,
)

# Composite backend: workspace as default, skills and memory mounted
backend = CompositeBackend(
    default=_workspace_backend,
    routes={
        "/skills/": _skills_backend,
        "/memory/": _memory_backend,
    },
)

tool_registry = {
    "think_tool": think_tool,
    "tavily_search": tavily_search,
}

# Base tools that every agent variant gets (before MCP)
BASE_TOOLS = [think_tool, skill_manager]

# Cache MCP tools by the effective config signature to avoid reconnecting
# to MCP servers on every `/new` when config is unchanged.
_MCP_TOOLS_CACHE_KEY: str | None = None
_MCP_TOOLS_CACHE_VALUE: dict[str, list] | None = None


def _mcp_config_signature() -> str:
    """Return a stable signature for the effective MCP config."""
    from .mcp.client import load_mcp_config

    cfg = load_mcp_config()
    if not cfg:
        return ""
    try:
        return json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    except TypeError:
        # Fallback for non-JSON-serializable values (should be rare)
        return repr(cfg)


def _load_mcp_tools_cached() -> dict[str, list]:
    """Load MCP tools with config-aware caching."""
    global _MCP_TOOLS_CACHE_KEY, _MCP_TOOLS_CACHE_VALUE

    cfg_key = _mcp_config_signature()
    if not cfg_key:
        _MCP_TOOLS_CACHE_KEY = ""
        _MCP_TOOLS_CACHE_VALUE = {}
        return {}

    if _MCP_TOOLS_CACHE_KEY == cfg_key and _MCP_TOOLS_CACHE_VALUE is not None:
        return {k: list(v) for k, v in _MCP_TOOLS_CACHE_VALUE.items()}

    loaded = load_mcp_tools()
    _MCP_TOOLS_CACHE_KEY = cfg_key
    _MCP_TOOLS_CACHE_VALUE = {k: list(v) for k, v in loaded.items()}
    return {k: list(v) for k, v in loaded.items()}


def _inject_subagent_middleware(subs: list[dict]) -> None:
    """Ensure every subagent gets ToolErrorHandlerMiddleware.

    Without this, subagent tool errors are caught by LangGraph's default
    ToolNode handler which produces terse messages without tracebacks or
    retry guidance — reducing the subagent's ability to self-recover.
    """
    for sa in subs:
        sa.setdefault("middleware", []).append(ToolErrorHandlerMiddleware())


def _build_base_kwargs(base_backend, base_middleware):
    """Build agent kwargs *without* MCP (fast, no subprocess spawning)."""
    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=tool_registry,
        prompt_refs=_build_prompt_refs(),
    )
    _inject_subagent_middleware(subs)
    return dict(
        name="EvoScientist",
        model=chat_model,
        tools=list(BASE_TOOLS),
        backend=base_backend,
        subagents=subs,
        middleware=base_middleware,
        system_prompt=SYSTEM_PROMPT,
        skills=["/skills/"],
    )


def load_mcp_and_build_kwargs(base_backend, base_middleware):
    """Load MCP tools (cached by config) and build agent kwargs.

    Re-connects to MCP servers only when the effective MCP config changes.
    Falls back to base kwargs if no MCP configured.
    """
    mcp_by_agent = _load_mcp_tools_cached()
    if not mcp_by_agent:
        return _build_base_kwargs(base_backend, base_middleware)

    # Fresh tool registry — start from base tools + MCP tools
    registry = dict(tool_registry)
    for tools in mcp_by_agent.values():
        for t in tools:
            registry[t.name] = t

    mcp_main = mcp_by_agent.pop("main", [])

    subs = load_subagents(
        SUBAGENTS_CONFIG,
        tool_registry=registry,
        prompt_refs=_build_prompt_refs(),
    )

    _inject_subagent_middleware(subs)

    # Inject MCP tools into subagents by name
    for sa in subs:
        if sa_tools := mcp_by_agent.get(sa["name"], []):
            sa.setdefault("tools", []).extend(sa_tools)

    return dict(
        name="EvoScientist",
        model=chat_model,
        tools=BASE_TOOLS + mcp_main,
        backend=base_backend,
        subagents=subs,
        middleware=base_middleware,
        system_prompt=SYSTEM_PROMPT,
        skills=["/skills/"],
    )


def _build_prompt_refs() -> dict:
    """Build prompt references with the current date (not frozen at import)."""
    return {
        "RESEARCHER_INSTRUCTIONS": RESEARCHER_INSTRUCTIONS.format(
            date=datetime.now().strftime("%Y-%m-%d"),
        ),
    }

base_middleware = [
    ToolErrorHandlerMiddleware(),
    create_memory_middleware(MEMORY_DIR, extraction_model=chat_model),
]

# Default agent (no checkpointer) — used by langgraph dev / LangSmith / notebooks.
# Lazily constructed on first access so MCP tools are included without
# spawning subprocesses at import time.
_EvoScientist_agent = None


def _get_default_agent():
    """Build the default agent (with MCP, no checkpointer) on first access."""
    global _EvoScientist_agent
    if _EvoScientist_agent is None:
        kwargs = load_mcp_and_build_kwargs(backend, base_middleware)
        _EvoScientist_agent = create_deep_agent(**kwargs).with_config(
            {"recursion_limit": 500}
        )
    return _EvoScientist_agent


def __getattr__(name: str):
    if name == "EvoScientist_agent":
        return _get_default_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_cli_agent(workspace_dir: str | None = None, checkpointer=None):
    """Create agent with checkpointer for CLI multi-turn support.

    A fresh backend is constructed on every call using the current
    ``paths.WORKSPACE_ROOT`` (or the explicit *workspace_dir*), so
    runtime ``set_workspace_root()`` changes are always respected.

    Args:
        workspace_dir: Per-session workspace directory. If ``None``,
            defaults to the current ``paths.WORKSPACE_ROOT``.
        checkpointer: Optional LangGraph checkpointer. If ``None``,
            falls back to ``InMemorySaver`` (non-persistent).
    """
    import os as _os
    from . import paths as _paths

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore[import-untyped]
        checkpointer = InMemorySaver()

    # When no explicit workspace_dir is provided, apply config.default_workdir
    # as a fallback.  This covers direct callers (notebooks, iMessage server)
    # that never call set_workspace_root() themselves.  CLI callers always
    # pass workspace_dir explicitly, so their --workdir is never overwritten.
    if workspace_dir is None:
        if _config.default_workdir:
            set_workspace_root(
                _os.path.abspath(_os.path.expanduser(_config.default_workdir))
            )
        workspace_dir = str(_paths.WORKSPACE_ROOT)

    # Read paths dynamically so runtime set_workspace_root() changes are picked up
    _mem_dir = str(_paths.MEMORY_DIR)
    _usr_skills_dir = str(_paths.USER_SKILLS_DIR)

    # Always construct fresh backends from current paths (avoids stale
    # module-level backend when workspace root changed at runtime).
    set_active_workspace(workspace_dir)
    ws_backend = CustomSandboxBackend(
        root_dir=workspace_dir,
        virtual_mode=True,
        timeout=300,
    )
    sk_backend = MergedReadOnlyBackend(
        primary_dir=_usr_skills_dir,
        secondary_dir=SKILLS_DIR,
    )
    # Memory always uses SHARED directory (not per-session) for cross-session persistence
    mem_backend = FilesystemBackend(
        root_dir=_mem_dir,
        virtual_mode=True,
    )
    be = CompositeBackend(
        default=ws_backend,
        routes={
            "/skills/": sk_backend,
            "/memory/": mem_backend,
        },
    )

    mw = [
        ToolErrorHandlerMiddleware(),
        create_memory_middleware(_mem_dir, extraction_model=chat_model),
    ]

    # Re-load MCP tools from current config (picks up /mcp add changes)
    kwargs = load_mcp_and_build_kwargs(be, mw)

    return create_deep_agent(
        **kwargs,
        checkpointer=checkpointer,
    ).with_config({"recursion_limit": 500})
