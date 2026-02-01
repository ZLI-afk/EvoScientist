"""
EvoScientist Agent CLI

Command-line interface with streaming output for the EvoScientist research agent.

Features:
- Thinking panel (blue) - shows model reasoning
- Tool calls with status indicators (green/yellow/red dots)
- Tool results in tree format with folding
- Response panel (green) - shows final response
- Thread ID support for multi-turn conversations
- Interactive mode with prompt_toolkit
"""

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime
from typing import Any, AsyncIterator

from dotenv import load_dotenv  # type: ignore[import-untyped]
from prompt_toolkit import PromptSession  # type: ignore[import-untyped]
from prompt_toolkit.history import FileHistory  # type: ignore[import-untyped]
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory  # type: ignore[import-untyped]
from prompt_toolkit.formatted_text import HTML  # type: ignore[import-untyped]
from rich.console import Console, Group  # type: ignore[import-untyped]
from rich.panel import Panel  # type: ignore[import-untyped]
from rich.markdown import Markdown  # type: ignore[import-untyped]
from rich.live import Live  # type: ignore[import-untyped]
from rich.text import Text  # type: ignore[import-untyped]
from rich.spinner import Spinner  # type: ignore[import-untyped]
from langchain_core.messages import AIMessage, AIMessageChunk  # type: ignore[import-untyped]

from .stream import (
    StreamEventEmitter,
    ToolCallTracker,
    ToolResultFormatter,
    DisplayLimits,
    ToolStatus,
    format_tool_compact,
    is_success,
)

load_dotenv(override=True)

console = Console(
    legacy_windows=(sys.platform == 'win32'),
    no_color=os.getenv('NO_COLOR') is not None,
)

formatter = ToolResultFormatter()


# =============================================================================
# Stream event generator
# =============================================================================

async def stream_agent_events(agent: Any, message: str, thread_id: str) -> AsyncIterator[dict]:
    """Stream events from the agent graph using async iteration.

    Uses agent.astream() with subgraphs=True to see sub-agent activity.

    Args:
        agent: Compiled state graph from create_deep_agent()
        message: User message
        thread_id: Thread ID for conversation persistence

    Yields:
        Event dicts: thinking, text, tool_call, tool_result,
                     subagent_start, subagent_tool_call, subagent_tool_result, subagent_end,
                     done, error
    """
    config = {"configurable": {"thread_id": thread_id}}
    emitter = StreamEventEmitter()
    main_tracker = ToolCallTracker()
    full_response = ""

    # Track sub-agent names
    _key_to_name: dict[str, str] = {}     # subagent_key → display name (cache)
    _announced_names: list[str] = []       # ordered queue of announced task names
    _assigned_names: set[str] = set()      # names already assigned to a namespace
    _announced_task_ids: list[str] = []     # ordered task tool_call_ids
    _task_id_to_name: dict[str, str] = {}  # tool_call_id → sub-agent name
    _subagent_trackers: dict[str, ToolCallTracker] = {}  # namespace_key → tracker

    def _register_task_tool_call(tc_data: dict) -> str | None:
        """Register or update a task tool call, return subagent name if started/updated."""
        tool_id = tc_data.get("id", "")
        if not tool_id:
            return None
        args = tc_data.get("args", {}) or {}
        desc = str(args.get("description", "")).strip()
        sa_name = str(args.get("subagent_type", "")).strip()
        if not sa_name:
            # Fallback to description snippet (may be empty during streaming)
            sa_name = desc[:30] + "..." if len(desc) > 30 else desc
        if not sa_name:
            sa_name = "sub-agent"

        if tool_id not in _announced_task_ids:
            _announced_task_ids.append(tool_id)
            _announced_names.append(sa_name)
            _task_id_to_name[tool_id] = sa_name
            return sa_name

        # Update mapping if we learned a better name later
        current = _task_id_to_name.get(tool_id, "sub-agent")
        if sa_name != "sub-agent" and current != sa_name:
            _task_id_to_name[tool_id] = sa_name
            try:
                idx = _announced_task_ids.index(tool_id)
                if idx < len(_announced_names):
                    _announced_names[idx] = sa_name
            except ValueError:
                pass
            return sa_name
        return None

    def _extract_task_id(namespace: tuple) -> tuple[str | None, str | None]:
        """Extract task tool_call_id from namespace if present.

        Returns (task_id, task_ns_element) or (None, None).
        """
        for part in namespace:
            part_str = str(part)
            if "task:" in part_str:
                tail = part_str.split("task:", 1)[1]
                task_id = tail.split(":", 1)[0] if tail else ""
                if task_id:
                    return task_id, part_str
        return None, None

    def _next_announced_name() -> str | None:
        """Get next announced name that hasn't been assigned yet."""
        for announced in _announced_names:
            if announced not in _assigned_names:
                _assigned_names.add(announced)
                return announced
        return None

    def _find_task_id_from_metadata(metadata: dict | None) -> str | None:
        """Try to find a task tool_call_id in metadata."""
        if not metadata:
            return None
        candidates = (
            "tool_call_id",
            "task_id",
            "parent_run_id",
            "root_run_id",
            "run_id",
        )
        for key in candidates:
            val = metadata.get(key)
            if val and val in _task_id_to_name:
                return val
        return None

    def _get_subagent_key(namespace: tuple, metadata: dict | None) -> str | None:
        """Stable key for tracker/mapping per sub-agent namespace."""
        if not namespace:
            return None
        task_id, task_ns = _extract_task_id(namespace)
        if task_ns:
            return task_ns
        meta_task_id = _find_task_id_from_metadata(metadata)
        if meta_task_id:
            return f"task:{meta_task_id}"
        if metadata:
            for key in ("parent_run_id", "root_run_id", "run_id", "graph_id", "node_id"):
                val = metadata.get(key)
                if val:
                    return f"{key}:{val}"
        return str(namespace)

    def _get_subagent_name(namespace: tuple, metadata: dict | None) -> str | None:
        """Resolve sub-agent name from namespace, or None if main agent.

        Priority:
        0) metadata["lc_agent_name"] — most reliable, set by DeepAgents framework.
        1) Match task_id embedded in namespace to announced tool_call_id.
        2) Use cached key mapping (only real names, never "sub-agent").
        3) Queue-based: assign next announced name to this key.
        4) Fallback: return "sub-agent" WITHOUT caching.
        """
        if not namespace:
            return None

        key = _get_subagent_key(namespace, metadata) or str(namespace)

        # 0) lc_agent_name from metadata — the REAL sub-agent name
        #    set by the DeepAgents framework on every namespace event.
        if metadata:
            lc_name = metadata.get("lc_agent_name", "")
            if isinstance(lc_name, str):
                lc_name = lc_name.strip()
            # Filter out generic/framework names
            if lc_name and lc_name not in (
                "sub-agent", "agent", "tools", "EvoScientist",
                "LangGraph", "",
            ):
                _key_to_name[key] = lc_name
                return lc_name

        # 1) Resolve by task_id if present in namespace
        task_id, _task_ns = _extract_task_id(namespace)
        if task_id and task_id in _task_id_to_name:
            name = _task_id_to_name[task_id]
            if name and name != "sub-agent":
                _assigned_names.add(name)
                _key_to_name[key] = name
                return name

        meta_task_id = _find_task_id_from_metadata(metadata)
        if meta_task_id and meta_task_id in _task_id_to_name:
            name = _task_id_to_name[meta_task_id]
            if name and name != "sub-agent":
                _assigned_names.add(name)
                _key_to_name[key] = name
                return name

        # 2) Cached real name for this key (skip if it's "sub-agent")
        cached = _key_to_name.get(key)
        if cached and cached != "sub-agent":
            return cached

        # 3) Assign next announced name from queue (skip "sub-agent" entries)
        for announced in _announced_names:
            if announced not in _assigned_names and announced != "sub-agent":
                _assigned_names.add(announced)
                _key_to_name[key] = announced
                return announced

        # 4) No real names available yet — return generic WITHOUT caching
        return "sub-agent"

    try:
        async for chunk in agent.astream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="messages",
            subgraphs=True,
        ):
            # With subgraphs=True, event is (namespace, (message, metadata))
            namespace: tuple = ()
            data: Any = chunk

            if isinstance(chunk, tuple) and len(chunk) >= 2:
                first = chunk[0]
                if isinstance(first, tuple):
                    # (namespace_tuple, (message, metadata))
                    namespace = first
                    data = chunk[1]
                else:
                    # (message, metadata) — no namespace
                    data = chunk

            # Unpack message + metadata from data
            msg: Any
            metadata: dict = {}
            if isinstance(data, tuple) and len(data) >= 2:
                msg = data[0]
                metadata = data[1] or {}
            else:
                msg = data

            subagent = _get_subagent_name(namespace, metadata)
            subagent_tracker = None
            if subagent:
                tracker_key = _get_subagent_key(namespace, metadata) or str(namespace)
                subagent_tracker = _subagent_trackers.setdefault(tracker_key, ToolCallTracker())

            # Process AIMessageChunk / AIMessage
            if isinstance(msg, (AIMessageChunk, AIMessage)):
                if subagent:
                    # Sub-agent content — emit sub-agent events
                    for ev in _process_chunk_content(msg, emitter, subagent_tracker):
                        if ev.type == "tool_call":
                            yield emitter.subagent_tool_call(
                                subagent, ev.data["name"], ev.data["args"], ev.data.get("id", "")
                            ).data
                        # Skip text/thinking from sub-agents (too noisy)

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name", "")
                            args = tc.get("args", {})
                            tool_id = tc.get("id", "")
                            # Skip empty-name chunks (incomplete streaming fragments)
                            if not name and not tool_id:
                                continue
                            yield emitter.subagent_tool_call(
                                subagent, name, args if isinstance(args, dict) else {}, tool_id
                            ).data
                else:
                    # Main agent content
                    for ev in _process_chunk_content(msg, emitter, main_tracker):
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        yield ev.data

                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for ev in _process_tool_calls(msg.tool_calls, emitter, main_tracker):
                            yield ev.data
                            # Detect task tool calls → announce sub-agent
                            tc_data = ev.data
                            if tc_data.get("name") == "task":
                                started_name = _register_task_tool_call(tc_data)
                                if started_name:
                                    desc = str(tc_data.get("args", {}).get("description", "")).strip()
                                    yield emitter.subagent_start(started_name, desc).data

            # Process ToolMessage (tool execution result)
            elif hasattr(msg, "type") and msg.type == "tool":
                if subagent:
                    if subagent_tracker:
                        subagent_tracker.finalize_all()
                        for info in subagent_tracker.emit_all_pending():
                            yield emitter.subagent_tool_call(
                                subagent,
                                info.name,
                                info.args,
                                info.id,
                            ).data
                    name = getattr(msg, "name", "unknown")
                    raw_content = str(getattr(msg, "content", ""))
                    content = raw_content[:DisplayLimits.TOOL_RESULT_MAX]
                    success = is_success(content)
                    yield emitter.subagent_tool_result(subagent, name, content, success).data
                else:
                    for ev in _process_tool_result(msg, emitter, main_tracker):
                        yield ev.data
                        # Tool result can re-emit tool_call with full args; update task mapping
                        if ev.type == "tool_call" and ev.data.get("name") == "task":
                            started_name = _register_task_tool_call(ev.data)
                            if started_name:
                                desc = str(ev.data.get("args", {}).get("description", "")).strip()
                                yield emitter.subagent_start(started_name, desc).data
                    # Check if this is a task result → sub-agent ended
                    name = getattr(msg, "name", "")
                    if name == "task":
                        tool_call_id = getattr(msg, "tool_call_id", "")
                        # Find the sub-agent name via tool_call_id map
                        sa_name = _task_id_to_name.get(tool_call_id, "sub-agent")
                        yield emitter.subagent_end(sa_name).data

    except Exception as e:
        yield emitter.error(str(e)).data
        raise

    yield emitter.done(full_response).data


def _process_chunk_content(chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
    """Process content blocks from an AI message chunk."""
    content = chunk.content

    if isinstance(content, str):
        if content:
            yield emitter.text(content)
            return

    blocks = None
    if hasattr(chunk, "content_blocks"):
        try:
            blocks = chunk.content_blocks
        except Exception:
            blocks = None

    if blocks is None:
        if isinstance(content, dict):
            blocks = [content]
        elif isinstance(content, list):
            blocks = content
        else:
            return

    for raw_block in blocks:
        block = raw_block
        if not isinstance(block, dict):
            if hasattr(block, "model_dump"):
                block = block.model_dump()
            elif hasattr(block, "dict"):
                block = block.dict()
            else:
                continue

        block_type = block.get("type")

        if block_type in ("thinking", "reasoning"):
            thinking_text = block.get("thinking") or block.get("reasoning") or ""
            if thinking_text:
                yield emitter.thinking(thinking_text)

        elif block_type == "text":
            text = block.get("text") or block.get("content") or ""
            if text:
                yield emitter.text(text)

        elif block_type in ("tool_use", "tool_call"):
            tool_id = block.get("id", "")
            name = block.get("name", "")
            args = block.get("input") if block_type == "tool_use" else block.get("args")
            args_payload = args if isinstance(args, dict) else {}

            if tool_id:
                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

        elif block_type == "input_json_delta":
            partial_json = block.get("partial_json", "")
            if partial_json:
                tracker.append_json_delta(partial_json, block.get("index", 0))

        elif block_type == "tool_call_chunk":
            tool_id = block.get("id", "")
            name = block.get("name", "")
            if tool_id:
                tracker.update(tool_id, name=name)
            partial_args = block.get("args", "")
            if isinstance(partial_args, str) and partial_args:
                tracker.append_json_delta(partial_args, block.get("index", 0))


def _process_tool_calls(tool_calls: list, emitter: StreamEventEmitter, tracker: ToolCallTracker):
    """Process tool_calls from chunk.tool_calls attribute."""
    for tc in tool_calls:
        tool_id = tc.get("id", "")
        if tool_id:
            name = tc.get("name", "")
            args = tc.get("args", {})
            args_payload = args if isinstance(args, dict) else {}

            tracker.update(tool_id, name=name, args=args_payload)
            if tracker.is_ready(tool_id):
                tracker.mark_emitted(tool_id)
                yield emitter.tool_call(name, args_payload, tool_id)


def _process_tool_result(chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
    """Process a ToolMessage result."""
    tracker.finalize_all()

    # Re-emit all tool calls with complete args
    for info in tracker.get_all():
        yield emitter.tool_call(info.name, info.args, info.id)

    name = getattr(chunk, "name", "unknown")
    raw_content = str(getattr(chunk, "content", ""))
    content = raw_content[:DisplayLimits.TOOL_RESULT_MAX]
    if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
        content += "\n... (truncated)"

    success = is_success(content)
    yield emitter.tool_result(name, content, success)


# =============================================================================
# Stream state
# =============================================================================

class SubAgentState:
    """Tracks a single sub-agent's activity."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tool_calls: list[dict] = []
        self.tool_results: list[dict] = []
        self._result_map: dict[str, dict] = {}  # tool_call_id → result
        self.is_active = True

    def add_tool_call(self, name: str, args: dict, tool_id: str = ""):
        # Skip empty-name calls without an id (incomplete streaming chunks)
        if not name and not tool_id:
            return
        tc_data = {"id": tool_id, "name": name, "args": args}
        if tool_id:
            for i, tc in enumerate(self.tool_calls):
                if tc.get("id") == tool_id:
                    # Merge: keep the non-empty name/args
                    if name:
                        self.tool_calls[i]["name"] = name
                    if args:
                        self.tool_calls[i]["args"] = args
                    return
        # Skip if name is empty and we can't deduplicate by id
        if not name:
            return
        self.tool_calls.append(tc_data)

    def add_tool_result(self, name: str, content: str, success: bool = True):
        result = {"name": name, "content": content, "success": success}
        self.tool_results.append(result)
        # Try to match result to the first unmatched tool call with same name
        for tc in self.tool_calls:
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            if tc_id and tc_id not in self._result_map and tc_name == name:
                self._result_map[tc_id] = result
                return
        # Fallback: match first unmatched tool call
        for tc in self.tool_calls:
            tc_id = tc.get("id", "")
            if tc_id and tc_id not in self._result_map:
                self._result_map[tc_id] = result
                return

    def get_result_for(self, tc: dict) -> dict | None:
        """Get matched result for a tool call."""
        tc_id = tc.get("id", "")
        if tc_id:
            return self._result_map.get(tc_id)
        # Fallback: index-based matching
        try:
            idx = self.tool_calls.index(tc)
            if idx < len(self.tool_results):
                return self.tool_results[idx]
        except ValueError:
            pass
        return None


class StreamState:
    """Accumulates stream state for display updates."""

    def __init__(self):
        self.thinking_text = ""
        self.response_text = ""
        self.tool_calls = []
        self.tool_results = []
        self.is_thinking = False
        self.is_responding = False
        self.is_processing = False
        # Sub-agent tracking
        self.subagents: list[SubAgentState] = []
        self._subagent_map: dict[str, SubAgentState] = {}  # name → state
        # Todo list tracking
        self.todo_items: list[dict] = []
        # Latest text segment (reset on each tool_call)
        self.latest_text = ""

    def _get_or_create_subagent(self, name: str, description: str = "") -> SubAgentState:
        if name not in self._subagent_map:
            # Case 1: real name arrives, "sub-agent" entry exists → rename it
            if name != "sub-agent" and "sub-agent" in self._subagent_map:
                old_sa = self._subagent_map.pop("sub-agent")
                old_sa.name = name
                if description:
                    old_sa.description = description
                self._subagent_map[name] = old_sa
                return old_sa
            # Case 2: "sub-agent" arrives but a pre-registered real-name entry
            #         exists with no tool calls → merge into it
            if name == "sub-agent":
                active_named = [
                    sa for sa in self.subagents
                    if sa.is_active and sa.name != "sub-agent"
                ]
                if len(active_named) == 1 and not active_named[0].tool_calls:
                    self._subagent_map[name] = active_named[0]
                    return active_named[0]
            sa = SubAgentState(name, description)
            self.subagents.append(sa)
            self._subagent_map[name] = sa
        else:
            existing = self._subagent_map[name]
            if description and not existing.description:
                existing.description = description
            # If this entry was created as "sub-agent" placeholder and the
            # actual name is different, update.
            if name != "sub-agent" and existing.name == "sub-agent":
                existing.name = name
        return self._subagent_map[name]

    def _resolve_subagent_name(self, name: str) -> str:
        """Resolve "sub-agent" to the single active named sub-agent when possible."""
        if name != "sub-agent":
            return name
        active_named = [
            sa.name for sa in self.subagents
            if sa.is_active and sa.name != "sub-agent"
        ]
        if len(active_named) == 1:
            return active_named[0]
        return name

    def handle_event(self, event: dict) -> str:
        """Process a single stream event, update internal state, return event type."""
        event_type: str = event.get("type", "")

        if event_type == "thinking":
            self.is_thinking = True
            self.is_responding = False
            self.is_processing = False
            self.thinking_text += event.get("content", "")

        elif event_type == "text":
            self.is_thinking = False
            self.is_responding = True
            self.is_processing = False
            text_content = event.get("content", "")
            self.response_text += text_content
            self.latest_text += text_content

        elif event_type == "tool_call":
            self.is_thinking = False
            self.is_responding = False
            self.is_processing = False
            self.latest_text = ""  # Reset — next text segment is a new message

            tool_id = event.get("id", "")
            tool_name = event.get("name", "unknown")
            tool_args = event.get("args", {})
            tc_data = {
                "id": tool_id,
                "name": tool_name,
                "args": tool_args,
            }

            if tool_id:
                updated = False
                for i, tc in enumerate(self.tool_calls):
                    if tc.get("id") == tool_id:
                        self.tool_calls[i] = tc_data
                        updated = True
                        break
                if not updated:
                    self.tool_calls.append(tc_data)
            else:
                self.tool_calls.append(tc_data)

            # Capture todo items from write_todos args (most reliable source)
            if tool_name == "write_todos":
                todos = tool_args.get("todos", [])
                if isinstance(todos, list) and todos:
                    self.todo_items = todos

        elif event_type == "tool_result":
            self.is_processing = True
            result_name = event.get("name", "unknown")
            result_content = event.get("content", "")
            self.tool_results.append({
                "name": result_name,
                "content": result_content,
            })
            # Update todo list from write_todos / read_todos results (fallback)
            if result_name in ("write_todos", "read_todos"):
                parsed = _parse_todo_items(result_content)
                if parsed:
                    self.todo_items = parsed

        elif event_type == "subagent_start":
            name = event.get("name", "sub-agent")
            desc = event.get("description", "")
            sa = self._get_or_create_subagent(name, desc)
            sa.is_active = True

        elif event_type == "subagent_tool_call":
            sa_name = self._resolve_subagent_name(event.get("subagent", "sub-agent"))
            sa = self._get_or_create_subagent(sa_name)
            sa.add_tool_call(
                event.get("name", "unknown"),
                event.get("args", {}),
                event.get("id", ""),
            )

        elif event_type == "subagent_tool_result":
            sa_name = self._resolve_subagent_name(event.get("subagent", "sub-agent"))
            sa = self._get_or_create_subagent(sa_name)
            sa.add_tool_result(
                event.get("name", "unknown"),
                event.get("content", ""),
                event.get("success", True),
            )

        elif event_type == "subagent_end":
            name = self._resolve_subagent_name(event.get("name", "sub-agent"))
            if name in self._subagent_map:
                self._subagent_map[name].is_active = False
            elif name == "sub-agent":
                # Couldn't resolve — deactivate the oldest active sub-agent
                for sa in self.subagents:
                    if sa.is_active:
                        sa.is_active = False
                        break

        elif event_type == "done":
            self.is_processing = False
            if not self.response_text:
                self.response_text = event.get("response", "")

        elif event_type == "error":
            self.is_processing = False
            self.is_thinking = False
            self.is_responding = False
            error_msg = event.get("message", "Unknown error")
            self.response_text += f"\n\n[Error] {error_msg}"

        return event_type

    def get_display_args(self) -> dict:
        """Get kwargs for create_streaming_display()."""
        return {
            "thinking_text": self.thinking_text,
            "response_text": self.response_text,
            "latest_text": self.latest_text,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "is_thinking": self.is_thinking,
            "is_responding": self.is_responding,
            "is_processing": self.is_processing,
            "subagents": self.subagents,
            "todo_items": self.todo_items,
        }


# =============================================================================
# Display functions
# =============================================================================

def _parse_todo_items(content: str) -> list[dict] | None:
    """Parse todo items from write_todos output.

    Attempts to extract a list of dicts with 'status' and 'content' keys
    from the tool result string. Returns None if parsing fails.

    Handles formats like:
      - Raw JSON/Python list: [{"content": "...", "status": "..."}]
      - Prefixed: "Updated todo list to [{'content': '...', ...}]"
    """
    import ast
    import json

    content = content.strip()

    def _try_parse(text: str) -> list[dict] | None:
        """Try JSON then Python literal parsing."""
        text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            data = ast.literal_eval(text)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data
        except (ValueError, SyntaxError):
            pass
        return None

    # Try the full content directly
    result = _try_parse(content)
    if result:
        return result

    # Extract embedded [...] from content (e.g. "Updated todo list to [{...}]")
    bracket_start = content.find("[")
    if bracket_start != -1:
        bracket_end = content.rfind("]")
        if bracket_end > bracket_start:
            embedded = content[bracket_start:bracket_end + 1]
            result = _try_parse(embedded)
            if result:
                return result

    # Try line-by-line scan
    for line in content.split("\n"):
        line = line.strip()
        if "[" in line:
            start = line.find("[")
            end = line.rfind("]")
            if end > start:
                result = _try_parse(line[start:end + 1])
                if result:
                    return result

    return None


def _build_todo_stats(items: list[dict]) -> str:
    """Build stats string like '2 active | 1 pending | 3 done'."""
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "todo")).lower()
        # Normalize status names
        if status in ("done", "completed", "complete"):
            status = "done"
        elif status in ("active", "in_progress", "in-progress", "working"):
            status = "active"
        else:
            status = "pending"
        counts[status] = counts.get(status, 0) + 1

    parts = []
    for key in ("active", "pending", "done"):
        if counts.get(key, 0) > 0:
            parts.append(f"{counts[key]} {key}")
    return " | ".join(parts) if parts else f"{len(items)} items"


def _format_single_todo(item: dict) -> Text:
    """Format a single todo item with status symbol."""
    status = str(item.get("status", "todo")).lower()
    content_text = str(item.get("content", item.get("task", item.get("title", ""))))

    if status in ("done", "completed", "complete"):
        symbol = "\u2713"
        label = "done  "
        style = "green dim"
    elif status in ("active", "in_progress", "in-progress", "working"):
        symbol = "\u25cf"
        label = "active"
        style = "yellow"
    else:
        symbol = "\u25cb"
        label = "todo  "
        style = "dim"

    line = Text()
    line.append(f"    {symbol} ", style=style)
    line.append(label, style=style)
    line.append(" ", style="dim")
    # Truncate long content
    if len(content_text) > 60:
        content_text = content_text[:57] + "..."
    line.append(content_text, style=style)
    return line


def format_tool_result_compact(_name: str, content: str, max_lines: int = 5) -> list:
    """Format tool result as tree output.

    Special handling for write_todos: shows formatted checklist with status symbols.
    """
    elements = []

    if not content.strip():
        elements.append(Text("  \u2514 (empty)", style="dim"))
        return elements

    # Special handling for write_todos
    if _name == "write_todos":
        items = _parse_todo_items(content)
        if items:
            stats = _build_todo_stats(items)
            stats_line = Text()
            stats_line.append("  \u2514 ", style="dim")
            stats_line.append(stats, style="dim")
            elements.append(stats_line)
            elements.append(Text("", style="dim"))  # blank line

            max_preview = 4
            for item in items[:max_preview]:
                elements.append(_format_single_todo(item))

            remaining = len(items) - max_preview
            if remaining > 0:
                elements.append(Text(f"    ... {remaining} more", style="dim italic"))

            return elements

    lines = content.strip().split("\n")
    total_lines = len(lines)

    display_lines = lines[:max_lines]
    for i, line in enumerate(display_lines):
        prefix = "\u2514" if i == 0 else " "
        if len(line) > 80:
            line = line[:77] + "..."
        style = "dim" if is_success(content) else "red dim"
        elements.append(Text(f"  {prefix} {line}", style=style))

    remaining = total_lines - max_lines
    if remaining > 0:
        elements.append(Text(f"    ... +{remaining} lines", style="dim italic"))

    return elements


def _render_tool_call_line(tc: dict, tr: dict | None) -> Text:
    """Render a single tool call line with status indicator."""
    is_task = tc.get('name', '').lower() == 'task'

    if tr is not None:
        content = tr.get('content', '')
        if is_success(content):
            style = "bold green"
            indicator = "\u2713" if is_task else ToolStatus.SUCCESS.value
        else:
            style = "bold red"
            indicator = "\u2717" if is_task else ToolStatus.ERROR.value
    else:
        style = "bold yellow" if not is_task else "bold cyan"
        indicator = "\u25b6" if is_task else ToolStatus.RUNNING.value

    tool_compact = format_tool_compact(tc['name'], tc.get('args'))
    tool_text = Text()
    tool_text.append(f"{indicator} ", style=style)
    tool_text.append(tool_compact, style=style)
    return tool_text


def _render_subagent_section(sa: 'SubAgentState', compact: bool = False) -> list:
    """Render a sub-agent's activity as a bordered section.

    Args:
        sa: Sub-agent state to render
        compact: If True, render minimal 1-line summary (completed sub-agents)

    Header uses "Cooking with {name}" style matching task tool format.
    Active sub-agents show bordered tool list; completed ones collapse to 1 line.
    """
    elements = []
    BORDER = "dim cyan" if sa.is_active else "dim"

    # Filter out tool calls with empty names
    valid_calls = [tc for tc in sa.tool_calls if tc.get("name")]

    # Split into completed and pending
    completed = []
    pending = []
    for tc in valid_calls:
        tr = sa.get_result_for(tc)
        if tr is not None:
            completed.append((tc, tr))
        else:
            pending.append(tc)

    succeeded = sum(1 for _, tr in completed if tr.get("success", True))
    failed = len(completed) - succeeded

    # Build display name
    display_name = f"Cooking with {sa.name}"
    if sa.description:
        desc = sa.description[:50] + "..." if len(sa.description) > 50 else sa.description
        display_name += f" \u2014 {desc}"

    # --- Compact mode: 1-line summary for completed sub-agents ---
    if compact:
        line = Text()
        if not sa.is_active:
            line.append("\u2713 ", style="green")
            line.append(display_name, style="green dim")
            total = len(valid_calls)
            line.append(f" ({total} tools)", style="dim")
        else:
            line.append("\u25b6 ", style="cyan")
            line.append(display_name, style="bold cyan")
        elements.append(line)
        return elements

    # --- Full mode: bordered section for Live streaming ---

    # Header
    header = Text()
    header.append("\u250c ", style=BORDER)
    if sa.is_active:
        header.append(f"\u25b6 {display_name}", style="bold cyan")
    else:
        header.append(f"\u2713 {display_name}", style="bold green")
    elements.append(header)

    # Show every tool call with its status
    for tc, tr in completed:
        tc_line = Text("\u2502 ", style=BORDER)
        tc_name = format_tool_compact(tc["name"], tc.get("args"))
        if tr.get("success", True):
            tc_line.append(f"\u2713 {tc_name}", style="green")
        else:
            tc_line.append(f"\u2717 {tc_name}", style="red")
            content = tr.get("content", "")
            first_line = content.strip().split("\n")[0][:70]
            if first_line:
                err_line = Text("\u2502   ", style=BORDER)
                err_line.append(f"\u2514 {first_line}", style="red dim")
                elements.append(tc_line)
                elements.append(err_line)
                continue
        elements.append(tc_line)

    # Pending/running tools
    for tc in pending:
        tc_line = Text("\u2502 ", style=BORDER)
        tc_name = format_tool_compact(tc["name"], tc.get("args"))
        tc_line.append(f"\u25cf {tc_name}", style="bold yellow")
        elements.append(tc_line)
        spinner_line = Text("\u2502   ", style=BORDER)
        spinner_line.append("\u21bb running...", style="yellow dim")
        elements.append(spinner_line)

    # Footer
    if not sa.is_active:
        total = len(valid_calls)
        footer = Text(f"\u2514 done ({total} tools)", style="dim green")
        elements.append(footer)
    elif valid_calls:
        footer = Text("\u2514 running...", style="dim cyan")
        elements.append(footer)

    return elements


def _render_todo_panel(todo_items: list[dict]) -> Panel:
    """Render a bordered Task List panel from todo items.

    Matches the style: cyan border, status icons per item.
    """
    lines = Text()
    for i, item in enumerate(todo_items):
        if i > 0:
            lines.append("\n")
        status = str(item.get("status", "todo")).lower()
        content_text = str(item.get("content", item.get("task", item.get("title", ""))))

        if status in ("done", "completed", "complete"):
            symbol = "\u2713"  # ✓
            style = "green dim"
        elif status in ("active", "in_progress", "in-progress", "working"):
            symbol = "\u23f3"  # ⏳
            style = "yellow"
        else:
            symbol = "\u25a1"  # □
            style = "dim"

        lines.append(f"{symbol} ", style=style)
        lines.append(content_text, style=style)

    return Panel(
        lines,
        title="Task List",
        title_align="center",
        border_style="cyan",
        padding=(0, 1),
    )


def create_streaming_display(
    thinking_text: str = "",
    response_text: str = "",
    latest_text: str = "",
    tool_calls: list | None = None,
    tool_results: list | None = None,
    is_thinking: bool = False,
    is_responding: bool = False,
    is_waiting: bool = False,
    is_processing: bool = False,
    show_thinking: bool = True,
    subagents: list | None = None,
    todo_items: list | None = None,
) -> Any:
    """Create Rich display layout for streaming output.

    Returns:
        Rich Group for Live display
    """
    elements = []
    tool_calls = tool_calls or []
    tool_results = tool_results or []
    subagents = subagents or []

    # Initial waiting state
    if is_waiting and not thinking_text and not response_text and not tool_calls:
        spinner = Spinner("dots", text=" Thinking...", style="cyan")
        elements.append(spinner)
        return Group(*elements)

    # Thinking panel
    if show_thinking and thinking_text:
        thinking_title = "Thinking"
        if is_thinking:
            thinking_title += " ..."
        display_thinking = thinking_text
        if len(display_thinking) > DisplayLimits.THINKING_STREAM:
            display_thinking = "..." + display_thinking[-DisplayLimits.THINKING_STREAM:]
        elements.append(Panel(
            Text(display_thinking, style="dim"),
            title=thinking_title,
            border_style="blue",
            padding=(0, 1),
        ))

    # Tool calls and results paired display
    # Collapse older completed tools to prevent overflow in Live mode
    # Task tool calls are ALWAYS visible (they represent sub-agent delegations)
    MAX_VISIBLE_TOOLS = 4
    MAX_VISIBLE_RUNNING = 3

    if tool_calls:
        # Split into categories
        completed_regular = []   # completed non-task tools
        task_tools = []          # task tools (always visible)
        running_regular = []     # running non-task tools

        for i, tc in enumerate(tool_calls):
            has_result = i < len(tool_results)
            tr = tool_results[i] if has_result else None
            is_task = tc.get('name') == 'task'

            if is_task:
                # Skip task calls with empty args (still streaming)
                if tc.get('args'):
                    task_tools.append((tc, tr))
            elif has_result:
                completed_regular.append((tc, tr))
            else:
                running_regular.append((tc, None))

        # --- Completed regular tools (collapsible) ---
        slots = max(0, MAX_VISIBLE_TOOLS - len(running_regular))
        hidden = completed_regular[:-slots] if slots and len(completed_regular) > slots else (completed_regular if not slots else [])
        visible = completed_regular[-slots:] if slots else []

        if hidden:
            ok = sum(1 for _, tr in hidden if is_success(tr.get('content', '')))
            fail = len(hidden) - ok
            summary = Text()
            summary.append(f"\u2713 {ok} completed", style="dim green")
            if fail > 0:
                summary.append(f" | {fail} failed", style="dim red")
            elements.append(summary)

        for tc, tr in visible:
            elements.append(_render_tool_call_line(tc, tr))
            content = tr.get('content', '') if tr else ''
            if tr and not is_success(content):
                result_elements = format_tool_result_compact(
                    tr['name'], content, max_lines=5,
                )
                elements.extend(result_elements)

        # --- Running regular tools (limit visible) ---
        hidden_running = len(running_regular) - MAX_VISIBLE_RUNNING
        if hidden_running > 0:
            summary = Text()
            summary.append(f"\u25cf {hidden_running} more running...", style="dim yellow")
            elements.append(summary)
            running_regular = running_regular[-MAX_VISIBLE_RUNNING:]

        for tc, tr in running_regular:
            elements.append(_render_tool_call_line(tc, tr))
            spinner = Spinner("dots", text=" Running...", style="yellow")
            elements.append(spinner)

        # Task tool calls are rendered as part of sub-agent sections below

    # Response text handling
    has_pending_tools = len(tool_calls) > len(tool_results)
    any_active_subagent = any(sa.is_active for sa in subagents)
    has_used_tools = len(tool_calls) > 0
    all_done = not has_pending_tools and not any_active_subagent and not is_processing

    # Intermediate narration (tools still running) — dim italic above Task List
    if latest_text and has_used_tools and not all_done:
        preview = latest_text.strip()
        if preview:
            last_line = preview.split("\n")[-1].strip()
            if last_line:
                if len(last_line) > 80:
                    last_line = last_line[:77] + "..."
                elements.append(Text(f"    {last_line}", style="dim italic"))

    # Task List panel (persistent, updates on write_todos / read_todos)
    todo_items = todo_items or []
    if todo_items:
        elements.append(Text(""))  # blank separator
        elements.append(_render_todo_panel(todo_items))

    # Sub-agent activity sections
    # Active: full bordered view; Completed: compact 1-line summary
    for sa in subagents:
        if sa.tool_calls or sa.is_active:
            elements.extend(_render_subagent_section(sa, compact=not sa.is_active))

    # Processing state after tool execution
    if is_processing and not is_thinking and not is_responding and not response_text:
        # Check if any sub-agent is active
        any_active = any(sa.is_active for sa in subagents)
        if not any_active:
            spinner = Spinner("dots", text=" Analyzing results...", style="cyan")
            elements.append(spinner)

    # Final response — render as Markdown when all work is done
    if response_text and all_done:
        elements.append(Text(""))  # blank separator
        elements.append(Markdown(response_text))
    elif is_responding and not thinking_text and not has_pending_tools:
        elements.append(Text("Generating response...", style="dim"))

    return Group(*elements) if elements else Text("Processing...", style="dim")


def display_final_results(
    state: StreamState,
    thinking_max_length: int = DisplayLimits.THINKING_FINAL,
    show_thinking: bool = True,
    show_tools: bool = True,
) -> None:
    """Display final results after streaming completes."""
    if show_thinking and state.thinking_text:
        display_thinking = state.thinking_text
        if len(display_thinking) > thinking_max_length:
            half = thinking_max_length // 2
            display_thinking = display_thinking[:half] + "\n\n... (truncated) ...\n\n" + display_thinking[-half:]
        console.print(Panel(
            Text(display_thinking, style="dim"),
            title="Thinking",
            border_style="blue",
        ))

    if show_tools and state.tool_calls:
        shown_sa_names: set[str] = set()

        for i, tc in enumerate(state.tool_calls):
            has_result = i < len(state.tool_results)
            tr = state.tool_results[i] if has_result else None
            content = tr.get('content', '') if tr is not None else ''
            is_task = tc.get('name', '').lower() == 'task'

            # Task tools: show delegation line + compact sub-agent summary
            if is_task:
                console.print(_render_tool_call_line(tc, tr))
                sa_name = tc.get('args', {}).get('subagent_type', '')
                task_desc = tc.get('args', {}).get('description', '')
                matched_sa = None
                for sa in state.subagents:
                    if sa.name == sa_name or (task_desc and task_desc in (sa.description or '')):
                        matched_sa = sa
                        break
                if matched_sa:
                    shown_sa_names.add(matched_sa.name)
                    for elem in _render_subagent_section(matched_sa, compact=True):
                        console.print(elem)
                continue

            # Regular tools: show tool call line + result
            console.print(_render_tool_call_line(tc, tr))
            if has_result and tr is not None:
                result_elements = format_tool_result_compact(
                    tr['name'],
                    content,
                    max_lines=10,
                )
                for elem in result_elements:
                    console.print(elem)

        # Render any sub-agents not already shown via task tool calls
        for sa in state.subagents:
            if sa.name not in shown_sa_names and (sa.tool_calls or sa.is_active):
                for elem in _render_subagent_section(sa, compact=True):
                    console.print(elem)

        console.print()

    # Task List panel in final output
    if state.todo_items:
        console.print(_render_todo_panel(state.todo_items))
        console.print()

    if state.response_text:
        console.print()
        console.print(Markdown(state.response_text))
        console.print()


# =============================================================================
# Async-to-sync bridge
# =============================================================================

def _run_streaming(
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool,
    interactive: bool,
) -> None:
    """Run async streaming and render with Rich Live display.

    Bridges the async stream_agent_events() into synchronous Rich Live rendering
    using asyncio.run().

    Args:
        agent: Compiled agent graph
        message: User message
        thread_id: Thread ID
        show_thinking: Whether to show thinking panel
        interactive: If True, use simplified final display (no panel)
    """
    state = StreamState()

    async def _consume() -> None:
        async for event in stream_agent_events(agent, message, thread_id):
            event_type = state.handle_event(event)
            live.update(create_streaming_display(
                **state.get_display_args(),
                show_thinking=show_thinking,
            ))
            if event_type in (
                "tool_call", "tool_result",
                "subagent_start", "subagent_tool_call",
                "subagent_tool_result", "subagent_end",
            ):
                live.refresh()

    with Live(console=console, refresh_per_second=10, transient=True) as live:
        live.update(create_streaming_display(is_waiting=True))
        asyncio.run(_consume())

    if interactive:
        display_final_results(
            state,
            thinking_max_length=500,
            show_thinking=False,
            show_tools=True,
        )
    else:
        console.print()
        display_final_results(
            state,
            show_tools=True,
        )


# =============================================================================
# CLI commands
# =============================================================================

EVOSCIENTIST_ASCII_LINES = [
    r" ███████╗ ██╗   ██╗  ██████╗  ███████╗  ██████╗ ██╗ ███████╗ ███╗   ██╗ ████████╗ ██╗ ███████╗ ████████╗",
    r" ██╔════╝ ██║   ██║ ██╔═══██╗ ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗  ██║ ╚══██╔══╝ ██║ ██╔════╝ ╚══██╔══╝",
    r" █████╗   ██║   ██║ ██║   ██║ ███████╗ ██║      ██║ █████╗   ██╔██╗ ██║    ██║    ██║ ███████╗    ██║   ",
    r" ██╔══╝   ╚██╗ ██╔╝ ██║   ██║ ╚════██║ ██║      ██║ ██╔══╝   ██║╚██╗██║    ██║    ██║ ╚════██║    ██║   ",
    r" ███████╗  ╚████╔╝  ╚██████╔╝ ███████║ ╚██████╗ ██║ ███████╗ ██║ ╚████║    ██║    ██║ ███████║    ██║   ",
    r" ╚══════╝   ╚═══╝    ╚═════╝  ╚══════╝  ╚═════╝ ╚═╝ ╚══════╝ ╚═╝  ╚═══╝    ╚═╝    ╚═╝ ╚══════╝    ╚═╝   ",
]

# Blue gradient: deep navy → royal blue → sky blue → cyan
_GRADIENT_COLORS = ["#1a237e", "#1565c0", "#1e88e5", "#42a5f5", "#64b5f6", "#90caf9"]


def print_banner(thread_id: str, workspace_dir: str | None = None):
    """Print welcome banner with ASCII art logo, thread ID, and workspace path."""
    for line, color in zip(EVOSCIENTIST_ASCII_LINES, _GRADIENT_COLORS):
        console.print(Text(line, style=f"{color} bold"))
    info = Text()
    info.append("  Thread: ", style="dim")
    info.append(thread_id, style="yellow")
    if workspace_dir:
        info.append("\n  Workspace: ", style="dim")
        info.append(workspace_dir, style="cyan")
    info.append("\n  Commands: ", style="dim")
    info.append("/exit", style="bold")
    info.append(", ", style="dim")
    info.append("/new", style="bold")
    info.append(" (new session), ", style="dim")
    info.append("/thread", style="bold")
    info.append(" (show thread ID)", style="dim")
    console.print(info)
    console.print()


def cmd_interactive(agent: Any, show_thinking: bool = True, workspace_dir: str | None = None) -> None:
    """Interactive conversation mode with streaming output.

    Args:
        agent: Compiled agent graph
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
    """
    thread_id = str(uuid.uuid4())
    print_banner(thread_id, workspace_dir)

    history_file = str(os.path.expanduser("~/.EvoScientist_history"))
    session = PromptSession(
        history=FileHistory(history_file),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )

    def _print_separator():
        """Print a horizontal separator line spanning the terminal width."""
        width = console.size.width
        console.print(Text("\u2500" * width, style="dim"))

    _print_separator()
    while True:
        try:
            user_input = session.prompt(
                HTML('<ansiblue><b>&gt;</b></ansiblue> ')
            ).strip()

            if not user_input:
                # Erase the empty prompt line so it looks like nothing happened
                sys.stdout.write("\033[A\033[2K\r")
                sys.stdout.flush()
                continue

            _print_separator()

            # Special commands
            if user_input.lower() in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == "/new":
                # New session: new workspace, new agent, new thread
                workspace_dir = _create_session_workspace()
                console.print("[dim]Loading new session...[/dim]")
                agent = _load_agent(workspace_dir=workspace_dir)
                thread_id = str(uuid.uuid4())
                console.print(f"[green]New session:[/green] [yellow]{thread_id}[/yellow]")
                console.print(f"[dim]Workspace:[/dim] [cyan]{workspace_dir}[/cyan]\n")
                continue

            if user_input.lower() == "/thread":
                console.print(f"[dim]Thread:[/dim] [yellow]{thread_id}[/yellow]")
                if workspace_dir:
                    console.print(f"[dim]Workspace:[/dim] [cyan]{workspace_dir}[/cyan]")
                console.print()
                continue

            # Stream agent response
            console.print()
            _run_streaming(agent, user_input, thread_id, show_thinking, interactive=True)
            _print_separator()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def cmd_run(agent: Any, prompt: str, thread_id: str | None = None, show_thinking: bool = True, workspace_dir: str | None = None) -> None:
    """Single-shot execution with streaming display.

    Args:
        agent: Compiled agent graph
        prompt: User prompt
        thread_id: Optional thread ID (generates new one if None)
        show_thinking: Whether to display thinking panels
        workspace_dir: Per-session workspace directory path
    """
    thread_id = thread_id or str(uuid.uuid4())

    width = console.size.width
    sep = Text("\u2500" * width, style="dim")
    console.print(sep)
    console.print(Text(f"> {prompt}"))
    console.print(sep)
    console.print(f"[dim]Thread: {thread_id}[/dim]")
    if workspace_dir:
        console.print(f"[dim]Workspace: {workspace_dir}[/dim]")
    console.print()

    try:
        _run_streaming(agent, prompt, thread_id, show_thinking, interactive=False)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


# =============================================================================
# Entry point
# =============================================================================

def _create_session_workspace() -> str:
    """Create a per-session workspace directory and return its path."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = os.path.join(".", "workspace", session_id)
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def _load_agent(workspace_dir: str | None = None):
    """Load the CLI agent (with InMemorySaver checkpointer for multi-turn).

    Args:
        workspace_dir: Optional per-session workspace directory.
    """
    from .EvoScientist import create_cli_agent
    return create_cli_agent(workspace_dir=workspace_dir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="EvoScientist Agent - AI-powered research & code execution CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python -m EvoScientist --interactive

  # Single-shot query
  python -m EvoScientist "What is quantum computing?"

  # Resume a conversation thread
  python -m EvoScientist --thread-id <uuid> "Follow-up question"

  # Disable thinking display
  python -m EvoScientist --no-thinking "Your query"
""",
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Query to execute (single-shot mode)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive conversation mode",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Thread ID for conversation persistence (resume session)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking display",
    )

    args = parser.parse_args()
    show_thinking = not args.no_thinking

    # Create per-session workspace
    workspace_dir = _create_session_workspace()

    # Load agent with session workspace
    console.print("[dim]Loading agent...[/dim]")
    agent = _load_agent(workspace_dir=workspace_dir)

    if args.interactive:
        cmd_interactive(agent, show_thinking=show_thinking, workspace_dir=workspace_dir)
    elif args.prompt:
        cmd_run(agent, args.prompt, thread_id=args.thread_id, show_thinking=show_thinking, workspace_dir=workspace_dir)
    else:
        # Default: interactive mode
        cmd_interactive(agent, show_thinking=show_thinking, workspace_dir=workspace_dir)


if __name__ == "__main__":
    main()
