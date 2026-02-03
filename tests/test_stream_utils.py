"""Tests for EvoScientist/stream/utils.py pure functions."""

from EvoScientist.stream.utils import (
    is_success,
    format_tool_compact,
    truncate,
    has_args,
    count_lines,
    truncate_with_line_hint,
    _shorten_path,
)


# === is_success ===

class TestIsSuccess:
    def test_ok_prefix(self):
        assert is_success("[OK] all good") is True

    def test_failed_prefix(self):
        assert is_success("[FAILED] bad") is False

    def test_traceback(self):
        assert is_success("Traceback (most recent call last)\n  File ...") is False

    def test_exception(self):
        assert is_success("Exception: something went wrong") is False

    def test_error(self):
        assert is_success("Error: file not found") is False

    def test_clean_output(self):
        assert is_success("file1.py\nfile2.py") is True

    def test_whitespace_stripped(self):
        assert is_success("  [OK] with spaces  ") is True


# === format_tool_compact ===

class TestFormatToolCompact:
    def test_no_args(self):
        assert format_tool_compact("execute", None) == "execute()"
        assert format_tool_compact("execute", {}) == "execute()"

    def test_execute(self):
        result = format_tool_compact("execute", {"command": "ls -la"})
        assert result == "execute(ls -la)"

    def test_execute_long_command(self):
        long_cmd = "x" * 60
        result = format_tool_compact("execute", {"command": long_cmd})
        assert len(result) < 70
        assert result.endswith("\u2026)")

    def test_read_file(self):
        result = format_tool_compact("read_file", {"path": "src/main.py"})
        assert result == "read_file(src/main.py)"

    def test_write_file(self):
        result = format_tool_compact("write_file", {"path": "out.txt"})
        assert result == "write_file(out.txt)"

    def test_edit_file(self):
        result = format_tool_compact("edit_file", {"path": "f.py"})
        assert result == "edit_file(f.py)"

    # Memory file special display
    def test_read_file_memory(self):
        result = format_tool_compact("read_file", {"path": "/memory/MEMORY.md"})
        assert result == "Reading memory"

    def test_write_file_memory(self):
        result = format_tool_compact("write_file", {"path": "/MEMORY.md"})
        assert result == "Updating memory"

    def test_write_file_memory_with_prefix(self):
        result = format_tool_compact("write_file", {"path": "/memory/MEMORY.md"})
        assert result == "Updating memory"

    def test_edit_file_memory(self):
        result = format_tool_compact("edit_file", {"path": "/memory/MEMORY.md"})
        assert result == "Updating memory"

    def test_glob(self):
        result = format_tool_compact("glob", {"pattern": "*.py"})
        assert result == "glob(*.py)"

    def test_grep(self):
        result = format_tool_compact("grep", {"pattern": "TODO", "path": "src/"})
        assert result == "grep(TODO, src/)"

    def test_ls(self):
        assert format_tool_compact("ls", {"path": "/src"}) == "ls(/src)"

    def test_write_todos_list(self):
        todos = [{"status": "todo", "content": "a"}, {"status": "todo", "content": "b"}]
        result = format_tool_compact("write_todos", {"todos": todos})
        assert result == "write_todos(2 items)"

    def test_write_todos_non_list(self):
        result = format_tool_compact("write_todos", {"todos": "something"})
        assert result == "write_todos(...)"

    def test_read_todos(self):
        assert format_tool_compact("read_todos", {}) == "read_todos()"

    def test_task_with_type_and_desc(self):
        result = format_tool_compact("task", {"subagent_type": "research-agent", "description": "Find papers"})
        assert "Cooking with research-agent" in result
        assert "Find papers" in result

    def test_task_with_type_only(self):
        result = format_tool_compact("task", {"subagent_type": "code-agent"})
        assert result == "Cooking with code-agent"

    def test_task_with_desc_only(self):
        result = format_tool_compact("task", {"description": "do stuff"})
        assert "Cooking with sub-agent" in result

    def test_task_no_info(self):
        result = format_tool_compact("task", {"other": "value"})
        assert result == "Cooking with sub-agent"

    def test_load_skill(self):
        result = format_tool_compact("load_skill", {"skill_name": "vllm"})
        assert result == "load_skill(vllm)"

    def test_load_skill_name_key(self):
        result = format_tool_compact("load_skill", {"name": "peft"})
        assert result == "load_skill(peft)"

    def test_tavily_search(self):
        result = format_tool_compact("tavily_search", {"query": "python testing"})
        assert result == "tavily_search(python testing)"

    def test_think_tool(self):
        result = format_tool_compact("think_tool", {"reflection": "need more data"})
        assert result == "think_tool(need more data)"

    def test_unknown_tool(self):
        result = format_tool_compact("custom_tool", {"key": "value"})
        assert "custom_tool(" in result
        assert "key=value" in result

    def test_unknown_tool_long_value(self):
        result = format_tool_compact("custom_tool", {"key": "a" * 30})
        assert "\u2026" in result


# === truncate ===

class TestTruncate:
    def test_within_limit(self):
        assert truncate("hello", 10) == "hello"

    def test_at_limit(self):
        assert truncate("hello", 5) == "hello"

    def test_over_limit(self):
        result = truncate("hello world", 5)
        assert result.startswith("hello")
        assert "truncated" in result


# === _shorten_path ===

class TestShortenPath:
    def test_short_path(self):
        assert _shorten_path("src/main.py") == "src/main.py"

    def test_long_path(self):
        long_path = "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r.py"
        result = _shorten_path(long_path, max_len=20)
        assert result.startswith(".../")
        assert result.endswith("r.py")


# === has_args ===

class TestHasArgs:
    def test_none(self):
        assert has_args(None) is False

    def test_empty_dict(self):
        assert has_args({}) is False

    def test_non_empty(self):
        assert has_args({"key": "val"}) is True


# === count_lines ===

class TestCountLines:
    def test_empty(self):
        assert count_lines("") == 0

    def test_single_line(self):
        assert count_lines("hello") == 1

    def test_multi_line(self):
        assert count_lines("a\nb\nc") == 3


# === truncate_with_line_hint ===

class TestTruncateWithLineHint:
    def test_within_limit(self):
        text, remaining = truncate_with_line_hint("a\nb\nc", max_lines=5)
        assert remaining == 0
        assert "a" in text

    def test_over_limit(self):
        text, remaining = truncate_with_line_hint("a\nb\nc\nd\ne\nf", max_lines=3)
        assert remaining == 3
        assert "d" not in text
