"""Middleware configuration for the EvoScientist agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware

from .backends import MergedReadOnlyBackend
from .memory import EvoMemoryMiddleware
from .paths import MEMORY_DIR as _DEFAULT_MEMORY_DIR

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel

_DEFAULT_SKILLS_DIR = str(Path(__file__).parent / "skills")


def create_skills_middleware(
    skills_dir: str = _DEFAULT_SKILLS_DIR,
    workspace_dir: str = ".",
    user_skills_dir: str | None = None,
) -> SkillsMiddleware:
    """Create a SkillsMiddleware that loads skills.

    Merges user-installed skills (./skills/) with system skills
    (package built-in). User skills take priority on name conflicts.

    Args:
        skills_dir: Path to the system skills directory (package built-in)
        workspace_dir: Path to the project root (user skills live under {workspace_dir}/skills/)
        user_skills_dir: Optional explicit path for user-installed skills. If set,
            this path is used directly instead of {workspace_dir}/skills.

    Returns:
        Configured SkillsMiddleware instance
    """
    if user_skills_dir is None:
        user_skills_dir = str(Path(workspace_dir) / "skills")
    merged = MergedReadOnlyBackend(
        primary_dir=user_skills_dir,
        secondary_dir=skills_dir,
    )
    return SkillsMiddleware(
        backend=merged,
        sources=["/"],
    )


def create_memory_middleware(
    memory_dir: str = str(_DEFAULT_MEMORY_DIR),
    extraction_model: BaseChatModel | None = None,
    trigger: tuple[str, int] = ("messages", 20),
) -> EvoMemoryMiddleware:
    """Create an EvoMemoryMiddleware for long-term memory.

    Uses a FilesystemBackend rooted at ``memory_dir`` so that memory
    persists across threads and sessions.

    Args:
        memory_dir: Path to the shared memory directory (not per-session).
        extraction_model: Chat model for auto-extraction (optional; if None,
            only prompt-guided manual memory updates via edit_file will work).
        trigger: When to auto-extract. Default: every 20 human messages.

    Returns:
        Configured EvoMemoryMiddleware instance.
    """
    memory_backend = FilesystemBackend(
        root_dir=memory_dir,
        virtual_mode=True,
    )
    return EvoMemoryMiddleware(
        backend=memory_backend,
        memory_path="/MEMORY.md",
        extraction_model=extraction_model,
        trigger=trigger,
    )
