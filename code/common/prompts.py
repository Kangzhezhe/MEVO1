"""Prompt 模板加载器；模板版本放在 prompt/lamp5，不在 Python 中硬编码。"""

from __future__ import annotations

from string import Template

from common.utils import project_root


def render_prompt(relative_path: str, **values) -> str:
    path = project_root() / "prompt" / relative_path
    template = Template(path.read_text(encoding="utf-8").strip())
    return template.substitute({key: str(value) for key, value in values.items()})
