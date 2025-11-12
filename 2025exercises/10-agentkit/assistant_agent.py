import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from agents import Agent, RunConfig, Runner, function_tool
from agents.model_settings import ModelSettings
from chatkit.agents import AgentContext


# 定义一个简单的工具函数
@function_tool
def save_note(content: str) -> str:
    """保存笔记到文件"""
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"{content}\n")
    return f"笔记已保存：{content}"


def _build_tools() -> List[Any]:
    return [save_note]


model_settings = ModelSettings(
    temperature=0.5,
    # reasoning={"effort": "minimal"},
)

# 创建 Agent
assistant_agent = Agent[AgentContext](
    model="gpt-3.5-turbo",
    model_settings=model_settings,
    name="MyFirstAgent",
    instructions="""你是一个笔记助手。
当用户需要记录信息时，使用 save_note 工具保存。
保持简洁友好的回复风格。""",
    tools=_build_tools(),
)