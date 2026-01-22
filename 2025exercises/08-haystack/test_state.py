from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import merge_lists, replace_values

schema = {
    "documents": {"type": list, "handler": merge_lists},  # 只有 list 类型的数据合并，其他都是覆盖更新
    "user_name": {"type": str},   # Uses replace_values by default
    "count": {"type": int}         # Uses replace_values by default
}
state = State(schema=schema)

## The messages field is available
print("messages" in state.schema)  # True
print(state.schema["messages"]["type"])  # list[ChatMessage]

## Lists are merged by default
state.set("documents", [1, 2])
state.set("documents", [{"title": "Doc 1", "content": "Content 1"}])
print(state.get("documents"))  # Output: [1, 2, 3, 4]

## Other values are replaced
state.set("user_name", "Alice")
state.set("user_name", "Bob")
print(state.get("user_name"))  # Output: "Bob"

# =================================== Using State with Agents ====================================== #

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

import os
from dotenv import load_dotenv
load_dotenv()   # 加载环境变量 (.env 文件)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

## Define a simple calculation tool
def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression."""
    result = eval(expression, {"__builtins__": {}})
    return {"func_result": result}

## Create a tool that writes to state
calculator_tool = Tool(
    name="calculator",
    description="Evaluate basic math expressions",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]
    },
    function=calculate,
    inputs_from_state={"input": "expression"}, 
    outputs_to_state={"calc_result": {"source": "func_result"}}  # Maps tool's "func_result" output to state's "calc_result"
)

## Create agent with state schema
generator = OpenAIChatGenerator(
    model='deepseek-chat',
    api_base_url=OPENAI_API_BASE,
    # api_key=Secret.from_token(OPENAI_API_KEY),
    timeout=60
)
agent = Agent(
    chat_generator=generator,
    tools=[calculator_tool],
    state_schema={
        "input": {"type": str},
        "calc_result": {"type": int}
    }
)

## Run the agent
result = agent.run(
    messages=[ChatMessage.from_user("Calculate 15 + 27")],
    user_name="Alice"  # All additional kwargs passed to Agent at runtime are put into State
)

## Access the state from results
calc_result = result["calc_result"]
print(calc_result)  # Output: 42

## Access messages from execution
for message in result["messages"]:
    print(f"{message.role}: {message.text}")