# https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/43_Building_a_Tool_Calling_Agent.ipynb

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import json
import requests

from haystack.components.agents import Agent
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
# from haystack.components.websearch.searchapi import SearchApiWebSearch
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage

from haystack.tools import create_tool_from_function    #
from haystack.tools import Tool
from haystack.components.tools import ToolInvoker


from haystack.utils import ComponentDevice
generator = HuggingFaceLocalChatGenerator(
    model="model/Qwen/Qwen3-0.6B", 
    task="text-generation",
    # device=ComponentDevice.from_str("cpu"),  #s
    device=ComponentDevice.resolve_device(None),
    generation_kwargs={
        "max_new_tokens": 2048,   # input prompt + max_new_tokens / max32768
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    }, 
    # streaming_callback=print_streaming_chunk
)
generator.warm_up()


def get_tool_list():
    TOOL_EXEC_API_URL = "..."
    TOOL_EXEC_API_KEY = "..."

    # 获取tools——list
    tools_list = []
    headers = {"Authorization": TOOL_EXEC_API_KEY, "Origin":"xxx.com"}
    data = {}
    source = 1101
    try:
        response = requests.post(TOOL_EXEC_API_URL+"?"+"source="+str(source), headers=headers, data=json.dumps(data), timeout=10)
        # print(json.loads(response.text))
        if response.status_code == 200:
            print("工具列表获取成功: 工具个数", len(json.loads(response.text)['data']))  # 28
            tools_list = json.loads(response.text)['data']
        else:
            tools_list = [{"请求出错":json.loads(response.text)}]
    except Exception as err:
        print(f'An error occurred: {err}')

    for tool_name in tools_list:
        if tool_name is not None:
            print(tool_name["name"])
    
    return tools_list


# 获取工具列表
tools_list = get_tool_list()


def replace_text_with_string(schema):
    """
    递归地将 schema 中所有的 type=text 替换为 type=string。
    JSON Schema 的核心定义了以下基本类型：https://json-schema.xiniushu.com/json-schema-reference/
    """
    if isinstance(schema, dict):
        for key, value in schema.items():
            # 处理 jsonschema 库不识别类
            if key == "type" and value == "text":
                schema[key] = "string"
            elif key == "type" and value == "decimal":
                schema[key] = "number"
            elif key=="type" and value == "int":
                schema[key] = "integer"
            elif key == "type" and value in ("date", "datetime"):
                schema[key] = "string"
                # schema["format"] = "date-time"
            else:
                replace_text_with_string(value)
    elif isinstance(schema, list):
        for item in schema:
            replace_text_with_string(item)

# schema类型转换
for tool in tools_list:
    if "inputSchema" in tool:
        replace_text_with_string(tool["inputSchema"])
        if "ai_required" in tool["inputSchema"]:
            tool["inputSchema"]['required'] = tool["inputSchema"]['ai_required']


# 函数映射
class func_map:
    def __init__(self, tool_name):
        self.tool_name = tool_name

    def tool_implement(self, **params):
        TOOL_EXEC_API_URL = "..."
        TOOL_EXEC_API_KEY = "..."

        tools_exec = []
        headers = {
            "Authorization": TOOL_EXEC_API_KEY,
            "Content-Type":"application/json",
            "Origin":"xxx.com"
        }

        data = {
            "tool_name": self.tool_name,
            "idaas_open_id": "...",
            "scope_description": "0,1101",
            "params": params
        }

        try:
            response = httpx.post(TOOL_EXEC_API_URL, headers=headers, data=json.dumps(data), timeout=10)
            if response.status_code == 200:
                tools_exec = json.loads(response.text)
            else:
                tools_exec = [{"请求出错":json.loads(response.text)}]
        except Exception as err:
            print(f'An error occurred: {err}')
            
        return (tools_exec)


# 1、通过工具获取员工个人信息（user_info_tool）
first_tool = next(tool for tool in tools_list if tool["name"] == "user_info_tool")
tools_list.pop(tools_list.index(first_tool))
print("工具个数", len(tools_list))  # 26

# Use the tool
tool = Tool(
    name=first_tool["name"],
    description=first_tool["description"],
    parameters=first_tool["inputSchema"],
    function=func_map("user_info_tool").tool_implement,
)
result = tool.invoke(fields=[])
employee_info = result['data']['data']['template']['data']['form_data']
# 打印结构化信息
for key, value in employee_info.items():
    print(f"{key}: {value}")


# 2、通过个人信息和Q进行工具调用

# 函数封装成工具类 Tool
toolset = [
    Tool(
        name=one_tool["name"],
        description=one_tool["description"],
        parameters=one_tool["inputSchema"],
        function=func_map(one_tool["name"]).tool_implement,
    )
    for one_tool in tools_list
]

# Create the agent with the web search tool
agent = Agent(
    chat_generator=generator, 
    tools=toolset, 
    # system_prompt="你是一个有用的机器人",
    # exit_conditions=["text"],    # List of conditions that will cause the agent to return.
    # max_agent_steps=2,            # Maximum number of steps the agent will run before stopping.
    # raise_on_tool_invocation_failure=True
)
agent.warm_up()
# agent.to_dict()

# Run the agent with a query
user_message = ChatMessage.from_user("我的年假还有多少天？")
employee_info = ChatMessage.from_user(str(employee_info))
result = agent.run(messages=[
    user_message, 
    employee_info, 
])

# Print the final response
print(result["messages"])
print(result["messages"][-1].text)
# 'finish_reason': 'stop'

