# Building a agentic RAG with Function Calling
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/40_Building_Chat_Application_with_Function_Calling.ipynb#scrollTo=ZE0SEGY92GHJ

"""
利用Haystack框架和Gradio库构建一个能够调用工具API的人力助手。
涵盖了数据获取、工具调用、语言模型集成，以及web交互界面

1、从指定API接口动态拉取可用的工具列表
2、执行 user_info_tool 工具，获取employee_info
3、创建工具集 List[Tool] 以及执行器 ToolInvoker
    - 工具类ToolFunction封装了工具的调用逻辑，通过HTTP POST请求调用外部API。
    - 每个工具都有一个名称、描述、参数和执行函数。
    - 工具调用器ToolInvoker管理多个工具实例，并负责调用它们。
4. 创建LLM 处理用户的输入消息，生成推理链
5. 定义一个动态工作流，在循环中检查是否需要调用工具，并将工具的结果整合回推理链中。
    - 当LLM生成包含工具调用的响应时，ToolInvoker会解析这些调用并执行相应的工具函数，然后将结果返回给LLM进行进一步处理。
    - 工具调用流程：tool calls + parameters → ToolInvoker→ToolFunction
"""

import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import httpx
import json
import requests
from typing import List
from pathlib import Path
import traceback

from haystack import Document
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses import StreamingChunk
from haystack.components.generators.utils import print_streaming_chunk

from typing import Annotated, Literal
from haystack.tools import create_tool_from_function    #
from haystack.tools import Tool
from haystack.components.tools import ToolInvoker

import time
from datetime import datetime
cur_time = (datetime.now().strftime("%Y-%m-%d"))


# 获取工具列表，保存在 tools_list.json
def get_tool_list():
    TOOL_EXEC_API_URL = "..."
    TOOL_EXEC_API_KEY = "..."

    # 获取tools——list
    tools_list = []
    headers = {"Authorization": TOOL_EXEC_API_KEY, "Origin":"xxx.com"}
    data = {}
    source = 1101
    try:
        response = requests.post(TOOL_EXEC_API_URL+"?"+"source="+str(source), headers=headers, data=json.dumps(data), timeout=5)
        # print(json.loads(response.text))
        if response.status_code == 200:
            print("工具列表获取成功: 工具个数", len(json.loads(response.text)['data']))  # 28
            tools_list = json.loads(response.text)['data']
        else:
            tools_list = [{"请求出错":json.loads(response.text)}]
    except Exception as err:
        print(f'An error occurred: {err}')

    # # 输出所有可用工具名称
    # for tool_name in tools_list:
    #     if tool_name is not None:
    #         print(tool_name["name"])
        
    # 保存json文件
    with open('tools_list.json', 'w', encoding='utf-8') as f:
        json.dump(tools_list, f, ensure_ascii=False, indent=4)
    
    return tools_list


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


# 工具类封装了工具调用逻辑，通过HTTP POST请求调用外部API。
class ToolFunction:
    def __init__(self, tool_name, scope_description):
        self.tool_name = tool_name
        self.scope_description = scope_description

    def run(self, **params):
        TOOL_EXEC_API_URL = "..."
        TOOL_EXEC_API_KEY = "..."

        tools_exec = []
        headers = {
            "Authorization": TOOL_EXEC_API_KEY,
            "Content-Type":"application/json",
            "Origin":"xxx.com"
        }

        # 人力助手
        data = {
            "tool_name": self.tool_name,  # "考勤-查询员工信息",
            "idaas_open_id": "...",
            "scope_description": self.scope_description,
            "params": params
        }

        try:
            response = httpx.post(TOOL_EXEC_API_URL, headers=headers, data=json.dumps(data), timeout=5)
            if response.status_code == 200:
                tools_exec = json.loads(response.text)
                tools_exec = tools_exec["data"]["data"]  # 
            else:
                tools_exec = [{"请求出错":json.loads(response.text)}]
        except Exception as err:
            print(f'An error occurred: {err}')
            
        return tools_exec


# 7.14新增 rag工具 https://li.feishu.cn/docx/RL53doJ9do7AU1x7dlMcLBq1nZg?from=from_copylink
def get_rag_answer(query):
    headers = {"Content-Type": "application/json; charset=utf-8"}  # 显式指定UTF-8编码
    response = requests.post(
        url="...", 
        data=json.dumps({"query": query}),
        headers=headers,
        timeout=30
    )

    # # 检查HTTP状态码（4xx/5xx会抛出异常）
    # response.raise_for_status()
    
    # 解析JSON响应（确保中文正常解析）
    result = response.json()
    result = {
        "query": result["query"], 
        "results": [item["content"] for item in result["results"]]
    }
    return result


# # param = {'query': "转正审批流程步骤及所需材料"}
# param = {'query': "劳动合同续签政策？"}
# print(get_rag_answer(**param))
# exit()


# 定义一个动态工作流程函数，并根据需要调用工具。
def dynamic_workflow(query, history=[]):
# def dynamic_workflow(message, history):
    global messages
    messages.append(ChatMessage.from_user(query))
    response = generator.run(messages=messages)
    # print(response)

    _history = ""
    print("\n==================start======================")

    while True:
        print(f"\n{response['replies']}\n")

        # # 流式输出：yield来生成一系列部分响应，每个响应都会替换前一个
        # _history += "\nLLM Result:\n"+str(response["replies"][0]._content)
        # yield _history

        # 执行工具，并将结果拼接回推理链中，直到 finish_reason=stop
        if response and response["replies"][0].tool_calls:
            tool_result_messages = tool_invoker.run(messages=response["replies"])["tool_messages"]
            print(f"~~~~~ tool messages: \n{tool_result_messages}")

            # # 流式输出：yield来生成一系列部分响应，每个响应都会替换前一个
            # _history += "\nTool Result:\n"+str(tool_result_messages[0]._content)
            # yield _history
            
            # Pass all the messages to the ChatGenerator with the correct order
            messages = messages + response["replies"] + tool_result_messages
            response = generator.run(messages=messages)

        elif response and "tool_call" in str(response["replies"][0]._content):
            # 存在'finish_reason': 'stop'，但实际输出中包含tool_call的情况？
            
            # 使用一个正则表达式直接捕获 name 和 arguments 的值
            # 注意：这个方法对于复杂的 arguments (如内嵌花括号) 可能会失效
            pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*({.*?})'
            match = re.search(pattern, str(response["replies"][0]._content), re.DOTALL)
            
            tool_name = None
            arguments_str = None
            
            if match:
                tool_name = match.group(1)      # 第一个捕获组是 name 的值
                arguments_str = match.group(2)  # 第二个捕获组是 arguments 的值 (字符串形式)
                # print(tool_name)
                # print(arguments_str)

                msg = [ChatMessage(_role='assistant', _content=[ToolCall(tool_name=tool_name, arguments=eval(arguments_str) )] )]
                tool_result_messages = tool_invoker.run(messages=msg)["tool_messages"]
                print(f"~~~~~ tool messages: \n{tool_result_messages}")

                # # 流式输出：yield来生成一系列部分响应，每个响应都会替换前一个
                # _history += "\nTool Result:\n"+str(tool_result_messages[0]._content)
                # yield _history

                # Pass all the messages to the ChatGenerator with the correct order
                messages = messages + response["replies"] + tool_result_messages
                response = generator.run(messages=messages)
            
        # Regular Conversation
        else:
            final_replies = response["replies"]
            print(f"≈≈≈≈≈≈ final replies: \n{final_replies}")

            # # 流式输出：yield来生成一系列部分响应，每个响应都会替换前一个
            # _history += "\n\n**Final Result:**\n"+str(final_replies[0]._content)
            # yield _history

            messages.append(response["replies"][0])
            break
    
    return response["replies"][0].text


# 1、从指定 API 接口动态拉取可用的工具列表
tools_list = get_tool_list()

# schema类型转换，确保兼容性
for tool in tools_list:
    if "inputSchema" in tool:
        replace_text_with_string(tool["inputSchema"])
        if "ai_required" in tool["inputSchema"]:
            tool["inputSchema"]['required'] = tool["inputSchema"]['ai_required']

# 2、调用工具（user_info_tool），结果保存在 employee_info
tool_name = "user_info_tool"
first_tool = next(tool for tool in tools_list if tool["name"] == tool_name)
tool = Tool(
    name=first_tool["name"],
    description=first_tool["description"],
    parameters=first_tool["inputSchema"],
    function=ToolFunction(tool_name, first_tool["scope_description"]).run,
)
result = tool.invoke(fields=[])
employee_info = result['template']['data']['form_data']
# 打印结构化信息
for key, value in employee_info.items():
    print(f"{key}: {value}")


# 3、创建工具集
tools_list.pop(tools_list.index(first_tool))    # 删除该工具
print("function calling 工具个数", len(tools_list))  # 26
toolset = [
    Tool(
        name=name_zh_to_en[_tool["name"]],
        description=_tool["description"],
        parameters=_tool["inputSchema"],
        function=ToolFunction(_tool["name"], _tool["scope_description"]).run,
    )
    for _tool in tools_list
]
# 7.14 新增 rag工具
rag_tool = Tool(
    name="retrieve_relevant_knowledge_from_document",
    description="""RAG文档检索工具。
    使用场景：
    - 政策查询（如请假、福利、培训等）
    - 流程咨询（如申请流程、审批流程等）
    - 规定说明（如制度、规范、标准等）
    - 其他需要检索的人力相关知识问题
    当且仅当其他工具都无法满足时才考虑该工具。""",
    parameters={
        "required": ["query"],
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要检索的查询文本",
                "example": "如何申请生育津贴"
            }
        }
    }, 
    function=get_rag_answer
)
toolset.append(rag_tool)

# 创建工具调用器实例，并负责调用它们
tool_invoker = ToolInvoker(tools=toolset)


# 4、创建 LLM

# 定义一个异步流式回调函数
async def async_print_streaming_chunk(chunk: StreamingChunk):
    print_streaming_chunk(chunk)

# OpenAI方式调用 LLM
openai_api_key = "ak-..."
openai_api_base = "https://.../v1/chat/completions"
os.environ["OPENAI_API_KEY"] = openai_api_key
generator = OpenAIChatGenerator(
    model='QWEN/qwen3-235b-a22b', 
    api_base_url=openai_api_base, 
    tools=toolset,
    generation_kwargs={
        "temperature": 0.6,
        "top_p": 0.7,
        # "max_tokens": 512,   # The maximum number of tokens the output text can have.
        # "frequency_penalty": 1.2,
    }, 
    timeout=60, 
    max_retries=2,
    # streaming_callback=print_streaming_chunk,    # 流式回调
    # streaming_callback=async_print_streaming_chunk,    # 异步回调 → stream=True
)


# # Test Workflow

# response = None
# messages = [
#     ChatMessage.from_system(
#         "你是一个人力助手，请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
#         + "当前的日期是：" + cur_time
#     ),
#     ChatMessage.from_system("员工的基本信息如下：\n" + str(employee_info))
# ]  # 保存 session对话，以及中间推理过程

# # query = {'query': "北京婚假"}
# query = {'query': "我的年假还有多少天？"}
# answers = dynamic_workflow(**query, history=[])
# print(answers)
# exit()


if __name__ == "__main__":

    import csv
    from tqdm import tqdm
    import pandas as pd
    df = pd.read_csv("测试集.csv", encoding="utf-8", keep_default_na=False)
    # 按 question_id 分组
    grouped_df = df.groupby('question_id')

    answers = []
    with open("测试集_haystack.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "query", "haystack_answer"])

        for question_id, group in grouped_df:
            print(f"\nQuestion ID: {question_id}")
            # 保存同一session的所有对话，以及中间推理过程
            messages = [
                ChatMessage.from_system(
                    "你是一个人力助手，请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
                    + "当前的日期是：" + cur_time
                ),
                ChatMessage.from_system("员工的基本信息如下：\n" + str(employee_info))
            ]
            # 遍历 session 下的每轮对话
            for i,row in group.iterrows():  
                print(f"\nturn ID: {row['turn_id']}")
                query = {'query': row["query"]}   # 我的年假还有多少天？
                try:
                    reply = dynamic_workflow(**query)
                except Exception as e:
                    traceback.print_exc()
                    reply = f"dynamic_workflow 执行失败. 错误信息：{e}"
                writer.writerow([i, row["query"], reply])


    # # 创建Gradio聊天界面
    # import gradio as gr
    # def build_gradio():
    #     demo = gr.ChatInterface(
    #         fn=dynamic_workflow,
    #         type="messages",
    #         examples=[
    #             "我的年假还有多少天？",
    #             "我想请明天的育儿假",
    #             "查下我的考勤日报",
    #             "把我的所有年假都请了",
    #             "年假是怎么计算的？", 
    #         ],
            
    #         title="SSC Agent!",
    #         theme=gr.themes.Ocean(),
    #     )

    #     # Uncomment the line below to launch the chat app with UI
    #     demo.launch(
    #         server_name="0.0.0.0", server_port=8086, share=False, inbrowser=True
    #     )
    
    # build_gradio()
