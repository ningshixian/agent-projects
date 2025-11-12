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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# OpenAI方式调用 LLM
openai_api_key = "ak-..."
openai_api_base = "https://.../v1/chat/completions"
os.environ["OPENAI_API_KEY"] = openai_api_key

import random, re
import httpx
import json
import requests
from typing import Annotated, Callable, Tuple, List
from dataclasses import dataclass, field
import inspect

from haystack import Document
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import create_tool_from_function    #
from haystack.tools import Tool
from haystack.components.tools import ToolInvoker
from haystack.utils import ComponentDevice

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
            
        return (tools_exec)


@dataclass
class SwarmAgent:
    name: str = "SwarmAgent"
    # 4、创建 LLM
    llm: object = OpenAIChatGenerator(
        model='QWEN/qwen3-235b-a22b', 
        api_base_url=openai_api_base, 
        # streaming_callback=print_streaming_chunk, 
        generation_kwargs={
            "temperature": 0.9,       # 保持一定随机性
            "top_p": 0.95,
        }, 
        timeout=60, 
        max_retries=2,
    )
    instructions: str = "你是一个乐于助人的智能Agent"
    functions: list[Callable] = field(default_factory=list)

    def __post_init__(self):
        self._system_message = ChatMessage.from_system(self.instructions)
        self.tools = []
        for _tool in self.functions:
            if inspect.isfunction(_tool):  # def
                self.tools.append(create_tool_from_function(_tool))
            elif isinstance(_tool, dict):   # tool
                self.tools.append(
                    Tool(
                        name=_tool["name"],
                        description=_tool["description"],
                        parameters=_tool["inputSchema"],
                        function=ToolFunction(_tool["name"], _tool["scope_description"]).run,
                    )
                )
            else:
                raise Exception("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 创建工具调用器实例，并负责调用它们
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:
        # generate response
        agent_message = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)["replies"][0]
        new_messages = [agent_message]

        # # 直接回复
        # if agent_message.text:
        #     print(f"\n{self.name}: {agent_message.text}")
        #     # print(f"~~~~~ final replies: {agent_message.text}")

        # 存在'finish_reason': 'stop'，但实际输出中包含tool_call的情况？
        if "tool_call" in agent_message.text:
            # 使用一个正则表达式直接捕获 name 和 arguments 的值
            # 注意：这个方法对于复杂的 arguments (如内嵌花括号) 可能会失效
            pattern = r'"name":\s*"([^"]+)",\s*"arguments":\s*({.*?})'
            match = re.search(pattern, agent_message.text, re.DOTALL)
            
            tool_name = match.group(1) if match else None
            arguments_str = match.group(2) if match else None
            agent_message.tool_calls = [ChatMessage(_role='assistant', _content=[ToolCall(tool_name=tool_name, arguments=eval(arguments_str) )] )]

        # 无工具调用时直接返回 final answer → 
        if not agent_message.tool_calls:
            print(f"\n{self.name}: {agent_message._content}")
            return self.name, new_messages

        # 处理工具调用（补充缺失的ID）
        for tc in agent_message.tool_calls:
            # trick: Ollama does not produce IDs, but OpenAI and Anthropic require them.
            if tc.id is None:
                tc.id = str(random.randint(0, 1000000))
        tool_results = self._tool_invoker.run(messages=[agent_message])["tool_messages"]
        new_messages.extend(tool_results)

        # 解析中文转接指令
        last_result = tool_results[-1].tool_call_result.result
        match = re.search(HANDOFF_PATTERN, last_result)
        new_agent_name = match.group(1) if match else self.name

        print(f"\n{self.name}: {agent_message._content}")
        return new_agent_name, new_messages


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

tools_list.pop(tools_list.index(first_tool))    # 删除该工具
print("function calling 工具个数", len(tools_list))  # 26

"""复杂多智能体系统"""

# 中文转接模板与匹配模式
HANDOFF_TEMPLATE = "已转接至：{agent_name}。请立即切换角色。"
HANDOFF_PATTERN = r"已转接至：(.*?)(?:。|$)"  # 匹配中文句号


def escalate_to_human(summary: Annotated[str, "问题摘要（中文描述）"]):
    """仅在用户明确要求转人工时调用此工具"""
    print("正在转接至人工客服...")
    print("\n=== 转接报告 ===")
    print(f"问题摘要：{summary}")
    print("=========================\n")
    exit()

def transfer_to_leave_application_agent():
    """用于所有假期申请相关的咨询（如年假、病假、事假等各类假期申请）"""
    return HANDOFF_TEMPLATE.format(agent_name="假期申请")

def transfer_to_status_query_agent():
    """用于查询员工个人考勤记录、假期余额、排班信息等"""
    return HANDOFF_TEMPLATE.format(agent_name="状态查询")

def transfer_to_leave_manage_agent():
    """用于管理已提交的请假申请（如撤销申请或提前结束假期）"""
    return HANDOFF_TEMPLATE.format(agent_name="假期管理")

def transfer_to_policy_query_agent():
    """用于解释公司的考勤与假期相关的政策、计算规则和资格条件"""
    return HANDOFF_TEMPLATE.format(agent_name="政策查询")

def transfer_to_system_support_agent():
    """用于负责处理假期系统的登录故障、申请提交报错、页面异常等技术问题"""
    return HANDOFF_TEMPLATE.format(agent_name="系统支持")

def transfer_back_to_triage():
    """当用户问题超出当前代理职责范围时调用（包括要求转人工）"""
    return HANDOFF_TEMPLATE.format(agent_name="分诊代理")


# 分诊代理（中文指令）
triage_agent = SwarmAgent(
    name="分诊代理",
    instructions=(
        "你是公司的人力客服机器人。"
        "首先用一句话简短自我介绍（例如：您好，我是人力客服分诊代理，有什么可以帮您？）。"
        "如果用户询问一般人力问题（如招聘流程、岗位信息、公司福利政策），直接用中文简短回答，无需转接。"
        "如果用户咨询假期申请相关问题（如年假、病假、事假等各类假期申请），调用工具转至假期申请代理。"
        "如果用户需要查询员工个人考勤记录、假期余额、排班信息等，调用工具转至状态查询代理。"
        "如果用户需要管理已提交的请假申请（如撤销申请或提前结束假期），调用工具转至假期管理代理。"
        "如果用户咨询公司考勤与假期相关的政策、计算规则或资格条件，调用工具转至政策查询代理。"
        "如果用户遇到假期系统的登录故障、申请提交报错、页面异常等技术问题，调用工具转至系统支持代理。"
        "仅在必要时调用工具，确保参数正确。"
    ),
    functions=[
        transfer_to_leave_application_agent, transfer_to_status_query_agent, transfer_to_leave_manage_agent, 
        transfer_to_policy_query_agent, transfer_to_system_support_agent, escalate_to_human
    ],
)


leave_application = {
    "考勤-请育儿假": "attendance_apply_for_childcare_leave",
    "考勤-请事假": "attendance_apply_for_personal_leave",
    "考勤-请年假": "attendance_apply_for_annual_leave",
    "考勤-请婚假": "attendance_apply_for_marriage_leave",
    "考勤-请病假": "attendance_apply_for_sick_leave",
    "考勤-请丧假": "attendance_apply_for_bereavement_leave",
    "考勤-请陪产假": "attendance_apply_for_paternity_leave",
    "考勤-请工伤假": "attendance_apply_for_work_injury_leave",
    "考勤-请产假": "attendance_apply_for_maternity_leave",
    "考勤-请产检假": "attendance_apply_for_prenatal_checkup_leave",
    "考勤-请独生子女护理假": "attendance_apply_for_only_child_care_leave",
    "考勤-请计划生育假": "attendance_apply_for_family_planning_leave",
    "考勤-请哺乳假": "attendance_apply_for_nursing_leave",
    "考勤-请跨国工作探亲假": "attendance_apply_for_overseas_family_visit_leave",
}
leave_manage = {
    "考勤-撤销请假": "attendance_cancel_leave_request",
    "考勤-销假": "attendance_end_leave_early",
}
status_query = {
    "考勤-查询请假记录": "attendance_get_leave_records",
    "考勤-查询跨国工作探亲假": "attendance_get_overseas_family_visit_leave",
    "考勤-查询育儿假": "attendance_get_childcare_leave_balance",
    "考勤-查询年假": "attendance_get_annual_leave_balance",
    "考勤-查询销假记录": "attendance_get_early_leave_ending_records",
    # "考勤-查询员工信息": "attendance_get_employee_info",
    "考勤-查询员工考勤日报": "attendance_get_employee_daily_report",
    "考勤-查询员工排班": "attendance_get_employee_schedule",
}
policy_query = {
    "考勤-查询离职年假计算规则": "attendance_get_resignation_annual_leave_rules",
    "考勤-查询年假计算规则": "attendance_get_annual_leave_rules",
}
system_support = {
    "考勤-页面访问记录": "attendance_get_page_access_log",
    "考勤-查询年假申请界面天数显示有误原因": "attendance_get_reason_for_leave_days_display_error"
}


# 假期申请代理
leave_application_agent = SwarmAgent(
    name="假期申请代理",
    instructions=(
        "你是的人力助手，负责各类假期申请相关的代理（如年假、病假、事假等）。回答始终控制在一句话内。"
        "请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
        # "按以下流程处理用户问题："
        # "1. 若用户未说明假期类型，先确认（如年假/病假/事假等）\n"
        # "2. 提醒需附相关证明材料（如病假需医疗证明、事假需说明事由）"
        f"\n当前的日期是：{str(cur_time)}。"
        f"\n员工的基本信息如下：{str(employee_info)}"
    ),
    functions=[x for x in tools_list if x['name'] in leave_application] + [transfer_back_to_triage],
    # functions=[execute_order, transfer_back_to_triage],
)

# 状态查询代理
status_query_agent = SwarmAgent(
    name="状态查询代理",
    instructions=(
        "你是的人力助手，负责员工个人状态查询的代理（如考勤记录、假期余额、排班信息等）。回答始终控制在一句话内。"
        "请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
        # "按以下流程处理用户问题："
        # "1. 若用户未明确查询内容，先确认（考勤记录/假期余额/排班信息）\n"
        f"\n当前的日期是：{str(cur_time)}。"
        f"\n员工的基本信息如下：{str(employee_info)}"
    ),
    functions=[x for x in tools_list if x['name'] in status_query] + [transfer_back_to_triage],
)

# 假期管理代理
leave_manage_agent = SwarmAgent(
    name="假期管理代理",
    instructions=(
        "你是的人力助手，负责假期管理的代理（如撤销申请或提前结束假期）。回答始终控制在一句话内。"
        "请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
        # "按以下流程处理用户问题："
        # "1. 若用户未说明操作类型，先确认（撤销申请/提前结束假期）\n"
        # "2. 提醒未审批申请可直接操作，已审批需联系直属上级确认"
        f"\n当前的日期是：{str(cur_time)}。"
        f"\n员工的基本信息如下：{str(employee_info)}"
    ),
    functions=[x for x in tools_list if x['name'] in leave_manage] + [transfer_back_to_triage],
)

# 政策查询代理
policy_query_agent = SwarmAgent(
    name="政策查询代理",
    instructions=(
        "你是的人力助手，负责政策查询的代理（如考勤与假期相关的政策、计算规则和资格条件）。回答始终控制在一句话内。"
        "请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
        # "按以下流程处理用户问题："
        # "1. 若用户未明确政策细节，先确认（如年假计算/病假资格/事假规则）\n"
        # "2. 依据《员工考勤与假期管理办法》简化解释（如“年假按司龄计算，满1年享5天”）\n"
        f"\n当前的日期是：{str(cur_time)}。"
        f"\n员工的基本信息如下：{str(employee_info)}"
    ),
    functions=[x for x in tools_list if x['name'] in policy_query] + [transfer_back_to_triage],
)

# 系统支持代理
system_support_agent = SwarmAgent(
    name="系统支持代理",
    instructions=(
        "你是的人力助手，负责系统支持的代理（如登录故障、申请提交报错、页面异常等）。回答始终控制在一句话内。"
        "请务必使用所提供的工具解决问题。对于函数参数的取值，切勿擅自假设。若用户的需求表述存在歧义，请主动询问以明确细节。"
        # "按以下流程处理用户问题："
        # "1. 若用户未描述具体问题，先确认（登录故障/提交报错/页面异常）\n"
        # "2. 若无效，指导拨打IT支持热线400-888-XXXX并提供报错截图"
        f"\n当前的日期是：{str(cur_time)}。"
        f"\n员工的基本信息如下：{str(employee_info)}"
    ),
    functions=[x for x in tools_list if x['name'] in system_support] + [transfer_back_to_triage],
)


# 5、代理注册与启动
agents = {agent.name: agent for agent in [triage_agent, leave_application_agent, leave_manage_agent, status_query_agent, policy_query_agent, system_support_agent]}

print("输入'quit'退出对话")  # 中文提示

messages: List[ChatMessage] = []
current_agent_name = "分诊代理"  # 初始代理为中文名称

# 6、创建 Workflow
while True:
    if not current_agent_name.endswith("代理"): 
        current_agent_name += "代理"
    
    agent = agents[current_agent_name]

    # 用户输入环节（仅当需要用户回复时）
    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("用户：")  # 中文输入提示
        if user_input.lower() == "quit":
            break
        messages.append(ChatMessage.from_user(user_input))

    # 代理处理与状态更新
    current_agent_name, new_messages = agent.run(messages)
    messages.extend(new_messages)
