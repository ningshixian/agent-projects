# Building a agentic RAG with Function Calling
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/40_Building_Chat_Application_with_Function_Calling.ipynb#scrollTo=ZE0SEGY92GHJ
import os
import sys
import random, re
import httpx
import json
import requests
import traceback
from typing import Annotated, Callable, Tuple, List
from dataclasses import dataclass, field
import inspect
from jinja2 import Template

# Haystack 核心组件
from haystack import Document, Pipeline
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

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
利用Haystack框架和Gradio库构建一个能够调用工具API的人力助手。
涵盖了数据获取、工具调用、语言模型集成，以及web交互界面

1、从指定API接口动态拉取可用的工具列表
2、执行【考勤-查询员工信息】工具，获取employee_info
3、创建工具集 List[Tool] 以及执行器 ToolInvoker
    - 工具类ToolFunction封装了工具的调用逻辑，通过HTTP POST请求调用外部API。
    - 每个工具都有一个名称、描述、参数和执行函数。
    - 工具调用器ToolInvoker管理多个工具实例，并负责调用它们。
4. 创建LLM 处理用户的输入消息，生成推理链
5. 定义一个动态工作流，在循环中检查是否需要调用工具，并将工具的结果整合回推理链中。
    - 当LLM生成包含工具调用的响应时，ToolInvoker会解析这些调用并执行相应的工具函数，然后将结果返回给LLM进行进一步处理。
    - 工具调用流程：tool calls + parameters → ToolInvoker→ToolFunction
"""

from dotenv import load_dotenv
# 加载环境变量 (.env 文件)
load_dotenv()

# =================配置区域=================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

TOOL_API_URL = os.getenv("TOOL_API_URL", "")
TOOL_API_TOKEN = os.getenv("TOOL_API_TOKEN", "Bearer YOUR_TOKEN")
TOOL_RAG_URL = "..."

# 动态获取工具列表
def get_tool_list():
    tools_list = []
    headers = {"Authorization": TOOL_API_TOKEN,
            "Origin":"chehejia.com"}
    data = {}
    source = 1101
    try:
        response = requests.post(TOOL_API_URL+"?"+"source="+str(source), headers=headers, data=json.dumps(data), timeout=5)
        if response.status_code == 200:
            print("工具列表获取成功: 工具个数", len(json.loads(response.text)['data']))  # 28
            tools_list = json.loads(response.text)['data']
        else:
            tools_list = [{"请求出错":json.loads(response.text)}]
    except Exception as err:
        print(f'get_tool_list() error: {err}')

    # # 输出所有可用工具名称
    # for tool_name in tools_list:
    #     if tool_name is not None:
    #         print(tool_name["name"])
    
    # 【考勤-页面访问记录】这个工具的 schema 不太对
    tools_list = [tool for tool in tools_list if tool["name"] != "考勤-页面访问记录"]   # 27

    # # 保存json文件
    # with open('tools_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(tools_list, f, ensure_ascii=False, indent=4)
    
    return tools_list


def replace_text_with_string(schema):
    """
    通过递归的方式，解决API数据定义使用的类型名称与标准 JSON Schema 规范 不兼容的问题。
    官方 JSON Schema 规范：https://json-schema.xiniushu.com/json-schema-reference/
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


# 统一封装工具调用逻辑
# 通过HTTP POST请求调用外部API
class ToolFunction:
    def __init__(self, tool_name, scope_description):
        self.tool_name = tool_name
        self.scope_description = scope_description

    def run(self, **params):
        tools_exec = []
        headers = {
            "Authorization": TOOL_API_TOKEN,
            "Content-Type":"application/json",
            "Origin":"chehejia.com"
        }

        # 人力助手
        data = {
            "tool_name": self.tool_name,                    # 要执行的工具名称
            "idaas_open_id": "3OqQ0gs3YuwUveqOylMjiw",      # 员工ID
            "scope_description": self.scope_description,    # 工具作用域范围
            "params": params                                # 工具参数 (ai_required 里对应的参数)
        }

        try:
            response = httpx.post(TOOL_API_URL, headers=headers, data=json.dumps(data), timeout=5)
            tools_exec = json.loads(response.text)
            tools_exec = tools_exec["data"]["data"]  # 
        except Exception as err:
            print(f'[ERROR] An error occurred: {err}')
            traceback.print_exc()
            tools_exec = "工具执行出错，请重新检查"
            
        return (tools_exec)


@dataclass
class SwarmAgent:
    name: str = "SwarmAgent"
    # 4、创建 LLM
    llm: object = OpenAIChatGenerator(
        model='deepseek-chat', 
        # api_key=Secret.from_token(OPENAI_API_KEY),
        api_base_url=OPENAI_API_BASE, 
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
                        name=zh2en(_tool["name"]),
                        description=_tool["description"],
                        parameters=_tool["inputSchema"],
                        function=ToolFunction(zh2en(_tool["name"]), _tool["scope_description"]).run,
                    )
                )
            else:
                raise Exception("tool 类型错误!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 创建工具调用器实例，并负责调用它们
        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None

    def run(self, messages: list[ChatMessage]) -> Tuple[str, list[ChatMessage]]:
        """
        执行 ReAct 循环：
        Reason -> Act -> Observe -> Reflect(若出错) -> Final Answer
        """
        _response = self.llm.run(messages=[self._system_message] + messages, tools=self.tools)
        agent_message = _response["replies"][0]
        new_messages = [agent_message]
        print(f"\n{self.name}: {agent_message._content}")

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
            # print(f"不使用工具~")
            return self.name, new_messages

        # 处理工具调用
        print(f"使用工具 {agent_message.tool_calls}")
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

        print(f"工具调用结果：{tool_results}")
        print(f"工具last_result：{tool_results[-1].tool_call_result.result}")

        return new_agent_name, new_messages


# 1、从指定 API 接口动态拉取可用的工具列表
tools_list = get_tool_list()

# schema类型转换，确保兼容性
for tool in tools_list:
    if "inputSchema" in tool:
        replace_text_with_string(tool["inputSchema"])
        if "ai_required" in tool["inputSchema"]:
            tool["inputSchema"]['required'] = tool["inputSchema"]['ai_required']

# 7.14新增 rag工具
def retrieve_knowledge_with_rag(query:str):
    """RAG文档检索工具。
    - 政策查询（如请假、福利、培训等）
    - 流程咨询（如申请流程、审批流程等）
    - 规定说明（如制度、规范、标准等）
    - 其他需要检索的人力相关知识问题
    当且仅当其他工具都无法满足时才考虑该工具。
    """
    headers = {"Content-Type": "application/json; charset=utf-8"}  # 显式指定UTF-8编码
    response = requests.post(
        url=TOOL_RAG_URL, 
        data=json.dumps({"query": query}),
        headers=headers,
        timeout=30
    )
    result = response.json()
    return {
        "query": result["query"], 
        "results": [item["content"] for item in result["results"]]
    }

# # 2、调用工具（考勤-查询员工信息），结果保存在 employee_info
# tool_name = "考勤-查询员工信息"
# first_tool = next(tool for tool in tools_list if tool["name"] == tool_name)
# tool = Tool(
#     name=first_tool["name"],
#     description=first_tool["description"],
#     parameters=first_tool["inputSchema"],
#     function=ToolFunction(tool_name, first_tool["scope_description"]).run,
# )
# result = tool.invoke(fields=[])
# print(result)
# employee_info = result['template']['data']['form_data']
# # 打印结构化信息
# for key, value in employee_info.items():
#     print(f"{key}: {value}")
# tools_list.pop(tools_list.index(first_tool))    # 删除该工具
# print("function calling 工具个数", len(tools_list))  # 26

employee_info = {
    "职级": "16",
    "际工作城市名称": "北京市",
    "社保城市名称": "北京市",
    "性别": "男",
    "入职日期": "2025-06-30",
    "工龄（年）": "1.1年",
    "工作城市名称": "北京市",
    "姓名": "张*涛",
    "工号": "001898",
    "员工组名称": "白领",
    "部门名称": "组织系统"
}

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

def transfer_to_leave_agent():
    """用于所有假期申请相关的咨询（如年假、病假、事假等各类假期申请）"""
    return HANDOFF_TEMPLATE.format(agent_name="假期申请代理")

def transfer_to_status_query_agent():
    """用于查询员工个人考勤记录、假期余额、排班信息等"""
    return HANDOFF_TEMPLATE.format(agent_name="状态查询代理")

def transfer_to_leave_manage_agent():
    """用于管理已提交的请假申请（如撤销申请或提前结束假期）"""
    return HANDOFF_TEMPLATE.format(agent_name="假期管理代理")

def transfer_to_policy_query_agent():
    """用于解释公司的考勤与假期相关的政策、计算规则和资格条件"""
    return HANDOFF_TEMPLATE.format(agent_name="政策查询代理")

def transfer_to_system_support_agent():
    """用于负责处理假期系统的登录故障、申请提交报错、页面异常等技术问题"""
    return HANDOFF_TEMPLATE.format(agent_name="系统支持代理")

def transfer_back_to_triage():
    """当用户问题超出当前代理职责范围时调用（包括要求转人工）"""
    return HANDOFF_TEMPLATE.format(agent_name="分诊代理")


triage_prompt = """
你将扮演理想汽车公司的人力客服机器人，负责处理员工的人力相关咨询并进行合理分诊。

请严格按照以下流程和规则执行任务：

1. **问题判断与处理**：
   - **一般人力问题**：如果用户询问招聘流程、岗位信息、公司福利政策等一般性人力问题，直接用中文简短回答，无需转接。
   - **假期申请问题**：如果用户咨询年假、病假、事假等各类假期申请相关问题，调用工具转至假期申请代理。
   - **状态查询问题**：如果用户需要查询员工个人考勤记录、假期余额、排班信息等，调用工具转至状态查询代理。
   - **假期管理问题**：如果用户需要管理已提交的请假申请（如撤销申请或提前结束假期），调用工具转至假期管理代理。
   - **政策查询问题**：如果用户咨询公司考勤与假期相关的政策、计算规则或资格条件，调用工具转至政策查询代理。
   - **系统技术问题**：如果用户遇到假期系统的登录故障、申请提交报错、页面异常等技术问题，调用工具转至系统支持代理。

2. **工具调用原则**：仅在必要时调用工具，确保调用格式正确且参数完整（当前场景下参数为空对象）。直接回答的问题无需调用任何工具。
"""


# 分诊代理 (Triage)
triage_agent = SwarmAgent(
    name="分诊代理",
    instructions=triage_prompt,
    functions=[
        transfer_to_leave_agent, transfer_to_status_query_agent, transfer_to_leave_manage_agent, 
        transfer_to_policy_query_agent, transfer_to_system_support_agent, escalate_to_human
    ],
)

# 假期申请代理-工具集
leave_map = {
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
leave_tools = [x for x in tools_list if x['name'] in leave_map]

# 假期管理代理-工具集
leave_manage_map = {
    "考勤-撤销请假": "attendance_cancel_leave_request",
    "考勤-销假": "attendance_end_leave_early",
}
leave_manage_tools = [x for x in tools_list if x['name'] in leave_manage_map]

# 状态查询代理-工具集
status_query_map = {
    "考勤-查询请假记录": "attendance_get_leave_records",
    "考勤-查询跨国工作探亲假": "attendance_get_overseas_family_visit_leave",
    "考勤-查询育儿假": "attendance_get_childcare_leave_balance",
    "考勤-查询年假": "attendance_get_annual_leave_balance",
    "考勤-查询销假记录": "attendance_get_early_leave_ending_records",
    # "考勤-查询员工信息": "attendance_get_employee_info",
    "考勤-查询员工考勤日报": "attendance_get_employee_daily_report",
    "考勤-查询员工排班": "attendance_get_employee_schedule",
}
status_query_tools = [x for x in tools_list if x['name'] in status_query_map]

# 政策查询代理-工具集
policy_query_map = {
    "考勤-查询离职年假计算规则": "attendance_get_resignation_annual_leave_rules",
    "考勤-查询年假计算规则": "attendance_get_annual_leave_rules",
}
policy_query_tools = [x for x in tools_list if x['name'] in policy_query_map]

# 系统支持代理-工具集
system_support_map = {
    "考勤-页面访问记录": "attendance_get_page_access_log",
    "考勤-查询年假申请界面天数显示有误原因": "attendance_get_reason_for_leave_days_display_error"
}
system_support_tools = [x for x in tools_list if x['name'] in system_support_map]


def zh2en(tool_zh_name):
    if tool_zh_name in leave_map:
        return leave_map[tool_zh_name]
    if tool_zh_name in leave_manage_map:
        return leave_manage_map[tool_zh_name]
    if tool_zh_name in status_query_map:
        return status_query_map[tool_zh_name]
    if tool_zh_name in policy_query_map:
        return policy_query_map[tool_zh_name]
    if tool_zh_name in system_support_map:
        return system_support_map[tool_zh_name]


worker_prompt = """
你是理想汽车的人力助手，负责{{task}}。你的所有回答必须严格控制在一句话内。

当前日期：
<current_date>
{{cur_time}}
</current_date>

员工基本信息：
<employee_details>
{{employee_info}}
</employee_details>

处理用户假期申请时，请严格遵循以下流程与规则：
1. 若用户未在查询中明确说明假期类型（如年假/病假/事假等），必须首先询问确认假期类型。
2. 确认假期类型后，需根据类型提醒用户提供相关证明材料（例如：病假需附医疗证明，事假需说明具体事由）。
3. 必须使用提供的工具处理问题，对于函数参数的取值，不得擅自假设任何信息。若用户的需求表述存在歧义或关键信息缺失，务必主动询问以明确所有细节。

请根据上述规则和信息，生成一句回答。
"""


# 假期申请代理
leave_agent = SwarmAgent(
    name="假期申请代理",
    instructions=Template(worker_prompt).render(
        task="处理员工各类假期申请的工作，包括但不限于年假、病假、事假、丧假等",
        cur_time=str(cur_time),
        employee_info=str(employee_info)
    ),
    functions=leave_tools + [transfer_back_to_triage],
)

# 状态查询代理
status_query_agent = SwarmAgent(
    name="状态查询代理",
    instructions=Template(worker_prompt).render(
        task="处理员工考勤状态查询的工作，包括但不限于考勤记录、假期余额、排班信息等",
        cur_time=str(cur_time),
        employee_info=str(employee_info)
    ),
    functions=status_query_tools + [transfer_back_to_triage],
)

# 假期管理代理
leave_manage_agent = SwarmAgent(
    name="假期管理代理",
    instructions=Template(worker_prompt).render(
        task="处理各类假期管理的工作，包括但不限于撤销申请、提前结束假期等",
        cur_time=str(cur_time),
        employee_info=str(employee_info)
    ),
    functions=leave_manage_tools + [transfer_back_to_triage],
)

# 政策查询代理
policy_query_agent = SwarmAgent(
    name="政策查询代理",
    instructions=Template(worker_prompt).render(
        task="处理政策查询的工作，包括但不限于各类假期的计算规则、资格条件等",
        cur_time=str(cur_time),
        employee_info=str(employee_info)
    ),
    functions=policy_query_tools + [retrieve_knowledge_with_rag] + [transfer_back_to_triage],
)

# 系统技术支持代理
system_support_agent = SwarmAgent(
    name="系统支持代理",
    instructions=Template(worker_prompt).render(
        task="处理系统技术支持代理工作，包括但不限于登录故障、申请提交报错、页面异常等",
        cur_time=str(cur_time),
        employee_info=str(employee_info)
    ),
    functions=system_support_tools + [transfer_back_to_triage],
)


# 5、代理注册与启动
agents = {agent.name: agent for agent in [triage_agent, leave_agent, leave_manage_agent, status_query_agent, policy_query_agent, system_support_agent]}

messages: List[ChatMessage] = []
current_agent_name = "分诊代理"  # 初始代理为中控

# 6、创建 Workflow
# 循环处理直到 LLM 停止调用工具
while True:
    agent = agents[current_agent_name]

    # 用户输入环节（仅当需要用户回复时）
    if not messages or messages[-1].role == ChatRole.ASSISTANT:
        user_input = input("用户：")  # 中文输入提示
        if user_input.lower() == "quit": break
        if user_input.strip() == "": continue
        messages.append(ChatMessage.from_user(user_input))

    # # 准备 System Message (包含当前 Agent 的指令)
    # sys_msg = ChatMessage.from_system(
    #     f"当前角色: {current_agent_name}\n"
    #     f"指令: {current_agent.instructions}\n"
    #     f"当前时间: {datetime.now().strftime('%Y-%m-%d')}"
    # )
    
    # 代理处理与状态更新
    current_agent_name, agent_msg = agent.run(messages)
    messages.extend(agent_msg)
