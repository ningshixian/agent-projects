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
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Literal, Annotated
from dotenv import load_dotenv

from haystack.components.agents import Agent
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator, OpenAIChatGenerator
# from haystack.components.websearch.searchapi import SearchApiWebSearch
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import merge_lists, replace_values
from haystack.tools import create_tool_from_function    #
from haystack.tools import Tool
from haystack.components.tools import ToolInvoker

load_dotenv()   # 加载环境变量 (.env 文件)
cur_time = (datetime.now().strftime("%Y-%m-%d"))

# =================配置区域=================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

TOOL_API_URL = os.getenv("TOOL_API_URL", "")
TOOL_API_TOKEN = os.getenv("TOOL_API_TOKEN", "Bearer YOUR_TOKEN")
TOOL_RAG_URL = "..."

# =================预设工具名映射 map=================
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
    "考勤-请返乡假": "attendance_apply_for_fanxiang_leav", 
}
leave_manage_map = {
    "考勤-撤销请假": "attendance_cancel_leave_request",
    "考勤-销假": "attendance_end_leave_early",
}
status_query_map = {
    "考勤-查询请假记录": "attendance_get_leave_records",
    "考勤-查询跨国工作探亲假": "attendance_get_overseas_family_visit_leave",
    "考勤-查询育儿假": "attendance_get_childcare_leave_balance",
    "考勤-查询年假": "attendance_get_annual_leave_balance",
    "考勤-查询销假记录": "attendance_get_early_leave_ending_records",
    "考勤-查询员工信息": "attendance_get_employee_info",
    "考勤-查询员工考勤日报": "attendance_get_employee_daily_report",
    "考勤-查询员工排班": "attendance_get_employee_schedule",
    "考勤-查询返乡假": "attendance_get_fanxiang_leave", 
}
policy_query_map = {
    "考勤-查询离职年假计算规则": "attendance_get_resignation_annual_leave_rules",
    "考勤-查询年假计算规则": "attendance_get_annual_leave_rules",
}
system_support_map = {
    "考勤-页面访问记录": "attendance_get_page_access_log",
    "考勤-查询年假申请界面天数显示有误原因": "attendance_get_reason_for_leave_days_display_error"
}

# 中文工具名转英文
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
    return tool_zh_name


def get_tool_list():
    # 获取tools——list
    tools_list = []
    headers = {"Authorization": TOOL_API_TOKEN, "Origin":"chehejia.com"}
    data = {}
    source = 1101
    try:
        response = requests.post(TOOL_API_URL+"?"+"source="+str(source), headers=headers, data=json.dumps(data), timeout=10)
        if response.status_code == 200:
            print("工具列表获取成功: 工具个数", len(json.loads(response.text)['data']))  # 28
            tools_list = json.loads(response.text)['data']
        else:
            tools_list = [{"请求出错":json.loads(response.text)}]
    except Exception as err:
        print(f'An error occurred: {err}')
    
    # 【考勤-页面访问记录】这个工具的 schema 不太对
    tools_list = [tool for tool in tools_list if tool["name"] != "考勤-页面访问记录"]   # 27

    for tool_name in tools_list:
        if tool_name is not None:
            print(tool_name["name"])
    
    return tools_list


class ToolUtils:
    @staticmethod
    def fix_json_schema(schema: Any):
        """递归修正非标准的 JSON Schema 类型"""
        if isinstance(schema, dict):
            for key, value in schema.items():
                if key == "type":
                    if value == "text": schema[key] = "string"
                    elif value == "decimal": schema[key] = "number"
                    elif value == "int": schema[key] = "integer"
                    elif value in ("date", "datetime"): schema[key] = "string"
                else:
                    ToolUtils.fix_json_schema(value)
            # 处理 ai_required
            if "ai_required" in schema:
                schema['required'] = schema['ai_required']
        
        elif isinstance(schema, list):
            for item in schema:
                ToolUtils.fix_json_schema(item)


# 函数映射
class func_map:
    def __init__(self, tool_name, scope):
        self.tool_name = tool_name
        self.scope = scope

    def tool_implement(self, **params):
        tools_exec = []
        headers = {
            "Authorization": TOOL_API_TOKEN,
            "Content-Type":"application/json",
            "Origin":"chehejia.com"
        }

        data = {
            "tool_name": self.tool_name,
            "idaas_open_id": "3OqQ0gs3YuwUveqOylMjiw",
            "scope_description": self.scope,
            "params": params
        }

        try:
            response = httpx.post(TOOL_API_URL, json=data, headers=headers, timeout=10)
            response.raise_for_status()
            tools_exec = response.json()
            print(tools_exec)
            result = tools_exec.get("data", {}).get("data", "工具执行成功但无返回数据")
            return result
        except Exception as e:
            print(f'An error occurred: {e}')
            return f"工具执行出错，请重新检查: {str(e)}"


# 获取工具列表
tools_list = get_tool_list()
# schema类型转换
for tool in tools_list:
    if "inputSchema" in tool:
        ToolUtils.fix_json_schema(tool["inputSchema"])

# # 1、通过工具获取员工个人信息（user_info_tool）
# first_tool = next(tool for tool in tools_list if tool["name"] == "user_info_tool")
# tools_list.pop(tools_list.index(first_tool))
# print("工具个数", len(tools_list))  # 26
# tool = Tool(
#     name=first_tool["name"],
#     description=first_tool["description"],
#     parameters=first_tool["inputSchema"],
#     function=func_map("user_info_tool").tool_implement,
# )
# result = tool.invoke(fields=[])
# employee_info = result['template']['data']['form_data']
# # 打印结构化信息
# for key, value in employee_info.items():
#     print(f"{key}: {value}")

schema = {
    "cur_time": {"type": str},
    "employee_info": {"type": dict},
}
state = State(schema=schema)
state.set("cur_time", cur_time)
state.set("employee_info", {
    "职级": "16",
    "社保城市名称": "北京市",
    "性别": "男",
    "入职日期": "2025-06-30",
    "工龄（年）": "1.1年",
    "工作城市名称": "北京市",
    "姓名": "张*涛",
    "工号": "001898",
    "员工组名称": "白领",
    "部门名称": "组织系统"}
)

# 2、通过个人信息和Q进行工具调用

# 函数封装成工具类 Tool
toolset = []
for _tool in tools_list:
    tool_func = func_map(_tool["name"], _tool["scope_description"]).tool_implement
    ok_tool = Tool(
        name=zh2en(_tool["name"]),  # 必须是英文数字_-
        description=_tool["description"],
        parameters=_tool["inputSchema"],    # {"type": "object", "properties": {}, "required": []}
        function=tool_func,
        # inputs_from_state=..., 
        # outputs_to_state=...,
    )
    toolset.append(ok_tool)

# Create the agent with the web search tool
generator = OpenAIChatGenerator(
    model='deepseek-chat',
    api_base_url=OPENAI_API_BASE,
    # api_key=Secret.from_token(OPENAI_API_KEY),
    # generation_kwargs={"temperature": 0.5},
    timeout=60, 
    max_retries=2,
    streaming_callback=print_streaming_chunk
)
agent = Agent(
    chat_generator=generator, 
    tools=toolset, 
    system_prompt=f"""你是人力考勤助手。当前时间: {state.get("cur_time")}\n员工基本信息：{state.get("employee_info")}""",
    # state_schema=schema, 
    # exit_conditions=["text"],    # List of conditions that will cause the agent to return.
    # max_agent_steps=2,            # Maximum number of steps the agent will run before stopping.
    # raise_on_tool_invocation_failure=True
)
agent.warm_up()
# agent.to_dict()

# Run the agent with a query
user_message = ChatMessage.from_user("我的年假还有多少天？")
result = agent.run(messages=[user_message])

# tool_result

for message in result["messages"]:
    print(f"{message.role}: {message.text}")
