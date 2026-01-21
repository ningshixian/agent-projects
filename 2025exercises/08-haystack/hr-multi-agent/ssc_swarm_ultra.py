# Building a agentic RAG with Function Calling
# https://colab.research.google.com/github/deepset-ai/haystack-tutorials/blob/main/tutorials/40_Building_Chat_Application_with_Function_Calling.ipynb#scrollTo=ZE0SEGY92GHJ
import os
import sys
import re
import json
import logging
import traceback
import inspect
import requests
import httpx
from typing import List, Dict, Any, Optional, Callable, Literal, Annotated
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from dataclasses import replace  # <--- 必须导入这个

from jinja2 import Template
from pydantic import BaseModel, Field, create_model, field_validator
from dotenv import load_dotenv
from utils import safe_parse_json, get_current_time_str

# Haystack 核心组件
from haystack import Document, Pipeline
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import create_tool_from_function    #
from haystack.tools import Tool
from haystack.components.tools import ToolInvoker

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SwarmSystem")

load_dotenv()   # 加载环境变量 (.env 文件)
cur_time = (datetime.now().strftime("%Y-%m-%d"))

"""
利用 Haystack 框架构建一个基于 Orchestrator-Workers 架构（中控-分诊模式）的智能人力助手。
系统集成了动态工具加载、多智能体协作、全局状态管理及自愈式 ReAct 循环。

1. 资源初始化与工具构建
  ○ 从 API 动态拉取工具列表，递归清洗 JSON Schema 以适配 LLM。
  ○ 注入 RAG 检索工具 用于处理非结构化政策查询。
  ○ 实例化 UserProfile，加载员工基础信息。
2. 智能体编排 (Agent Orchestration)
  ○ 定义 TriageAgent (中控)：配置路由 Prompt，负责意图识别与分发（Transfer Tools）。
  ○ 定义 Worker Agents (垂类专家)：如 LeaveAgent, StatusQueryAgent 等，配置专属 System Prompt（注入员工画像与时间）及领域工具集。
  ○ 封装 SwarmAgent 类：实现 ReAct 循环、错误自修正 (Reflection) 及工具调用。
3. 全局状态管理 (Global Context)
  ○ 初始化 GlobalContext，用于在不同 Agent 之间透传：
    ■ slots: 跨轮次累积的关键业务参数（如请假类型、时间）。
    ■ chat_history: 全量对话历史。
    ■ user_profile: 员工身份信息。
4. 主工作流循环 (Main Execution Loop)
  ○ Step 4.1 上下文注入：根据当前活跃 Agent，动态构建 Prompt，注入不同的全局状态。
  ○ Step 4.2 推理与决策：LLM 生成回复或工具调用请求 (tool_calls)。
  ○ Step 4.3 槽位收割：关键步骤，在执行工具前，拦截 tool_calls 参数并更新至全局 slots，实现被动信息抽取。
  ○ Step 4.4 路由与执行：
    ■ 若为 转接指令 (Transfer)：更新 current_agent 指针，生成摘要，切换至新 Agent。
    ■ 若为 业务工具 (Function)：ToolInvoker 执行 API 请求，并将工具的结果整合回推理链中，进行下一轮生成。
    》工具调用流程：tool calls + parameters → ToolInvoker→ToolFunction
  ○ Step 4.5 循环闭环：更新对话历史，等待用户下一轮输入。

补充：工具集 List[Tool] 以及执行器 ToolInvoker
    - 工具类ToolFunction封装了工具的调用逻辑，通过HTTP POST请求调用外部API。
    - 每个工具都有一个名称、描述、参数和执行函数。
    - 工具调用器ToolInvoker管理多个工具实例，并负责调用它们。
"""


# =================配置区域=================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

TOOL_API_URL = os.getenv("TOOL_API_URL", "")
TOOL_API_TOKEN = os.getenv("TOOL_API_TOKEN", "Bearer YOUR_TOKEN")
TOOL_RAG_URL = "..."

# 中文转接模板与匹配模式
HANDOFF_TEMPLATE = "已转接至：{agent_name}。请立即切换角色。"
HANDOFF_PATTERN = r"已转接至：(.*?)(?:。|$)"  # 匹配中文句号


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
    # "考勤-查询员工信息": "attendance_get_employee_info",
    "考勤-查询员工考勤日报": "attendance_get_employee_daily_report",
    "考勤-查询员工排班": "attendance_get_employee_schedule",
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


# --- 1. 全局状态管理 (State Management) ---
# 解决上下文污染和Token消耗，实现“按需共享”

class UserProfile(BaseModel):
    name: str = Field(alias="姓名", default="Unknown")
    employee_id: str = Field(alias="工号", default="Unknown")
    gender: Optional[str] = Field(alias="性别", default=None)
    # --- 职场信息 ---
    department: str = Field(alias="部门名称", default="Unknown")
    title_level: str = Field(alias="职级", default="Unknown")
    employee_group: str = Field(alias="员工组名称", default="Unknown")
    # --- 工作信息 ---
    location: str = Field(alias="工作城市名称", default="Unknown")
    social_security_city: Optional[str] = Field(alias="社保城市名称", default=None)
    join_date: Optional[str] = Field(alias="入职日期", default=None)
    tenure: float = Field(alias="工龄（年）", default=0.0)

    # --- 数据清洗验证器 ---
    @field_validator('tenure', mode='before')
    @classmethod
    def parse_tenure(cls, v):
        """
        处理 '1.1年' 格式，移除 '年' 并转换为 float
        """
        if isinstance(v, str) and v.endswith("年"):
            return float(v.replace("年", ""))
        return v


class GlobalContext(BaseModel):
    """全局状态对象，存储跨Agent的结构化数据"""
    user_profile: UserProfile = Field(default_factory=UserProfile)
    # 槽位：存储所有提取到的关键信息，key为参数名，value为值
    slots: Dict[str, Any] = Field(default_factory=dict)
    # 全量对话历史
    chat_history: List[ChatMessage] = Field(default_factory=list)
    # 对话历史摘要 (而非全量历史)
    conversation_summary: str = ""
    # 工具执行结果 (用于 Agent 间传递)
    last_tool_output: Optional[str] = None
    # 路由控制
    next_agent: str = "TriageAgent"     # 指针，决定下一个由谁接管

    def update_slots(self, new_slots: Dict[str, Any]):
        """
        智能更新槽位：
        1. 仅更新非空值
        2. 可以扩展逻辑：例如遇到冲突时保留最新的，或者保留更长的
        """
        if not new_slots: return
        # 过滤空值并更新
        cleaned = {k: v for k, v in new_slots.items() if v not in [None, ""]}
        self.slots.update(cleaned)


class AgentName(str, Enum):
    """枚举所有 Agent 名称，防止字符串硬编码错误"""
    TRIAGE = "TriageAgent"
    LEAVE = "LeaveAgent"
    STATUS = "StatusQueryAgent"
    MANAGE = "LeaveManageAgent"
    POLICY = "PolicyQueryAgent"
    SUPPORT = "SystemSupportAgent"

# 单独定义描述字典
AGENT_TASK_DESC = {
    AgentName.TRIAGE: "用户意图识别与任务分发",
    AgentName.LEAVE: "处理员工各类假期申请的工作，包括但不限于年假、病假、事假、丧假等",
    AgentName.STATUS: "处理员工考勤状态查询的工作，包括但不限于考勤记录、假期余额、排班信息等",
    AgentName.MANAGE: "处理各类假期管理的工作，包括但不限于撤销申请、提前结束假期等",
    AgentName.POLICY: "处理考勤与假期政策查询的工作，包括但不限于各类假期的计算规则、资格条件等",
    AgentName.SUPPORT: "处理系统技术支持代理工作，包括但不限于申请提交报错、年假申请界面天数显示有误、页面异常等",
}


# --- 2. 安全与风控层 (Safety Layer) ---

class SafetyGuard:
    """敏感词正则检测与合规风控"""
    SENSITIVE_PATTERNS = [
        r"(薪资|工资|薪酬|待遇).*(查询|看|多少)",
        r"(高管|CEO|VP).*(行程|住址|电话)",
        r"代查.*(考勤|打卡)"
    ]

    @staticmethod
    def check(text: str) -> bool:
        for pattern in SafetyGuard.SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True


# --- 3. 工具基础设施 (Tool Infrastructure) ---
# 将复杂的 Schema 递归清洗和 API 调用逻辑封装。

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


class ToolFactory:
    """工具工厂：负责从 API 加载和转换工具"""
    
    @staticmethod
    def create_api_tool(config: dict) -> Tool:
        tool_name = config["name"]
        scope = config.get("scope_description", "")
        # api_url = config.get("api_url", TOOL_API_URL)
        
        # 闭包函数捕获配置
        def api_executor(**kwargs):
            logger.info(f"[Tool Call] {tool_name} Args: {kwargs}")
            headers = {
                "Authorization": TOOL_API_TOKEN,
                "Content-Type": "application/json",
                "Origin":"chehejia.com"
            }
            payload = {
                "tool_name": tool_name,                    # 要执行的工具名称
                "idaas_open_id": "3OqQ0gs3YuwUveqOylMjiw",      # 员工ID
                "scope_description": scope,                     # 工具作用域范围 [0, 1101]
                "params": kwargs                                # 需要提取的所有参数 (ai_required 里对应的参数)
            }

            try:
                # 使用 httpx 同步调用 (Haystack Tool 目前多为同步)
                resp = httpx.post(TOOL_API_URL, json=payload, headers=headers, timeout=10)
                # resp = httpx.post(api_url, headers=headers, data=json.dumps(data), timeout=5)
                resp.raise_for_status()
                data = resp.json()
                return data.get("data", {}).get("data", "工具执行成功但无返回数据")
            except Exception as e:
                logger.error(f"{traceback.format_exc()}")
                logger.error(f"{tool_name} with params {payload} 工具执行失败 or 参数错误！")
                return f"工具执行出错，请重新检查: {str(e)}"

        return Tool(
            name=zh2en(config["name"]), # 建议此处做英文名映射
            description=config["description"],
            parameters=config["inputSchema"],
            function=api_executor
        )
    
    @staticmethod
    def fetch_tools_from_remote() -> List[dict]:
        """模拟/实际从远程获取工具列表"""
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
            print(f'An error occurred: {err}')
        
        # 【考勤-页面访问记录】这个工具的 schema 不太对
        tools_list = [tool for tool in tools_list if tool["name"] != "考勤-页面访问记录"]   # 27

        # schema类型转换，确保兼容性
        for tool in tools_list:
            if "inputSchema" in tool:
                ToolUtils.fix_json_schema(tool["inputSchema"])

        # # 保存json文件
        # with open('tools_list.json', 'w', encoding='utf-8') as f:
        #     json.dump(tools_list, f, ensure_ascii=False, indent=4)

        # # 输出所有可用工具名称
        # for tool_name in tools_list:
        #     if tool_name is not None:
        #         print(tool_name["name"])
        return tools_list


# --- 4. 核心 Agent 类 (SwarmAgent) (ReAct + Reflection) ---
# 构建Prompt -> LLM推理 -> 处理回复(槽位/工具) -> Reflection。


@dataclass
class SwarmAgent:
    name: str
    llm: Any  # OpenAIChatGenerator
    tools: List[Any]  # List[Tool]
    system_prompt_template: str  # Jinja2 模板字符串

    # dataclass 自动生成 __init__(self, name, llm, tools,...)

    def __post_init__(self):    # 做一些额外的初始化
        for i, tool in enumerate(self.tools):
            if inspect.isfunction(tool):  # def
                self.tools[i] = create_tool_from_function(tool)
            elif isinstance(tool, dict):   # tool
                self.tools[i] = ToolFactory.create_api_tool(tool)
            elif isinstance(tool, Tool):
                pass
            else:
                print("！！！！！！！", tool)
        
        # 创建工具调用器
        self.tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False) if self.tools else None
        # 自修正最大尝试次数
        self.max_reflection_steps = 2
    
    def _render_system_msg(self, context: GlobalContext):
        """
        [Helper] 负责根据当前 Context 动态渲染 System Prompt
        """
        template = Template(self.system_prompt_template)
        sys_content = template.render(
            task=AGENT_TASK_DESC[self.name], 
            cur_time=get_current_time_str(),
            user_profile=context.user_profile.model_dump(), # 注入用户信息
            slots=context.slots  # 已知槽位
        )
        return ChatMessage.from_system(sys_content)
    
    def _build_messages(self, context: GlobalContext) -> List[ChatMessage]:
        """动态构建消息"""

        # 1. 初始渲染 System Prompt
        sys_prompt = self._render_system_msg(context)

        # 2. 构建消息列表
        messages = [sys_prompt]
        valid_history = [m for m in context.chat_history if m.role != ChatRole.SYSTEM]

        # Triage 看摘要 + 最近 3 轮原始对话
        # Worker 看滑动窗口原始对话，但要注意话题切换时的截断。
        if self.name == AgentName.TRIAGE:
            if context.conversation_summary:
                # 可以在这里插入一个 Summary 消息
                messages[-1] += f"\n对话摘要：\n{context.conversation_summary}"
            messages.extend(valid_history[-3:]) 
        else:
            # Short-term Memory 保留所有细节参数（日期、数字），防止丢失关键信息
            messages.extend(valid_history[-10:])

        return messages
    
    def run(self, context: GlobalContext):
        """执行 ReAct 循环：
        Reason -> Act -> Observe -> Reflect(若出错) -> Final Answer
        返回: {'response': str, 'next_agent': str | None, 'messages': List[ChatMessage]}
        """
        messages = self._build_messages(context)
        initial_msg_count = len(messages)   # 记录初始长度，用于最后切片返回新增的消息

        print(f"\n[调试] {self.name} Agent, LLM Messages:")
        for m in messages:
            if m.text:
                log_content = m.text
            elif m.tool_calls:
                # 如果是 Assistant 发起的工具调用 (Tool Calls)
                tool_names = [tc.tool_name for tc in m.tool_calls]
                log_content = f"👉 [Calling Tools]: {', '.join(tool_names)}"
            elif m.tool_call_results:
                # 如果是 Tool 返回的结果 (Tool Result) <-- 修复这里的报错
                results = [str(tr.result) for tr in m.tool_call_results]
                log_content = f"✅ [Tool Result]: {', '.join(results)}"

            msg = str(log_content).replace("\n", " ")[:100]
            print(f"  - [{m.role.value}]: {msg}...") # 打印前100字

        # 安全检查 SafetyGuard
        
        # ReAct Loop (最大尝试 3 次)
        for step in range(self.max_reflection_steps + 1):
            # 1. LLM 生成
            try:
                # print("===", messages)
                response = self.llm.run(messages=messages, tools=self.tools)
                agent_msg = response["replies"][0]
                messages.append(agent_msg)
            except Exception as e:
                logger.error(f"LLM 调用失败: {traceback.format_exc()}")
                return {"response": "系统繁忙，请重试。", "next_agent": None, "new_messages": []}

            # logger.info(f"[调试 {self.name}] Output: {agent_msg.text[:50]}... ToolCalls: {len(agent_msg.tool_calls)}")

            # 2. 提取并更新槽位信息 (无论后续是否调用工具)
            new_slots, json_str = safe_parse_json(agent_msg.text)
            if json_str:
                new_slots = new_slots.get("slot_update")
                print(f"  [Slot Update] {new_slots}")
                # A. 更新全局状态
                context.update_slots(new_slots)
                # B. 【关键步骤】热更新 System Prompt (Hot Reload)
                messages[0] = self._render_system_msg(context)
                # # C. 模拟一条 Tool Output，明确告诉模型槽位已更新
                # slot_feedback_msg = ChatMessage(
                #     role=ChatRole.TOOL, # 伪装成工具返回
                #     text=f"[System Notification] Memory updated. Current Slots: {json.dumps(context.slots, ensure_ascii=False)}",
                #     tool_calls=[]
                # )
                # messages.append(slot_feedback_msg)

                # 清洗消息文本，移除 JSON 块，保留对用户的自然语言回复
                clean_text = agent_msg.text.replace(json_str, "").strip()
                # 重新构建消息
                new_msg = ChatMessage.from_assistant(
                    text=clean_text,
                    tool_calls=agent_msg.tool_calls,  # 重要：保留工具调用信息！
                    meta=agent_msg.meta               # 保留元数据
                )
                messages[-1] = new_msg        # 在列表中替换为新对象
                agent_msg = new_msg           # 更新当前引用

            # 3. 判断是否包含工具调用
            if not agent_msg.tool_calls:
                # Case A: 纯文本回复 (无工具调用)，结束
                logger.info("  →Case A: 直接回复 (无工具调用)")
                return {
                    "response": agent_msg.text, 
                    "next_agent": None, 
                    "new_messages":messages[initial_msg_count:]
                }

            # Case B: 处理工具调用
            tool_results = []
            for tc in agent_msg.tool_calls:
                # B-1 拦截转接指令 (Handoff)
                if tc.tool_name.startswith("transfer_to_"):
                    next_agent_candidate = tc.tool_name.replace("transfer_to_", "")
                    logger.info(f"  →Case B-1: 拦截转接指令 {next_agent_candidate}")

                    # 构造转接消息存入历史
                    # 必须为 TOOL 类型的 Msg，才能保证Function Calling 闭环 [Assistant(Call), Tool(Result)]
                    handoff_msg = ChatMessage.from_tool(f"正在为您转接至 {next_agent_candidate}...", origin=tc)    # tool_calls=[]
                    messages.append(handoff_msg)

                    # 转接时，不需要执行实际工具，直接返回控制权
                    return {
                        "response": f"正在为您转接至 {next_agent_candidate}...", 
                        "next_agent": next_agent_candidate, 
                        "new_messages": messages[initial_msg_count:]
                    }
                
                # B-2 执行业务工具
                try:
                    logger.info(f"  →Case B-2: 执行业务工具 Invoking {tc.tool_name}...")
                    res = self.tool_invoker.run(messages=[agent_msg])
                    tool_msg = res["tool_messages"][0]
                    tool_results.append(tool_msg)
                    # context.last_tool_output = tool_msg.text
                except Exception as e:
                    # Reflection: 将错误写回消息列表，让模型自修正
                    print(f"  [Reflection] Tool Error: {e}. Requesting fix...")
                    error_msg = ChatMessage.from_tool(f"Tool execution failed: {str(e)}. Please correct arguments and retry.", origin=tc)
                    tool_results.append(error_msg)

            if tool_results:
                messages.extend(tool_results)
                # 循环继续，LLM 看到 Tool Result 后会进行下一步生成
            else:
                # 如果有 tool_calls 但没产生 tool_results (极端情况)，避免死循环
                if step == self.max_reflection_steps:
                    return {"response": "系统暂时无法处理该请求，请稍后重试或联系人工。", "next_agent": None, "new_messages":[]}

        return {
            "response": messages[-1].text, 
            "next_agent": None, 
            "new_messages": messages[initial_msg_count:] 
        }


# --- 5. 编排系统 (Orchestrator System) ---


class SwarmSystem:
    def __init__(self):
        self.llm = OpenAIChatGenerator(
            model='deepseek-chat',
            api_base_url=OPENAI_API_BASE,
            # api_key=Secret.from_token(OPENAI_API_KEY),
            generation_kwargs={"temperature": 0.5},
            timeout=60, 
            max_retries=2,
        )
        self.agents = {}      # Agent 注册表
        self._init_agents()

    def _init_agents(self):
        from prompt import triage_prompt, worker_prompt

        # 加载转接工具 (Triage 专用)
        TRANS_TOOL_DESC = {
            AgentName.LEAVE: "转接给 LeaveAgent，处理员工各类假期申请的工作，包括但不限于年假、病假、事假、丧假等",
            AgentName.STATUS: "转接给 StatusQueryAgent，处理员工考勤状态查询的工作，包括但不限于考勤记录、假期余额、排班信息等",
            AgentName.MANAGE: "转接给 LeaveManageAgent，处理各类假期管理的工作，包括但不限于撤销申请、提前结束假期等",
            AgentName.POLICY: "转接给 PolicyQueryAgent，处理考勤与假期政策查询的工作，包括但不限于各类假期的计算规则、资格条件等",
            AgentName.SUPPORT: "转接给 SystemSupportAgent，处理系统技术支持代理工作，包括但不限于申请提交报错、年假申请界面天数显示有误、页面异常等"
        }
        transfer_tools = [
            Tool(
                name=f"transfer_to_{name}",
                description=desc,
                parameters={"type": "object", "properties": {}, "required": []},    # inputSchema
                function=lambda: f"Transferred to {name}"
            ) for name, desc in TRANS_TOOL_DESC.items()
        ]

        # 初始化 Triage Agent
        self.agents[AgentName.TRIAGE] = SwarmAgent(
            name=AgentName.TRIAGE,
            llm=self.llm,
            tools=transfer_tools, # + human_escalate
            system_prompt_template=triage_prompt
        )

        # 加载worker工具
        # 从指定 API 接口动态拉取可用的工具列表
        worker_tools = ToolFactory.fetch_tools_from_remote()
        WORKER_TOOL_DICT = {}
        # 假期申请代理-工具集
        WORKER_TOOL_DICT[AgentName.LEAVE] = [x for x in worker_tools if x['name'] in leave_map]
        # 假期管理代理-工具集
        WORKER_TOOL_DICT[AgentName.MANAGE] = [x for x in worker_tools if x['name'] in leave_manage_map]
        # 状态查询代理-工具集
        WORKER_TOOL_DICT[AgentName.STATUS] = [x for x in worker_tools if x['name'] in status_query_map]
        # 政策查询代理-工具集
        WORKER_TOOL_DICT[AgentName.POLICY] = [x for x in worker_tools if x['name'] in policy_query_map]
        # 系统支持代理-工具集
        WORKER_TOOL_DICT[AgentName.SUPPORT] = [x for x in worker_tools if x['name'] in system_support_map]
        
        # 额外定义一个跳转工具
        transfer_back_to_triage = {
            "name": "transfer_to_TriageAgent", 
            "description": "当前任务执行完成，或者需要转接至其他代理继续执行任务，调用此工具",
            "inputSchema": {"type": "object", "properties": {}, "required": []}, 
            "scope_description": "0,1101"
        }

        # 初始化 Worker Agents
        for name, tools in WORKER_TOOL_DICT.items():
            self.agents[name] = SwarmAgent(
                name=name,
                llm=self.llm,
                tools=tools + [transfer_back_to_triage],
                system_prompt_template=worker_prompt, 
            )
            # fix：PolicyQueryAgent
            self.agents[AgentName.POLICY] = SwarmAgent(
                name=AgentName.POLICY,
                llm=self.llm,
                tools=WORKER_TOOL_DICT[AgentName.POLICY] + [transfer_back_to_triage] + [retrieve_knowledge_with_rag],
                system_prompt_template=worker_prompt,
            )
    
    def run_turn(self, user_input: str, context: GlobalContext) -> str:
        """
        入口函数：处理单次用户请求

        1、Agent 激活：根据 next_agent 调度对应 Agent。
        2、执行agent.run()开始 ReAct Loop，追加新产生的 User/Assistant 消息到全局历史。
        3、处理 Agent 切换（可选）
            ⅰ. if Back to Triage: 清空业务参数 slots
            ⅱ. if 转接任务 Agent：继承状态
        4、返回 response
        """
        logger.info(f"--- User Input: {user_input} ---")

        # 1. 将用户输入添加全局历史
        context.chat_history.append(ChatMessage.from_user(user_input))

        # 2. 如果是新会话，默认从 Triage 开始；如果是多轮对话，可能维持在上一个 Agent
        # 这里可以加入逻辑：如果用户说“退出”、“返回主菜单”，强行重置为 Triage
        if any(x in user_input for x in ["退出", "返回", "取消"]):
            context.next_agent = AgentName.TRIAGE
            # 策略：返回主菜单时是否清空槽位？视业务而定，通常建议清空
            context.slots = {}
        
        max_handoffs = 3 # 防止无限转接死循环
        handoff_count = 0
        final_response = ""

        while handoff_count < max_handoffs:
            current_agent_name = context.next_agent or AgentName.TRIAGE
            agent = self.agents.get(current_agent_name)

            if not agent:
                # 容错：如果找不到 Agent，回退到 Triage
                logger.error(f"Agent {current_agent_name} not found. Fallback to Triage.")
                agent = self.agents[AgentName.TRIAGE]
            
            logger.info(f"--- Activating Agent: {agent.name} ---")

            # 执行 Agent
            result = agent.run(context)

            # # 打印当前槽位状态，方便调试
            # print(f"\n[Global State] Agent: {current_agent_name} | Slots: {context.slots}")

            # 更新历史 (追加 Agent 产生的思考过程、工具结果、回复)
            if result["new_messages"]:
                context.chat_history.extend(result["new_messages"])
            
            final_response = result["response"]
            next_agent = result["next_agent"]

            # 判断是否发生转接
            if next_agent and next_agent != current_agent_name:
                # 发生了转接 -> 更新状态，进入下一次循环
                print(f"\n🔄 Handoff: {context.next_agent} -> {next_agent} 🔄")
                context.next_agent = next_agent
                handoff_count += 1
                
                # === 关键生命周期管理 ===
                # 如果从 Triage 转出，通常意味着新任务开始
                # 如果从 Worker 转回 Triage，意味着任务结束
                # 这里可以插入逻辑来更新 Summary，或截断 History
                continue # 立即进入下一个 Agent 的处理流程
            else:
                # 没有转接，或者是转接给自己（通常意味着等待用户输入），结束本轮
                break
        
        return final_response


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


if __name__ == "__main__":
    # # 调用工具（考勤-查询员工信息），结果保存在 employee_info
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

    # 1. 初始化全局上下文
    employee_info = {
        "职级": "16",
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
    employee_info = UserProfile(**employee_info)    # 直接解包字典进行实例化
    context = GlobalContext(user_profile=employee_info)
    
    swarm_system = SwarmSystem()

    print(f"System Initialized. Current Agent: {context.next_agent}\n" + "="*50)

    # ------------------------------------------------------------------
    # 第一轮：闲聊 (由 Triage 处理)
    # ------------------------------------------------------------------
    user_input_1 = "你好，你是谁？"
    response_1 = swarm_system.run_turn(user_input_1, context)

    print(f"User: {user_input_1}")
    print(f"Bot : {response_1}")
    print(f"[State] Current Agent: {context.next_agent}") 
    # 预期: 仍在 TriageAgent
    print("-" * 30)

    # ------------------------------------------------------------------
    # 第二轮：触发转接 (重点！Triage -> LeaveAgent 自动跳转)
    # ------------------------------------------------------------------
    # 用户表达意图，Triage 识别后调用 transfer_to_LeaveAgent，
    # run_turn 中的 while 循环捕获到 next_agent 变化，立即执行 LeaveAgent
    user_input_2 = "我想请明天的年假"
    response_2 = swarm_system.run_turn(user_input_2, context)

    print(f"User: {user_input_2}")
    print(f"Bot : {response_2}") 
    # 预期输出: LeaveAgent 的回复，例如 "好的，请假日期是明天，请问需要请几天？"
    # 而不仅仅是 "正在为您转接..."

    print(f"[State] Current Agent: {context.next_agent}") 
    # 预期: LeaveAgent (控制权已移交)
    print(f"[State] Slots Captured: {context.slots}")    
    # 预期: 可能已经提取到了 {"leave_type": "年假", "start_date": "明天"}
    print("-" * 30)

    # ------------------------------------------------------------------
    # 第三轮：业务槽位填充 (由 LeaveAgent 继续处理)
    # ------------------------------------------------------------------
    user_input_3 = "请一天"
    response_3 = swarm_system.run_turn(user_input_3, context)

    print(f"User: {user_input_3}")
    print(f"Bot : {response_3}")
    print(f"[State] Slots Updated: {context.slots}")
    # 预期: {"leave_type": "年假", "start_date": "明天", "duration": "1天"}
    print("-" * 30)

    user_input_4 = "累了想休息"
    response_4 = swarm_system.run_turn(user_input_4, context)

    print(f"User: {user_input_4}")
    print(f"Bot : {response_4}")
    print(f"[State] Slots Updated: {context.slots}")
    # 预期: {'leave_type': '年假', 'start_date': '2026-01-21', 'duration': '全天', 'qjly': '累了想休息'}
    print("-" * 30)

    # ------------------------------------------------------------------

    while True:
        # 获取用户输入
        user_input = input("\nUser: ")
        response = swarm_system.run_turn(user_input, context)

        print(f"User: {user_input}")
        print(f"Bot : {response}")
        print(f"[State] Slots Updated: {context.slots}")
        print("-" * 30)
