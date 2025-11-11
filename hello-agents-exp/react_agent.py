import re 
from llm import HelloAgentsLLM
from tool import search, ToolExecutor
from prompt import REACT_PROMPT_TEMPLATE

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # ... (后续的解析、执行、整合步骤)

            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具

            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None


if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)


# 工具 'Search' 已注册。
# --- 第 1 步 ---
# 🧠 正在调用 deepseek-chat 模型...
# ✅ 大语言模型响应成功:

# 思考: 用户询问华为最新的手机型号及其主要卖点。由于手机产品更新频繁，我的知识截止到2024年7月，可能不是最新的。我应该搜索当前信息来确认华为最新发布的手机型号和其关键特性。
# 🎬 行动: Search[华为最新手机型号 主要卖点]
# 🔍 正在执行 [SerpApi] 网页搜索: 华为最新手机型号 主要卖点
# 👀 观察: ...
# --- 第 2 步 ---
# 🧠 正在调用 deepseek-chat 模型...
# ✅ 大语言模型响应成功:

# 思考: 用户询问华为最新的手机型号及其主要卖点。从搜索历史看，之前的搜索返回了华为官网和一些较旧的信息（如Mate 30 Pro），但没有明确给出最新型号。我需要搜索更具体的关键词来获取华为当前最新手机的准确信息，比如"华为最新手机2024"或"华为Pura 70"（因为我知道P系列已更名为Pura系列，且近期可能有新品）。
# 🎬 行动: Search[华为最新手机2024 Pura 70 主要卖点]
# 🔍 正在执行 [SerpApi] 网页搜索: 华为最新手机2024 Pura 70 主要卖点
# 👀 观察: ...
# --- 第 3 步 ---
# 🧠 正在调用 deepseek-chat 模型...
# ✅ 大语言模型响应成功:
# 思考: 从搜索历史来看，第一次搜索得到了关于华为手机的一般性信息，但不够具体。第二次搜索专门针对华为Pura 70系列，这是华为最新的手机系列。从搜索结果中可以看到Pura 70的主要卖点包括：风向标设计、超高速风驰闪拍、超级微距、全焦段超清影像、超聚光视频、第二代昆仑玻璃、HarmonyOS 4.2等。现在我已经收集到足够的信息来回答用户的问题。
# 行动: Finish
# 🎉 最终答案: 华为最新的手机是HUAWEI Pura 70系列。其主要卖点包括：1）全新风向标设计，引领美学新风向；2）超高速风驰闪拍技术；3）超级微距和全焦段超清影像能力；4）超聚光视频功能；5）第二代昆仑玻璃提供更好的防护；6）搭载HarmonyOS 4.2操作系统；7）支持5G网络和全新卫星通信技术。
