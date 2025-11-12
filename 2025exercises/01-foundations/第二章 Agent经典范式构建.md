> [<font style="color:rgb(66, 185, 131);">第四章 智能体经典范式构建</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=%e7%ac%ac%e5%9b%9b%e7%ab%a0-%e6%99%ba%e8%83%bd%e4%bd%93%e7%bb%8f%e5%85%b8%e8%8c%83%e5%bc%8f%e6%9e%84%e5%bb%ba)
>
> [万字长文！掌握Agent设计的九大模式](https://mp.weixin.qq.com/s/WxuhGLg7JRCa4aJYY210Ew)
>
> [Agent 的九种设计模式](https://www.53ai.com/news/qianyanjishu/913.html)
>

## [<font style="color:rgb(66, 185, 131);">智能体经典范式构建</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=%e7%ac%ac%e5%9b%9b%e7%ab%a0-%e6%99%ba%e8%83%bd%e4%bd%93%e7%bb%8f%e5%85%b8%e8%8c%83%e5%bc%8f%e6%9e%84%e5%bb%ba)
一个现代的智能体，它能够自主地理解用户意图、拆解复杂任务，并通过调用代码解释器、搜索引擎、API等一系列“工具”，来获取信息、执行操作，最终达成目标。 然而，智能体并非万能，它同样面临着来自大模型本身的“幻觉”问题、在复杂任务中可能陷入推理循环、以及对工具的错误使用等挑战，这些也构成了智能体的能力边界。

为了更好地组织智能体的“思考”与“行动”过程，业界涌现出了多种经典的架构范式。在本章中，我们将聚焦于其中最具代表性的三种，并一步步从零实现它们：

+ **ReAct (Reasoning and Acting)：** 一种将“思考”和“行动”紧密结合的范式，让智能体边想边做，动态调整。
+ **Plan-and-Solve：** 一种“三思而后行”的范式，智能体首先生成一个完整的行动计划，然后严格执行。
+ **Reflection：** 一种赋予智能体“反思”能力的范式，通过自我批判和修正来优化结果。

<font style="color:rgb(52, 73, 94);">了解了这些之后，你可能会问，市面上已有LangChain、LlamaIndex等众多优秀框架，为何还要“重复造轮子”？答案在于，尽管成熟的框架在工程效率上优势显著，但直接使用高度抽象的工具，并不利于我们了解背后的设计机制是怎么运行的，或者是有何好处。其次，这个过程会暴露出项目的工程挑战。框架为我们处理了许多问题，例如模型输出格式的解析、工具调用失败的重试、防止智能体陷入死循环等。亲手处理这些问题，是培养系统设计能力的最直接方式。最后，也是最重要的一点，掌握了设计原理，你才能真正地从一个框架的“使用者”转变为一个智能体应用的“创造者”。当标准组件无法满足你的复杂需求时，你将拥有深度定制乃至从零构建一个全新智能体的能力。</font>

### <font style="color:rgb(51, 51, 51);">ReAct</font>
ReAct 框架（[Yao et al., 2022](https://huggingface.co/papers/2210.03629)）是目前构建 agent 的主要方法。

该名称基于两个词的组合："Reason" （推理）和 "Act" （行动）。实际上，遵循此架构的 agent 将根据需要尽可能多的步骤来解决其任务，每个步骤包括一个推理步骤，然后是一个行动步骤，形成一个“思考-行动-观察”的循环。

以下是其工作原理的视频概述：[react.gif](https://cas-bridge.xethub.hf.co/xet-bridge-us/621ffdd236468d709f1835cf/00598243d02aefce27b6f2a315b745b55556b191fd8f472a95d3c0f2695ed84d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20251112%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251112T021615Z&X-Amz-Expires=3600&X-Amz-Signature=26ec7e87d2b134d3a71e63aebc70171f16a65ec1da37ccb3df0a38e7a60d5ff0&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=626905bf7e4a0f69140d398b&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27Agent_ManimCE.gif%3B+filename%3D%22Agent_ManimCE.gif%22%3B&response-content-type=image%2Fgif&x-id=GetObject&Expires=1762917375&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2MjkxNzM3NX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MjFmZmRkMjM2NDY4ZDcwOWYxODM1Y2YvMDA1OTgyNDNkMDJhZWZjZTI3YjZmMmEzMTViNzQ1YjU1NTU2YjE5MWZkOGY0NzJhOTVkM2MwZjI2OTVlZDg0ZCoifV19&Signature=l%7ESmiba0nV62dnXeTU5c3yXZxczIOCNXVMhiyydoYjKOUMKRhmi1FVnwY%7EdbeULboebMoxC64ounR93VGuIIEks0CtaxopgpigqxXEy7uCPXedC%7ErqiFNSN67BzHun-u63O4I-Pad6lDbm0RA3IZzj7EyeAHtEIfMAj-trwh2q9Te%7E0YbdqmsllLNwofuzQK-RNsGf3RmDIXbEgafJt42IKZ7v571GCXZ28oIFPI9FidfyW23rpTlWEKDaevPGU%7EPpr9jkx5%7Eld8R4AqBZq1HwsVpFh6XA1Tb32GWMN7guCo5gUI%7EIDz2pgjO69BmOqFpP%7EABHZP6Roj193sxvffBg__&Key-Pair-Id=K2L8F4GPSG1IFC)

#### [<font style="color:rgb(66, 185, 131);">ReAct 的工作流程</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_421-react-%e7%9a%84%e5%b7%a5%e4%bd%9c%e6%b5%81%e7%a8%8b)
<font style="color:rgb(52, 73, 94);">在ReAct诞生之前，主流的方法可以分为两类：一类是“纯思考”型，如</font>**<font style="color:rgb(44, 62, 80);">思维链 (Chain-of-Thought)</font>**<font style="color:rgb(52, 73, 94);">，它能引导模型进行复杂的逻辑推理，但无法与外部世界交互，容易产生事实幻觉；另一类是“纯行动”型，模型直接输出要执行的动作，但缺乏规划和纠错能力。</font>

<font style="color:rgb(52, 73, 94);">ReAct的巧妙之处在于，它认识到</font>**<font style="color:rgb(44, 62, 80);">思考与行动是相辅相成的</font>**<font style="color:rgb(52, 73, 94);">。思考指导行动，而行动的结果又反过来修正思考。为此，ReAct范式通过一种特殊的提示工程来引导模型，使其每一步的输出都遵循一个固定的轨迹：</font>

+ **<font style="color:rgb(44, 62, 80);">Thought (思考)：</font>**<font style="color:rgb(52, 73, 94);"> </font><font style="color:rgb(52, 73, 94);">这是智能体的“内心独白”。它会分析当前情况、分解任务、制定下一步计划，或者反思上一步的结果。</font>
+ **<font style="color:rgb(44, 62, 80);">Action (行动)：</font>**<font style="color:rgb(52, 73, 94);"> </font><font style="color:rgb(52, 73, 94);">这是智能体决定采取的具体动作，通常是调用一个外部工具，例如</font><font style="color:rgb(52, 73, 94);"> </font>`<font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">Search['华为最新款手机']</font>`<font style="color:rgb(52, 73, 94);">。</font>
+ **<font style="color:rgb(44, 62, 80);">Observation (观察)：</font>**<font style="color:rgb(52, 73, 94);"> </font><font style="color:rgb(52, 73, 94);">这是执行</font>`<font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">Action</font>`<font style="color:rgb(52, 73, 94);">后从外部工具返回的结果，例如搜索结果的摘要或API的返回值。</font>

<font style="color:rgb(52, 73, 94);">智能体将不断重复这个 </font>**<font style="color:rgb(44, 62, 80);">Thought -> Action -> Observation</font>**<font style="color:rgb(52, 73, 94);"> 的循环，将新的观察结果追加到历史记录中，形成一个不断增长的上下文，直到它在</font>`<font style="color:rgb(233, 105, 0);background-color:rgb(248, 248, 248);">Thought</font>`<font style="color:rgb(52, 73, 94);">中认为已经找到了最终答案，然后输出结果。这个过程形成了一个强大的协同效应：</font>**<font style="color:rgb(44, 62, 80);">推理使得行动更具目的性，而行动则为推理提供了事实依据。</font>**

![](https://cdn.nlark.com/yuque/0/2025/png/8420697/1762509159404-67fe03e1-c90a-4d90-a94b-80b62ecfcc06.png)

![ReAct agent 的框架](https://cdn.nlark.com/yuque/0/2025/png/8420697/1762913634656-bef5d858-b69c-47df-a00a-99bc0d13f757.png)

> <font style="color:rgb(52, 73, 94);">[!TIP]</font>
>

<font style="color:rgb(52, 73, 94);">这种机制特别适用于以下场景：</font>

+ **<font style="color:rgb(44, 62, 80);">需要外部知识的任务</font>**<font style="color:rgb(52, 73, 94);">：如查询实时信息（天气、新闻、股价）、搜索专业领域的知识等。</font>
+ **<font style="color:rgb(44, 62, 80);">需要精确计算的任务</font>**<font style="color:rgb(52, 73, 94);">：将数学问题交给计算器工具，避免LLM的计算错误。</font>
+ **<font style="color:rgb(44, 62, 80);">需要与API交互的任务</font>**<font style="color:rgb(52, 73, 94);">：如操作数据库、调用某个服务的API来完成特定功能。</font>



### <font style="color:rgb(51, 51, 51);">Plan and Solve</font>
<font style="color:rgb(52, 73, 94);">在我们掌握了 ReAct 这种反应式的、步进决策的智能体范式后，接下来将探讨一种风格迥异但同样强大的方法，</font>**<font style="color:rgb(44, 62, 80);">Plan-and-Solve</font>**<font style="color:rgb(25, 27, 31);">，旨在处理更复杂、多步骤的任务</font><font style="color:rgb(52, 73, 94);">。顾名思义，这种范式将任务处理明确地分为两个阶段：</font>**<font style="color:rgb(44, 62, 80);">先规划 (Plan)，后执行 (Solve)</font>**<font style="color:rgb(52, 73, 94);">。</font>

<font style="color:rgb(52, 73, 94);">如果说 ReAct 像一个经验丰富的侦探，根据现场的蛛丝马迹（Observation）一步步推理，随时调整自己的调查方向；那么 Plan-and-Solve 则更像一位建筑师，在动工之前必须先绘制出完整的蓝图（Plan），然后严格按照蓝图来施工（Solve）。</font>

#### [<font style="color:rgb(66, 185, 131);">Plan-and-Solve 的工作原理</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_431-plan-and-solve-%e7%9a%84%e5%b7%a5%e4%bd%9c%e5%8e%9f%e7%90%86)
<font style="color:rgb(52, 73, 94);">Plan-and-Solve Prompting 由 Lei Wang 在2023年提出</font><sup><font style="color:rgb(52, 73, 94);">[2]</font></sup><font style="color:rgb(52, 73, 94);">。其核心动机是为了解决思维链在处理多步骤、复杂问题时容易“偏离轨道”的问题。</font>

<font style="color:rgb(52, 73, 94);">与 ReAct 将思考和行动融合在每一步不同，Plan-and-Solve 将整个流程解耦为两个核心阶段，如图所示：</font>

+ **<font style="color:rgb(25, 27, 31);">规划阶段</font>**<font style="color:rgb(25, 27, 31);">：Agent 首先对接收到的复杂任务或目标进行整体分析和理解。然后，它会生成一个高层次的计划，将原始任务分解为一系列更小、更易于管理的子任务或步骤。这种分解有助于在执行阶段减少处理每个子任务所需的上下文长度，这个计划通常是一个有序的行动序列，指明了要达成最终目标需要完成哪些关键环节。这个蓝图可以先呈现给用户，允许用户在执行开始前对计划步骤给出修改意见。</font>
+ **<font style="color:rgb(25, 27, 31);">执行阶段</font>**<font style="color:rgb(25, 27, 31);">：计划制定完成后（可能已采纳用户意见），Agent 进入执行阶段。它会按照规划好的步骤逐一执行每个子任务。在执行每个子任务时，Agent 可以采用标准的 ReAct 循环来处理该子任务的具体细节，例如调用特定工具、与外部环境交互、或进行更细致的推理。执行过程中，Agent 会监控每个子任务的完成情况。如果某个子任务成功，则继续下一个；如果遇到失败或预期之外的情况，Agent 可能需要重新评估当前计划，可以动态调整计划或返回到规划阶段进行修正。此阶段同样可以引入用户参与，允许用户对子任务的执行过程或结果进行反馈，甚至提出调整建议。</font>

![Plan-and-Solve 范式的两阶段工作流](https://cdn.nlark.com/yuque/0/2025/png/8420697/1756200602117-971d2288-930f-436d-b5af-5e0b7609a532.png)

<font style="color:rgb(51, 51, 51);">Plan and Solve 模式的交互流程如下：</font>

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令。</font>
+ <font style="color:rgb(51, 51, 51);">规划（Plan）：Agent 根据任务目标生成一个多步计划，明确每个步骤的具体内容和顺序。</font>
+ <font style="color:rgb(51, 51, 51);">执行（Solve）：Agent 按照计划逐步执行每个步骤。</font>
+ <font style="color:rgb(51, 51, 51);">观察（Observation）：Agent 对执行结果进行观察，判断是否需要重新规划。</font>
+ <font style="color:rgb(51, 51, 51);">重新规划（Replan）：如果发现计划不可行或需要调整，Agent 会根据当前状态重新生成计划，并继续执行。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代：重复执行、观察和重新规划的过程，直到任务完成。</font>

<font style="color:rgb(25, 27, 31);">与标准的 ReAct 相比，Plan-and-Execute 模式的主要优势在于：</font>

+ **<font style="color:rgb(25, 27, 31);">结构化与上下文优化</font>**<font style="color:rgb(25, 27, 31);">：通过预先规划将复杂任务分解为小步骤，不仅使 Agent 行为更有条理，还有效减少了执行各子任务时的上下文长度，提升了处理长链条任务的效率和稳定性。</font>
+ **<font style="color:rgb(25, 27, 31);">提升鲁棒性</font>**<font style="color:rgb(25, 27, 31);">：将大问题分解为小问题，降低了单步决策的复杂性。如果某个子任务失败，影响范围相对可控，也更容易进行针对性的调整。</font>
+ **<font style="color:rgb(25, 27, 31);">增强可解释性与人机协同</font>**<font style="color:rgb(25, 27, 31);">：清晰的计划和分步执行过程使得 Agent 的行为更容易被理解和调试。更重要的是，任务的分解为用户在规划审批和执行监控等环节的参与提供了便利，用户可以对任务的执行步骤给出修改意见，从而实现更高效的人机协作，确保任务结果更符合预期。</font>

<font style="color:rgb(51, 51, 51);">示例：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/8420697/1756200691533-e92e2690-b5e5-41a7-b06e-f5ad4d7af5f3.png)

<font style="color:rgb(52, 73, 94);">Plan-and-Solve 尤其适用于那些结构性强、可以被清晰分解的复杂任务，例如：</font>

+ **<font style="color:rgb(44, 62, 80);">多步数学应用题</font>**<font style="color:rgb(52, 73, 94);">：需要先列出计算步骤，再逐一求解。</font>
+ **<font style="color:rgb(44, 62, 80);">需要整合多个信息源的报告撰写</font>**<font style="color:rgb(52, 73, 94);">：需要先规划好报告结构（引言、数据来源A、数据来源B、总结），再逐一填充内容。</font>
+ **<font style="color:rgb(44, 62, 80);">代码生成任务</font>**<font style="color:rgb(52, 73, 94);">：需要先构思好函数、类和模块的结构，再逐一实现。</font>

<font style="color:rgb(52, 73, 94);"></font>

### <font style="color:rgb(51, 51, 51);">Reason without Observation</font>
<font style="color:rgb(51, 51, 51);">Reason without Observation 模式原理</font>

<font style="color:rgb(51, 51, 51);">Reason without Observation（REWOO）模式是一种创新的 Agent 设计模式，在传统 ReAct 模式的基础上进行了优化，去掉了显式观察（Observation）步骤，而是将观察结果隐式地嵌入到下一步的执行中。这种模式的核心在于通过推理（Reasoning）和行动（Action）的紧密协作，实现更加高效和连贯的任务执行。</font>



**<font style="color:rgb(51, 51, 51);">REWOO 模式中，Agent 交互流程：</font>**

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令。</font>
+ <font style="color:rgb(51, 51, 51);">推理（Reasoning）：Agent 根据当前的任务和已有的知识进行推理，生成初步的行动计划。</font>
+ <font style="color:rgb(51, 51, 51);">行动（Action）：Agent 执行推理得出的行动。</font>
+ <font style="color:rgb(51, 51, 51);">隐式观察（Implicit Observation）：Agent 在执行行动的过程中，自动将结果反馈到下一步的推理中，而不是显式地进行观察。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代：Agent 根据新的信息重新进行推理，生成新的行动计划，并继续执行行动，直到任务完成。</font>

**<font style="color:rgb(51, 51, 51);">示例：</font>**<font style="color:rgb(51, 51, 51);">Agent 完成审批流程的任务：</font>

```plain
class REWOOAgent:
    def __init__(self):
        self.knowledge_base = {
            "steps": [
                "从部门 A 获取文件 a",
                "拿着文件 a 去部门 B 办理文件 b",
                "拿着文件 b 去部门 C 办理文件 c"
            ]
        }

    def reason(self, task, current_step):
        # 推理：根据任务和当前步骤生成行动计划
        if current_step == 0:
            return"从部门 A 获取文件 a"
        elif current_step == 1:
            return"拿着文件 a 去部门 B 办理文件 b"
        elif current_step == 2:
            return"拿着文件 b 去部门 C 办理文件 c"

    def act(self, action):
        # 行动：执行具体的行动
        if action == "从部门 A 获取文件 a":
            return"文件 a 已获取"
        elif action == "拿着文件 a 去部门 B 办理文件 b":
            return"文件 b 已办理"
        elif action == "拿着文件 b 去部门 C 办理文件 c":
            return"文件 c 已办理"

    def run(self, task):
        # 主循环：推理 -> 行动 -> 隐式观察 -> 循环迭代
        steps = self.knowledge_base["steps"]
        for current_step in range(len(steps)):
            action = self.reason(task, current_step)
            action_result = self.act(action)
            print(f"Action: {action}")
            print(f"Result: {action_result}")
            if"文件 c 已办理"in action_result:
                print("任务完成：审批流程完成")
                break
# 示例运行
agent = REWOOAgent()
agent.run("完成审批流程")
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
Action: 从部门 A 获取文件 a
Result: 文件 a 已获取
Action: 拿着文件 a 去部门 B 办理文件 b
Result: 文件 b 已办理
Action: 拿着文件 b 去部门 C 办理文件 c
Result: 文件 c 已办理
任务完成：审批流程完成
```

<font style="color:rgb(51, 51, 51);">优势：REWOO 模式如何通过推理和行动的紧密协作，实现高效的任务执行。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如工作流程自动化、多步骤任务处理等，通过不断优化 Agent 的推理和行动策略，实现更加智能和高效的任务执行。</font>

+ <font style="color:rgb(51, 51, 51);">高效性：去掉了显式的观察步骤，减少了交互的复杂性，提高了任务执行的效率。</font>
+ <font style="color:rgb(51, 51, 51);">连贯性：通过隐式观察，Agent 的行动更加连贯，避免了不必要的重复操作。</font>
+ <font style="color:rgb(51, 51, 51);">适应性：Agent 能够根据任务的复杂性和环境的变化灵活调整行动策略。</font>

<font style="color:rgb(51, 51, 51);"></font>

### <font style="color:rgb(51, 51, 51);">LLMCompiler 模式</font>
**<font style="color:rgb(51, 51, 51);">LLMCompiler 模式原理</font>**

<font style="color:rgb(51, 51, 51);">LLMCompiler 模式是一种通过并行函数调用提高效率的 Agent 设计模式。该模式的核心在于优化任务的编排，使得 Agent 能够同时处理多个任务，从而显著提升任务处理的速度和效率。这种模式特别适用于需要同时处理多个子任务的复杂任务场景，例如多任务查询、数据并行处理等。</font>

<font style="color:rgb(51, 51, 51);">流程：</font>

**<font style="color:rgb(51, 51, 51);">LLMCompiler 模式的交互流程：</font>**

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令，任务可能包含多个子任务。</font>
+ <font style="color:rgb(51, 51, 51);">任务分解（Task Decomposition）：Agent 将复杂任务分解为多个子任务，并确定这些子任务之间的依赖关系。</font>
+ <font style="color:rgb(51, 51, 51);">并行执行（Parallel Execution）：Agent 根据子任务之间的依赖关系，将可以并行处理的子任务同时发送给多个执行器进行处理。</font>
+ <font style="color:rgb(51, 51, 51);">结果合并（Result Merging）：各个执行器完成子任务后，Agent 将结果合并，形成最终的输出。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代（Iteration）：如果任务需要进一步处理或调整，Agent 会根据当前结果重新分解任务，并继续并行执行和结果合并，直到任务完成。</font>

**<font style="color:rgb(51, 51, 51);">优势在于：</font>**

+ <font style="color:rgb(51, 51, 51);">高效率：通过并行处理多个子任务，显著减少了任务完成的总时间。</font>
+ <font style="color:rgb(51, 51, 51);">灵活性：能够根据任务的复杂性和子任务之间的依赖关系动态调整任务分解和执行策略。</font>
+ <font style="color:rgb(51, 51, 51);">可扩展性：适用于大规模任务和复杂任务场景，能够有效利用多核处理器和分布式计算资源。</font>

<font style="color:rgb(51, 51, 51);">LLMCompiler 模式的交互流程可以用以下图示来表示：</font>

```plain
+-------------------+
|     接收任务      |
+-------------------+
           |
           v
+-------------------+
| 任务分解（Task Decomposition）|
+-------------------+
           |
           v
+-------------------+
| 并行执行（Parallel Execution）|
+-------------------+
           |
           v
+-------------------+
| 结果合并（Result Merging）|
+-------------------+
           |
           v
+-------------------+
|     循环迭代      |
+-------------------+
```

<font style="color:rgb(51, 51, 51);">示例：</font>

<font style="color:rgb(51, 51, 51);">Agent 同时查询两个人的年龄并计算年龄差的任务：</font>

```plain
import concurrent.futures

class LLMCompilerAgent:
    def __init__(self):
        self.knowledge_base = {
            "person_age": {
                "张译": 40,
                "吴京": 48
            }
        }

    def query_age(self, name):
        # 查询年龄
        return self.knowledge_base["person_age"].get(name, "未知")

    def calculate_age_difference(self, age1, age2):
        # 计算年龄差
        try:
            return abs(int(age1) - int(age2))
        except ValueError:
            return"无法计算年龄差"

    def run(self, task):
        # 主流程：任务分解 -> 并行执行 -> 结果合并 -> 循环迭代
        if task == "查询张译和吴京的年龄差":
            # 任务分解
            tasks = ["查询张译的年龄", "查询吴京的年龄"]
            results = {}

            # 并行执行
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.query_age, name): name for name in ["张译", "吴京"]}
                for future in concurrent.futures.as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        results[name] = f"查询失败: {e}"

            # 结果合并
            age_difference = self.calculate_age_difference(results["张译"], results["吴京"])
            print(f"张译的年龄: {results['张译']}")
            print(f"吴京的年龄: {results['吴京']}")
            print(f"年龄差: {age_difference}")

# 示例运行
agent = LLMCompilerAgent()
agent.run("查询张译和吴京的年龄差")
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
张译的年龄: 40
吴京的年龄: 48
年龄差: 8
```

<font style="color:rgb(51, 51, 51);">LLMCompiler 模式如何通过任务分解、并行执行和结果合并来高效完成任务。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如多任务查询、数据并行处理等，通过不断优化 Agent 的任务分解和并行执行策略，实现更加智能和高效的任务执行。</font>

### [<font style="color:rgb(66, 185, 131);">Reflection</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_44-reflection)
<font style="color:rgb(52, 73, 94);">在我们已经实现的 ReAct 和 Plan-and-Solve 范式中，智能体一旦完成了任务，其工作流程便告结束。然而，它们生成的初始答案，无论是行动轨迹还是最终结果，都可能存在谬误或有待改进之处。Reflection 机制的核心思想，正是为智能体引入一种</font>**<font style="color:rgb(44, 62, 80);">事后（post-hoc）的自我校正循环</font>**<font style="color:rgb(52, 73, 94);">，使其能够像人类一样，审视自己的工作，发现不足，并进行迭代优化。</font>

#### [<font style="color:rgb(66, 185, 131);">Reflection 机制的核心思想</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_441-reflection-%e6%9c%ba%e5%88%b6%e7%9a%84%e6%a0%b8%e5%bf%83%e6%80%9d%e6%83%b3)
<font style="color:rgb(52, 73, 94);">Reflection 机制的灵感来源于人类的学习过程：我们完成初稿后会进行校对，解出数学题后会进行验算。这一思想在多个研究中得到了体现，例如 Shinn, Noah 在2023年提出的 Reflexion 框架</font><sup><font style="color:rgb(52, 73, 94);">[3]</font></sup><font style="color:rgb(52, 73, 94);">。其核心工作流程可以概括为一个简洁的三步循环：</font>**<font style="color:rgb(44, 62, 80);">执行 -> 反思 -> 优化</font>**<font style="color:rgb(52, 73, 94);">。</font>

1. **<font style="color:rgb(44, 62, 80);">执行 (Execution)</font>**<font style="color:rgb(52, 73, 94);">：首先，智能体使用我们熟悉的方法（如 ReAct 或 Plan-and-Solve）尝试完成任务，生成一个初步的解决方案或行动轨迹。这可以看作是“初稿”。</font>
2. **<font style="color:rgb(44, 62, 80);">反思 (Reflection)</font>**<font style="color:rgb(52, 73, 94);">：接着，智能体进入反思阶段。它会调用一个独立的、或者带有特殊提示词的大语言模型实例，来扮演一个“评审员”的角色。这个“评审员”会审视第一步生成的“初稿”，并从多个维度进行评估，例如：</font>
    - **<font style="color:rgb(44, 62, 80);">事实性错误</font>**<font style="color:rgb(52, 73, 94);">：是否存在与常识或已知事实相悖的内容？</font>
    - **<font style="color:rgb(44, 62, 80);">逻辑漏洞</font>**<font style="color:rgb(52, 73, 94);">：推理过程是否存在不连贯或矛盾之处？</font>
    - **<font style="color:rgb(44, 62, 80);">效率问题</font>**<font style="color:rgb(52, 73, 94);">：是否有更直接、更简洁的路径来完成任务？</font>
    - **<font style="color:rgb(44, 62, 80);">遗漏信息</font>**<font style="color:rgb(52, 73, 94);">：是否忽略了问题的某些关键约束或方面？ 根据评估，它会生成一段结构化的</font>**<font style="color:rgb(44, 62, 80);">反馈 (Feedback)</font>**<font style="color:rgb(52, 73, 94);">，指出具体的问题所在和改进建议。</font>
3. **<font style="color:rgb(44, 62, 80);">优化 (Refinement)</font>**<font style="color:rgb(52, 73, 94);">：最后，智能体将“初稿”和“反馈”作为新的上下文，再次调用大语言模型，要求它根据反馈内容对初稿进行修正，生成一个更完善的“修订稿”。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/8420697/1762877412903-8f437393-8324-4905-8486-b2f25409f263.png)<font style="color:rgb(51, 51, 51);">  
</font>![](https://cdn.nlark.com/yuque/0/2025/png/8420697/1762877422246-0894e892-d8fc-40a7-9382-c80f08dda0de.png)

**<font style="color:rgb(51, 51, 51);">Basic Reflection 模式的交互流程如下：</font>**

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令。</font>
+ <font style="color:rgb(51, 51, 51);">生成初始响应（Initial Response）：Agent 根据任务生成一个初步的回答或解决方案。</font>
+ <font style="color:rgb(51, 51, 51);">反思（Reflection）：Agent 对初始响应进行评估，检查是否存在错误、遗漏或可以改进的地方。</font>
+ <font style="color:rgb(51, 51, 51);">修正（Revision）：根据反思的结果，Agent 对初始响应进行修正，生成最终的输出。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代（Iteration）：如果任务需要进一步优化，Agent 会重复反思和修正的过程，直到输出满足要求。</font>

**<font style="color:rgb(52, 73, 94);">与 ReAct 和 Plan-and-Solve 范式相比，Reflection 的价值在于：</font>**

+ <font style="color:rgb(52, 73, 94);">它为智能体提供了一个内部纠错回路，使其不再完全依赖于外部工具的反馈（ReAct 的 Observation），从而能够修正更高层次的逻辑和策略错误。</font>
+ <font style="color:rgb(52, 73, 94);">它将一次性的任务执行，转变为一个持续优化的过程，显著提升了复杂任务的最终成功率和答案质量。</font>
+ <font style="color:rgb(52, 73, 94);">它为智能体构建了一个临时的</font>**<font style="color:rgb(44, 62, 80);">“短期记忆”</font>**<font style="color:rgb(52, 73, 94);">。整个“执行-反思-优化”的轨迹形成了一个宝贵的经验记录，智能体不仅知道最终答案，还记得自己是如何从有缺陷的初稿迭代到最终版本的。更进一步，这个记忆系统还可以是</font>**<font style="color:rgb(44, 62, 80);">多模态的</font>**<font style="color:rgb(52, 73, 94);">，允许智能体反思和修正文本以外的输出（如代码、图像等），为构建更强大的多模态智能体奠定了基础。</font>

<font style="color:rgb(52, 73, 94);"></font>

**<font style="color:rgb(51, 51, 51);">Reflection 模式的交互流程示例</font>**<font style="color:rgb(51, 51, 51);">：（回答数学问题）</font>

```python
class BasicReflectionAgent:
    def __init__(self):
        self.knowledge_base = {
            "math_problems": {
                "1+1": 2,
                "2*2": 4,
                "3*3": 9
            }
        }

    def initial_response(self, task):
        # 生成初始响应：根据任务生成初步回答
        return self.knowledge_base["math_problems"].get(task, "未知")

    def reflect(self, response):
        # 反思：检查初始响应是否准确
        if response == "未知":
            return"需要进一步查找答案"
        else:
            return"答案正确"

    def revise(self, response, reflection):
        # 修正：根据反思结果调整响应
        if reflection == "需要进一步查找答案":
            return"抱歉，我没有找到答案"
        else:
            return response

    def run(self, task):
        # 主流程：生成初始响应 -> 反思 -> 修正 -> 循环迭代
        initial_response = self.initial_response(task)
        reflection = self.reflect(initial_response)
        final_response = self.revise(initial_response, reflection)
        print(f"Initial Response: {initial_response}")
        print(f"Reflection: {reflection}")
        print(f"Final Response: {final_response}")

# 示例运行
agent = BasicReflectionAgent()
agent.run("1+1")
agent.run("5*5")
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
Initial Response: 2
Reflection: 答案正确
Final Response: 2
Initial Response: 未知
Reflection: 需要进一步查找答案
Final Response: 抱歉，我没有找到答案
```

<font style="color:rgb(51, 51, 51);">Basic Reflection 模式如何通过生成初始响应、反思和修正的过程来优化 Agent 的行为。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如智能客服、自动问答系统等，通过不断优化 Agent 的反思和修正策略，实现更加智能和高效的任务执行。</font>

### <font style="color:rgb(51, 51, 51);">Reflexion 模式</font>
<font style="color:rgb(51, 51, 51);">Reflexion 模式原理</font>

<font style="color:rgb(51, 51, 51);">Reflexion 模式是一种基于强化学习的 Agent 设计模式，旨在通过引入外部数据评估和自我反思机制，进一步优化 Agent 的行为和输出。与 Basic Reflection 模式相比，Reflexion 模式不仅对初始响应进行反思和修正，还通过外部数据来评估回答的准确性和完整性，从而生成更具建设性的修正建议。</font>

<font style="color:rgb(51, 51, 51);">Reflexion 模式的交互流程如下：</font>

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令。</font>
+ <font style="color:rgb(51, 51, 51);">生成初始响应（Initial Response）：Agent 根据任务生成一个初步的回答或解决方案。</font>
+ <font style="color:rgb(51, 51, 51);">外部评估（External Evaluation）：引入外部数据或标准，对初始响应进行评估，检查是否存在错误、遗漏或可以改进的地方。</font>
+ <font style="color:rgb(51, 51, 51);">反思（Reflection）：Agent 根据外部评估的结果，对初始响应进行自我反思，识别问题所在。</font>
+ <font style="color:rgb(51, 51, 51);">修正（Revision）：根据反思的结果，Agent 对初始响应进行修正，生成最终的输出。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代（Iteration）：如果任务需要进一步优化，Agent 会重复外部评估、反思和修正的过程，直到输出满足要求。</font>

<font style="color:rgb(51, 51, 51);">优势在于：</font>

+ <font style="color:rgb(51, 51, 51);">提高准确性：通过外部数据评估和自我反思，Agent 能够更准确地识别错误和遗漏，从而提高输出的准确性。</font>
+ <font style="color:rgb(51, 51, 51);">增强适应性：Agent 能够根据不同的任务和环境调整自己的行为策略，增强适应性。</font>
+ <font style="color:rgb(51, 51, 51);">提升用户体验：通过不断优化输出，Agent 能够提供更高质量的服务，提升用户体验。</font>
+ <font style="color:rgb(51, 51, 51);">强化学习：引入外部数据评估机制，使 Agent 的学习过程更加科学和有效，能够更好地适应复杂任务和动态环境。</font>

<font style="color:rgb(51, 51, 51);">Reflexion 模式的交互流程可以用以下图示来表示：</font>

```plain
+-------------------+
|     接收任务      |
+-------------------+
           |
           v
+-------------------+
| 生成初始响应（Initial Response）|
+-------------------+
           |
           v
+-------------------+
| 外部评估（External Evaluation）|
+-------------------+
           |
           v
+-------------------+
|     反思（Reflection）|
+-------------------+
           |
           v
+-------------------+
|     修正（Revision）|
+-------------------+
           |
           v
+-------------------+
|     循环迭代      |
+-------------------+
```

<font style="color:rgb(51, 51, 51);">示例：</font>

<font style="color:rgb(51, 51, 51);">Agent 回答数学问题的任务：</font>

```plain
class ReflexionAgent:
    def __init__(self):
        self.knowledge_base = {
            "math_problems": {
                "1+1": 2,
                "2*2": 4,
                "3*3": 9
            }
        }
        self.external_data = {
            "1+1": 2,
            "2*2": 4,
            "3*3": 9,
            "5*5": 25
        }

    def initial_response(self, task):
        # 生成初始响应：根据任务生成初步回答
        return self.knowledge_base["math_problems"].get(task, "未知")

    def external_evaluation(self, response, task):
        # 外部评估：检查初始响应是否准确
        correct_answer = self.external_data.get(task, "未知")
        if response == correct_answer:
            return"答案正确"
        else:
            returnf"答案错误，正确答案是 {correct_answer}"

    def reflect(self, evaluation):
        # 反思：根据外部评估的结果进行自我反思
        if"答案错误"in evaluation:
            return"需要修正答案"
        else:
            return"无需修正"

    def revise(self, response, reflection, evaluation):
        # 修正：根据反思结果调整响应
        if reflection == "需要修正答案":
            correct_answer = evaluation.split("正确答案是 ")[1]
            return correct_answer
        else:
            return response

    def run(self, task):
        # 主流程：生成初始响应 -> 外部评估 -> 反思 -> 修正 -> 循环迭代
        initial_response = self.initial_response(task)
        evaluation = self.external_evaluation(initial_response, task)
        reflection = self.reflect(evaluation)
        final_response = self.revise(initial_response, reflection, evaluation)
        print(f"Initial Response: {initial_response}")
        print(f"External Evaluation: {evaluation}")
        print(f"Reflection: {reflection}")
        print(f"Final Response: {final_response}")

# 示例运行
agent = ReflexionAgent()
agent.run("1+1")
agent.run("5*5")
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
Initial Response: 2
External Evaluation: 答案正确
Reflection: 无需修正
Final Response: 2
Initial Response: 未知
External Evaluation: 答案错误，正确答案是 25
Reflection: 需要修正答案
Final Response: 25
```

<font style="color:rgb(51, 51, 51);">Reflexion 模式如何通过生成初始响应、外部评估、反思和修正的过程来优化 Agent 的行为。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如智能客服、自动问答系统等，通过不断优化 Agent 的反思和修正策略，实现更加智能和高效的任务执行。</font>

### <font style="color:rgb(51, 51, 51);">Language Agent Tree Search 模式</font>
<font style="color:rgb(51, 51, 51);">Language Agent Tree Search 模式原理</font>

<font style="color:rgb(51, 51, 51);">Language Agent Tree Search（LATS）模式是一种融合了树搜索、ReAct、Plan & Solve 以及反思机制的 Agent 设计模式。它通过多轮迭代和树搜索的方式，对可能的解决方案进行探索和评估，从而找到最优解。这种模式特别适用于复杂任务的解决，尤其是在需要对多种可能性进行评估和选择的场景中。</font>

<font style="color:rgb(51, 51, 51);">流程：</font>

<font style="color:rgb(51, 51, 51);">LATS 模式的交互流程如下：</font>

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令。</font>
+ <font style="color:rgb(51, 51, 51);">树搜索（Tree Search）：Agent 构建一个搜索树，将任务分解为多个子任务，并探索所有可能的解决方案路径。</font>
+ <font style="color:rgb(51, 51, 51);">ReAct 交互：在树搜索的过程中，Agent 使用 ReAct 模式对每个子任务进行推理和行动，获取反馈信息。</font>
+ <font style="color:rgb(51, 51, 51);">Plan & Solve 执行：Agent 根据树搜索的结果，生成一个多步计划，并逐步执行计划中的每个步骤。</font>
+ <font style="color:rgb(51, 51, 51);">反思与修正（Reflection & Revision）：Agent 对执行结果进行反思，评估每个步骤的正确性和效率，根据反思结果对计划进行修正。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代（Iteration）：Agent 重复树搜索、ReAct 交互、Plan & Solve 执行和反思修正的过程，直到找到最优解或任务完成。</font>

<font style="color:rgb(51, 51, 51);">优势在于：</font>

+ <font style="color:rgb(51, 51, 51);">全局优化：通过树搜索，Agent 能够全面探索所有可能的解决方案，找到最优路径。</font>
+ <font style="color:rgb(51, 51, 51);">灵活性：结合 ReAct 和 Plan & Solve 模式，Agent 能够灵活应对任务中的动态变化。</font>
+ <font style="color:rgb(51, 51, 51);">准确性：通过反思机制，Agent 能够不断优化自己的行为，提高任务完成的准确性。</font>
+ <font style="color:rgb(51, 51, 51);">适应性：适用于复杂任务和多步骤任务，能够有效管理任务的各个阶段。</font>

<font style="color:rgb(51, 51, 51);">LATS 模式的交互流程可以用以下图示来表示：</font>

```plain
+-------------------+
|     接收任务      |
+-------------------+
           |
           v
+-------------------+
|     树搜索（Tree Search）|
+-------------------+
           |
           v
+-------------------+
| ReAct 交互（ReAct Interaction）|
+-------------------+
           |
           v
+-------------------+
| Plan & Solve 执行（Plan & Solve Execution）|
+-------------------+
           |
           v
+-------------------+
| 反思与修正（Reflection & Revision）|
+-------------------+
           |
           v
+-------------------+
|     循环迭代      |
+-------------------+
```

<font style="color:rgb(51, 51, 51);">示例：</font>

<font style="color:rgb(51, 51, 51);">Agent 解决一个复杂的任务，例如规划一条旅行路线并优化行程：</font>

```plain
class LATSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

class LATSAgent:
    def __init__(self):
        self.knowledge_base = {
            "cities": ["北京", "上海", "广州", "深圳"],
            "distances": {
                ("北京", "上海"): 1300,
                ("北京", "广州"): 2000,
                ("北京", "深圳"): 2200,
                ("上海", "广州"): 1200,
                ("上海", "深圳"): 1500,
                ("广州", "深圳"): 100
            }
        }

    def tree_search(self, start, goal):
        # 树搜索：构建搜索树并找到最优路径
        open_list = [LATSNode(start)]
        while open_list:
            current_node = open_list.pop(0)
            if current_node.state == goal:
                return self.get_path(current_node)
            for city in self.knowledge_base["cities"]:
                if city != current_node.state:
                    new_node = LATSNode(city, current_node, f"前往 {city}")
                    open_list.append(new_node)
        returnNone

    def get_path(self, node):
        # 获取路径
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1]

    def react_interaction(self, path):
        # ReAct 交互：对每个步骤进行推理和行动
        observations = []
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            distance = self.knowledge_base["distances"].get((start, end), 0)
            observations.append(f"从 {start} 到 {end} 的距离是 {distance} 公里")
        return observations

    def plan_and_solve(self, observations):
        # Plan & Solve 执行：根据观察结果生成计划并执行
        plan = []
        for observation in observations:
            plan.append(f"根据 {observation}，调整行程")
        return plan

    def reflect_and_revise(self, plan):
        # 反思与修正：评估计划并进行修正
        revised_plan = []
        for step in plan:
            if"调整行程"in step:
                revised_plan.append("优化行程")
        return revised_plan

    def run(self, start, goal):
        # 主流程：树搜索 -> ReAct 交互 -> Plan & Solve 执行 -> 反思与修正 -> 循环迭代
        path = self.tree_search(start, goal)
        if path:
            observations = self.react_interaction(path)
            plan = self.plan_and_solve(observations)
            revised_plan = self.reflect_and_revise(plan)
            print(f"路径: {path}")
            print(f"观察结果: {observations}")
            print(f"初始计划: {plan}")
            print(f"修正后的计划: {revised_plan}")
        else:
            print("未找到路径")

# 示例运行
agent = LATSAgent()
agent.run("北京", "深圳")
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
路径: ['北京', '上海', '深圳']
观察结果: ['从 北京 到 上海 的距离是 1300 公里', '从 上海 到 深圳 的距离是 1500 公里']
初始计划: ['根据 从 北京 到 上海 的距离是 1300 公里，调整行程', '根据 从 上海 到 深圳 的距离是 1500 公里，调整行程']
修正后的计划: ['优化行程', '优化行程']
```

<font style="color:rgb(51, 51, 51);">LATS 模式如何通过树搜索、ReAct 交互、Plan & Solve 执行和反思修正的多轮迭代来优化 Agent 的行为。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如路径规划、资源优化等，通过不断优化 Agent 的行为策略，实现更加智能和高效的任务执行。</font>

### <font style="color:rgb(51, 51, 51);">Self-Discover 模式</font>
<font style="color:rgb(51, 51, 51);">Self-Discover 模式原理</font>

<font style="color:rgb(51, 51, 51);">Self-Discover 模式是一种让 Agent 在更小粒度上对任务本身进行反思的设计模式。这种模式的核心在于通过自我发现和自我调整，使 Agent 能够更深入地理解任务的本质和需求，从而优化行为和输出。与 Reflexion 模式相比，Self-Discover 模式不仅关注任务的执行结果，还注重任务本身的逻辑和结构，通过自我发现潜在问题和改进点，实现更深层次的优化。</font>

<font style="color:rgb(51, 51, 51);">流程：</font>

<font style="color:rgb(51, 51, 51);">Self-Discover 模式的交互流程如下：</font>

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令。</font>
+ <font style="color:rgb(51, 51, 51);">任务分析（Task Analysis）：Agent 对任务进行初步分析，识别任务的关键要素和目标。</font>
+ <font style="color:rgb(51, 51, 51);">自我发现（Self-Discovery）：Agent 对任务本身进行反思，发现潜在的问题、遗漏或可以改进的地方。这一步骤包括对任务逻辑、数据需求和目标的深入分析。</font>
+ <font style="color:rgb(51, 51, 51);">调整策略（Strategy Adjustment）：根据自我发现的结果，Agent 调整任务执行策略，优化行为路径。</font>
+ <font style="color:rgb(51, 51, 51);">执行与反馈（Execution & Feedback）：Agent 按照调整后的策略执行任务，并收集反馈信息，进一步优化行为。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代（Iteration）：Agent 重复自我发现、调整策略和执行任务的过程，直到任务完成且达到最优解。</font>

<font style="color:rgb(51, 51, 51);">优势在于：</font>

+ <font style="color:rgb(51, 51, 51);">深度优化：通过自我发现和调整策略，Agent 能够深入理解任务的本质，实现更深层次的优化。</font>
+ <font style="color:rgb(51, 51, 51);">适应性强：Agent 能够根据任务的变化和复杂性灵活调整行为策略，增强适应性。</font>
+ <font style="color:rgb(51, 51, 51);">提高效率：通过不断优化任务执行路径，Agent 能够减少不必要的操作，提高任务完成的效率。</font>
+ <font style="color:rgb(51, 51, 51);">提升用户体验：通过提供更高质量的服务，Agent 能够更好地满足用户需求，提升用户体验。</font>

<font style="color:rgb(51, 51, 51);">Self-Discover 模式的交互流程可以用以下图示来表示：</font>

```plain
+-------------------+
|     接收任务      |
+-------------------+
           |
           v
+-------------------+
|     任务分析（Task Analysis）|
+-------------------+
           |
           v
+-------------------+
|     自我发现（Self-Discovery）|
+-------------------+
           |
           v
+-------------------+
|     调整策略（Strategy Adjustment）|
+-------------------+
           |
           v
+-------------------+
| 执行与反馈（Execution & Feedback）|
+-------------------+
           |
           v
+-------------------+
|     循环迭代      |
+-------------------+
```

<font style="color:rgb(51, 51, 51);">示例：</font>

<font style="color:rgb(51, 51, 51);">Self-Discover 模式代码示例，用于实现一个 Agent 优化一个简单的数据分类任务：</font>

```plain
class SelfDiscoverAgent:
    def __init__(self):
        self.knowledge_base = {
            "data": [
                {"feature1": 1, "feature2": 2, "label": "A"},
                {"feature1": 2, "feature2": 3, "label": "B"},
                {"feature1": 3, "feature2": 4, "label": "A"},
                {"feature1": 4, "feature2": 5, "label": "B"}
            ],
            "initial_strategy": "simple_threshold"
        }

    def task_analysis(self, task):
        # 任务分析：识别任务的关键要素和目标
        returnf"任务分析：{task}"

    def self_discovery(self, analysis):
        # 自我发现：发现潜在的问题和改进点
        if self.knowledge_base["initial_strategy"] == "simple_threshold":
            return"发现：初始策略过于简单，可能无法准确分类"
        else:
            return"无需改进"

    def strategy_adjustment(self, discovery):
        # 调整策略：根据自我发现的结果优化行为路径
        if"过于简单"in discovery:
            return"调整策略：采用更复杂的分类算法"
        else:
            return"保持原策略"

    def execute_and_feedback(self, strategy):
        # 执行与反馈：执行任务并收集反馈信息
        if strategy == "调整策略：采用更复杂的分类算法":
            # 假设新策略提高了分类准确率
            return"执行结果：分类准确率提高到90%"
        else:
            return"执行结果：分类准确率60%"

    def run(self, task):
        # 主流程：任务分析 -> 自我发现 -> 调整策略 -> 执行与反馈 -> 循环迭代
        analysis = self.task_analysis(task)
        discovery = self.self_discovery(analysis)
        strategy = self.strategy_adjustment(discovery)
        feedback = self.execute_and_feedback(strategy)
        print(f"Task Analysis: {analysis}")
        print(f"Self-Discovery: {discovery}")
        print(f"Strategy Adjustment: {strategy}")
        print(f"Execution & Feedback: {feedback}")

# 示例运行
agent = SelfDiscoverAgent()
agent.run("数据分类任务")
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
Task Analysis: 任务分析：数据分类任务
Self-Discovery: 发现：初始策略过于简单，可能无法准确分类
Strategy Adjustment: 调整策略：采用更复杂的分类算法
Execution & Feedback: 执行结果：分类准确率提高到90%
```

<font style="color:rgb(51, 51, 51);">Self-Discover 模式如何通过任务分析、自我发现、调整策略和执行反馈的多轮迭代来优化 Agent 的行为。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如机器学习模型优化、智能决策系统等，通过不断优化 Agent 的行为策略，实现更加智能和高效的任务执行。</font>

### <font style="color:rgb(51, 51, 51);">Storm 模式</font>
<font style="color:rgb(51, 51, 51);">Storm 模式原理</font>

<font style="color:rgb(51, 51, 51);">Storm 模式是一种专注于从零开始生成复杂内容的 Agent 设计模式，特别适用于需要系统化构建和优化内容生成的任务，例如生成类似维基百科的文章、报告或知识库。其核心在于通过逐步构建大纲，并根据大纲逐步丰富内容，从而生成高质量、结构化的文本。</font>

<font style="color:rgb(51, 51, 51);">Storm 模式的交互流程：</font>

+ <font style="color:rgb(51, 51, 51);">接收任务：Agent 接收到用户或系统的任务指令，明确需要生成的内容主题。</font>
+ <font style="color:rgb(51, 51, 51);">构建大纲（Outline Construction）：Agent 根据任务主题生成一个详细的大纲，明确内容的结构和各个部分的主题。</font>
+ <font style="color:rgb(51, 51, 51);">内容生成（Content Generation）：Agent 根据大纲逐步生成每个部分的具体内容，确保内容的连贯性和准确性。</font>
+ <font style="color:rgb(51, 51, 51);">内容优化（Content Optimization）：Agent 对生成的内容进行优化，包括语言润色、逻辑调整和信息补充，以提高内容的质量。</font>
+ <font style="color:rgb(51, 51, 51);">循环迭代（Iteration）：Agent 重复内容生成和优化的过程，直到内容满足用户需求或达到预设的质量标准。</font>

<font style="color:rgb(51, 51, 51);">优势在于：</font>

+ <font style="color:rgb(51, 51, 51);">系统化生成：通过构建大纲和逐步填充内容，确保生成内容的结构化和系统性。</font>
+ <font style="color:rgb(51, 51, 51);">高质量输出：通过多轮优化，Agent 能够生成高质量、连贯且准确的内容。</font>
+ <font style="color:rgb(51, 51, 51);">适应性强：适用于多种内容生成任务，包括但不限于文章、报告、知识库等。</font>
+ <font style="color:rgb(51, 51, 51);">可扩展性：可以根据任务的复杂性和需求灵活调整大纲和内容生成策略。</font>

<font style="color:rgb(51, 51, 51);">Storm 模式的交互流程可以用以下图示来表示：</font>

```plain
+-------------------+
|     接收任务      |
+-------------------+
           |
           v
+-------------------+
| 构建大纲（Outline Construction）|
+-------------------+
           |
           v
+-------------------+
| 内容生成（Content Generation）|
+-------------------+
           |
           v
+-------------------+
| 内容优化（Content Optimization）|
+-------------------+
           |
           v
+-------------------+
|     循环迭代      |
+-------------------+
```

<font style="color:rgb(51, 51, 51);">示例：</font>

<font style="color:rgb(51, 51, 51);">Storm 模式代码示例，用于实现一个 Agent 生成一篇关于“人工智能”的维基百科风格文章：</font>

```plain
class StormAgent:
    def __init__(self):
        self.knowledge_base = {
            "topics": {
                "人工智能": {
                    "定义": "人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，旨在创建能够执行复杂任务的智能机器。",
                    "历史": "人工智能的发展可以追溯到20世纪40年代，当时科学家们开始探索如何使计算机模拟人类智能。",
                    "应用": "人工智能在医疗、金融、教育、交通等多个领域都有广泛的应用。",
                    "未来": "未来，人工智能有望在更多领域实现突破，推动社会的智能化发展。"
                }
            }
        }

    def outline_construction(self, topic):
        # 构建大纲：根据主题生成大纲
        outline = [
            "定义",
            "历史",
            "应用",
            "未来"
        ]
        return outline

    def content_generation(self, topic, section):
        # 内容生成：根据大纲部分生成具体内容
        return self.knowledge_base["topics"][topic][section]

    def content_optimization(self, content):
        # 内容优化：对生成的内容进行润色和调整
        optimized_content = content.replace("有望", "有巨大潜力")
        return optimized_content

    def run(self, topic):
        # 主流程：构建大纲 -> 内容生成 -> 内容优化 -> 循环迭代
        outline = self.outline_construction(topic)
        article = {}
        for section in outline:
            content = self.content_generation(topic, section)
            optimized_content = self.content_optimization(content)
            article[section] = optimized_content
        return article

# 示例运行
agent = StormAgent()
article = agent.run("人工智能")
for section, content in article.items():
    print(f"### {section}")
    print(content)
```

<font style="color:rgb(51, 51, 51);">输出结果</font>

```plain
### 定义
人工智能（Artificial Intelligence, AI）是计算机科学的一个分支，旨在创建能够执行复杂任务的智能机器。
### 历史
人工智能的发展可以追溯到20世纪40年代，当时科学家们开始探索如何使计算机模拟人类智能。
### 应用
人工智能在医疗、金融、教育、交通等多个领域都有广泛的应用。
### 未来
未来，人工智能有巨大潜力在更多领域实现突破，推动社会的智能化发展。
```

<font style="color:rgb(51, 51, 51);">Storm 模式如何通过构建大纲、内容生成和内容优化的多轮迭代来生成高质量的文章。这种模式在实际应用中可以扩展到更复杂的任务和场景，例如生成研究报告、知识库条目等，通过不断优化 Agent 的内容生成和优化策略，实现更加智能和高效的任务执行。</font>

## [<font style="color:rgb(66, 185, 131);">本章小结</font>](https://datawhalechina.github.io/hello-agents/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_45-%e6%9c%ac%e7%ab%a0%e5%b0%8f%e7%bb%93)
<font style="color:rgb(52, 73, 94);">在本章中，以第三章掌握的大语言模型知识为基础，我们通过“亲手造轮子”的方式，从零开始编码实现了三种业界经典的智能体构建范式:ReAct、Plan-and-Solve 与 Reflection。我们不仅探索了它们的核心工作原理，还通过具体的实战案例，深入了解了各自的优势、局限与适用场景。</font>

**<font style="color:rgb(44, 62, 80);">核心知识点回顾:</font>**

1. <font style="color:rgb(52, 73, 94);">ReAct:我们构建了一个能与外部世界交互的 ReAct 智能体。通过“思考-行动-观察”的动态循环，它成功地利用搜索引擎回答了自身知识库无法覆盖的实时性问题。其核心优势在于</font>**<font style="color:rgb(44, 62, 80);">环境适应性</font>**<font style="color:rgb(52, 73, 94);">和</font>**<font style="color:rgb(44, 62, 80);">动态纠错能力</font>**<font style="color:rgb(52, 73, 94);">，使其成为处理探索性、需要外部工具输入的任务的首选。</font>
2. <font style="color:rgb(52, 73, 94);">Plan-and-Solve:我们实现了一个先规划后执行的 Plan-and-Solve 智能体，并利用它解决了需要多步推理的数学应用题。它将复杂的任务分解为清晰的步骤，然后逐一执行。其核心优势在于</font>**<font style="color:rgb(44, 62, 80);">结构性</font>**<font style="color:rgb(52, 73, 94);">和</font>**<font style="color:rgb(44, 62, 80);">稳定性</font>**<font style="color:rgb(52, 73, 94);">，特别适合处理逻辑路径确定、内部推理密集的任务。</font>
3. <font style="color:rgb(52, 73, 94);">Reflection (自我反思与迭代):我们构建了一个具备自我优化能力的 Reflection 智能体。通过引入“执行-反思-优化”的迭代循环，它成功地将一个效率较低的初始代码方案，优化为了一个算法上更优的高性能版本。其核心价值在于能</font>**<font style="color:rgb(44, 62, 80);">显著提升解决方案的质量</font>**<font style="color:rgb(52, 73, 94);">，适用于对结果的准确性和可靠性有极高要求的场景。</font>

<font style="color:rgb(52, 73, 94);">本章探讨的三种范式，代表了智能体解决问题的三种不同策略，如表4.1所示。在实际应用中，选择哪一种，取决于任务的核心需求:</font>

![表 4.1 不同 Agent Loop 的选择策略](https://cdn.nlark.com/yuque/0/2025/png/8420697/1762877817344-93ae063c-40b2-42b7-aa7a-c5eaa9269c65.png)

<font style="color:rgb(52, 73, 94);">至此，我们已经掌握了构建单个智能体的核心技术。为了过渡知识，以及对实际应用更加深入。下一节我们将会探索不同低代码平台的使用方式以及轻代码构建agent的方案。</font>

