## 低代码框架 / Agentic Workflow
+ <font style="color:rgb(51, 51, 51);">新手快速开发→ Coze</font>
+ <font style="color:rgb(51, 51, 51);">多系统自动化→ n8n</font>
+ <font style="color:rgb(51, 51, 51);">企业知识库搭建→ FastGPT 或 RAGFlow</font>
+ <font style="color:rgb(51, 51, 51);">可扩展企业应用→ Dify</font>
+ <font style="color:rgb(51, 51, 51);">高要求RAG效果→ RAGFlow</font>

| <font style="color:rgb(51, 51, 51);">Agent框架类型</font> | <font style="color:rgb(51, 51, 51);">分析</font> | <font style="color:rgb(51, 51, 51);">热门框架</font> |
| --- | --- | --- |
| **<font style="color:rgb(51, 51, 51);">低代码</font>**<font style="color:rgb(51, 51, 51);">框架</font> | <font style="color:rgb(51, 51, 51);">无需写代码即可完成</font> | <font style="color:rgb(51, 51, 51);">Coze、Dify、n8n、RAGFlow、</font>**<font style="color:rgb(31, 35, 40);">TARS</font>** |




## AI Agent Frameworks
| **序号** | **<font style="color:rgb(74, 85, 101);">框架名称</font>** | **<font style="color:rgb(74, 85, 101);">学习成本</font>** | **基础 Agent 能力（Function Call、流式输出等）** | <font style="color:rgb(51, 51, 51);">应用场景</font> |
| :--- | :--- | :--- | :--- | :--- |
| 1 | [<font style="color:rgb(9, 105, 218);">OpenAI Agents SDK</font>](https://github.com/openai/openai-agents-python) | <font style="color:rgb(51, 51, 51);">简单。轻量级框架，封装程度低，易于理解和使用。</font> | <font style="color:rgb(74, 85, 101);">提供基本的 Agent 功能，支持 function call 和流式输出，适用于需要简单代理功能的应用。</font> | <font style="color:rgb(74, 85, 101);">独立功能开发、原型验证</font> |
| 2 | <font style="color:rgb(51, 51, 51);"></font>[LangGraph](https://langchain-ai.github.io/langgraph/) | <font style="color:rgb(51, 51, 51);">中等。采用图结构定义任务节点与关系，适合复杂工作流，但学习曲线较陡。</font> | <font style="color:rgb(74, 85, 101);">支持复杂工作流，包括循环、迭代等，适用于需要高级内存功能和人机交互的应用。</font> | <font style="color:rgb(74, 85, 101);">复杂工作流程、多轮对话</font> |
| 3 | [AutoGen](https://github.com/microsoft/autogen) | <font style="color:rgb(51, 51, 51);">中等。由微软开发，支持多代理通信，适用于复杂任务，但可能需要一定学习成本。</font> | <font style="color:rgb(74, 85, 101);">支持对话和任务分配，具备 function call 和流式输出能力</font> | <font style="color:rgb(74, 85, 101);">多代理协作、复杂任务分解</font> |
| 4 | <font style="color:rgb(51, 51, 51);"></font>[CrewAI](https://github.com/joaomdmoura/CrewAI) | <font style="color:rgb(51, 51, 51);">简单。采用角色扮演的方式，易于上手，适合初学者。</font> | <font style="color:rgb(31, 35, 40);">强调多代理协作，适用于需要团队协作的任务，支持 function call 和流式输出。CrewAI 不依赖外部依赖项，因此更精简、更快速、更简洁。</font> | <font style="color:rgb(31, 35, 40);"></font> |
| 5 | [CAMEL-AI（OWL）](https://github.com/camel-ai/owl) | <font style="color:rgb(51, 51, 51);">中等。专为自主和沟通代理研究设计，支持多模态处理，但可能需要一定学习成本。</font> | <font style="color:rgb(31, 35, 40);">具备处理多模态数据的能力，支持 function call 和流式输出，强调多智能体协作、支持复杂任务分解与动态交互</font> | <font style="color:rgb(31, 35, 40);">具身智能</font> |
| 6 | <font style="color:rgb(51, 51, 51);"></font>[smolagents](https://github.com/huggingface/smolagents) | <font style="color:rgb(51, 51, 51);">简单。极简轻量级 Agent 框架</font> | CodeAgent 代码驱动、执行效率高、安全沙箱、与 huggingface 深度集成！支持基本的工具调用和任务分解 | <font style="color:rgb(51, 51, 51);">快速原型开发和教学</font>，复杂应用需额外扩展 |
| 7 | [agno](https://github.com/agno-agi/agno) | <font style="color:rgb(51, 51, 51);">较难。主打“通用智能体操作系统”，强调安全可控和可扩展。</font> | 号称构建 Agentic Systems “最简单、最快、功能最全的开源项目。内置任务编排、内存、RAG 等能力，支持多模型协作 | 需要可扩展性和安全性的生产级 Agent 应用 |




## Agent 框架选型建议
+ 初学者：从 `OpenAI Agents SDK` 或 `smolagent` 开始，轻量级且易学习。
+ 企业应用：`AutoGen` 或 `langgraph` 更合适，提供完整的企业级功能。
+ 原型验证：`OpenAI Agents SDK` 、`SmolAgents`、`Agnos`
+ 分布式应用：`uagents` 去中心化特性使其在这些场景中具有独特优势。



## 教程 / 课程
- [ ] [台大李宏毅 2025 AI Agent 新课来了！](https://www.bestblogs.dev/article/0576e6)
- [ ] **CAMEL-AI** 框架：[Handy Multi-Agent Tutorial](https://fmhw1n4zpn.feishu.cn/docx/AF4XdOZpIo6TOaxzDK8cxInNnCe)
- [x] [huggingface Agents Course](https://huggingface.co/learn/agents-course/zh-CN/unit0/introduction)：系统学习 AI 智能体的理论架构、设计原理与实践应用，掌握主流 AI 智能体开发库的使用，包括 [**smolagents**](https://huggingface.co/docs/smolagents/en/index)、 LangChain 和 LlamaIndex.
- [ ] [~~AI Agents for Beginners~~](https://github.com/microsoft/ai-agents-for-beginners)~~：微软出的 AI Agent 新手教程，使用 AutoGen 以及 semantic-kernel~~
- [ ] [any-agent/cookbook](https://mozilla-ai.github.io/any-agent/cookbook/your_first_agent/)<font style="color:rgb(15, 20, 25);">:</font>**<font style="color:rgb(15, 20, 25);"> </font>**<font style="color:rgb(15, 20, 25);">由 Mozilla AI 开源，为多种 AI Agent 框架提供统一接口（</font>![](https://camo.githubusercontent.com/6c57c35364d764a573626d98713bb578d2ace59249f9bfdcc807508ff1242a02/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f476f6f676c6525323041444b2d3432383546343f6c6f676f3d676f6f676c65266c6f676f436f6c6f723d7768697465)<font style="color:rgb(31, 35, 40);"> </font>![](https://camo.githubusercontent.com/9ca2781ee921ead9b8a24fbbb4d3f498516ddace6f313e1329a5a3bd69c6ab4d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c616e67436861696e2d3165343534353f6c6f676f3d6c616e67636861696e266c6f676f436f6c6f723d7768697465)<font style="color:rgb(31, 35, 40);"> </font>![](https://camo.githubusercontent.com/47e58a6771c5c597825f8880adec4ef6091c40f487d36ecea152c1e7c55c56da/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f2546302539462541362539392532304c6c616d61496e6465782d666263666532)<font style="color:rgb(31, 35, 40);"> </font>![](https://camo.githubusercontent.com/f85e2e9f61919fd4a727025a8c26367dc7aeb591ae3fba36ba49488fc584748f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4f70656e41492532304167656e74732d626c61636b3f6c6f676f3d6f70656e6169)<font style="color:rgb(31, 35, 40);"> </font>![](https://camo.githubusercontent.com/8423dc1deb7bba724f5c8cb602a7aeb8b72cf15a343f5453c1e4658395fad67d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f536d6f6c6167656e74732d6666636233613f6c6f676f3d68756767696e6766616365266c6f676f436f6c6f723d7768697465)<font style="color:rgb(31, 35, 40);"> </font>![](https://camo.githubusercontent.com/7398f32a47079e3f529a8ab8adaa4368179bdfdf8dc9374369d46723b7de95ae/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f54696e794167656e74732d6666636233613f6c6f676f3d68756767696e6766616365266c6f676f436f6c6f723d7768697465)<font style="color:rgb(31, 35, 40);"> </font>[<font style="color:rgb(9, 105, 218);">Agno AI</font>](https://docs.agno.com/introduction)<font style="color:rgb(15, 20, 25);">），简化了开发者在不同框架间切换的复杂性，并提供评估工具 </font>

![](https://cdn.nlark.com/yuque/0/2025/png/8420697/1755592837272-94fc6a76-b594-4ce9-a789-271d79e4a0d8.png)



## 参考
+ [AI agent frameworks by example.](https://www.codeagents.dev/) Take a look at the different frameworks and choose the one that best fits your needs. Or take the best parts and build it yourself in your favorite language.
+ [awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents)





