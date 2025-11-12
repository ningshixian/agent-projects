> 内部分享学习：[关于多智能体的一些思考与复盘 - 飞书](https://li.feishu.cn/docx/VfvsdbSyvo2qdXxuH4mcTCLFnTd?from=from_copylink)
>



> 转载：[a-visual-guide-to-llm-agents](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-llm-agents)
>
> 翻译：[图解 LLM Agent——从基础到高级概念](https://mp.weixin.qq.com/s/QFJyS0TUCv-TT39isRLu3w)
>



## <font style="color:#000000;">Agent 定义</font>
过去这一年，AI 圈关于 Agent 的讨论可以说吵得天花乱坠，除了老生常谈的 LLM、Tool，大家还发明了一堆新名词，什么记忆、编排、规划、反思……光是看这些概念图，估计不少人第一眼就头大，第二眼直接选择性失忆。但是 Agent 到底如何定义却成为了一件雾里看花的事情。

比如下面，Anthropic 对「智能体」的最新定义是：一个真正的智能体必须在内部独立完成任务：**<font style="color:#ED740C;">「智能体能够动态地决定自己的执行流程和工具使用方式，自主掌控任务的完成过程」</font>**；而翁丽莲（Weng Lilian）博文([LLM Powered Autonomous Agents](https://zhuanlan.zhihu.com/p/639964649))详细介绍了Agent架构 = LLM + Planning + Feedback + Tool use；

![https://www.anthropic.com/engineering/building-effective-agents](https://cdn.nlark.com/yuque/0/2025/png/8420697/1755584842776-0afbebdf-00b0-43a1-b116-a41c942cef66.png)

[《AI 下半场，聊一聊 Agent 本质与变革》](https://mp.weixin.qq.com/s/0X58GZqPHbN9Sw5begW__w)，这篇推文算是比较清晰的定义了 Agent，中国 Agent 开发者的入门必读文章。

### Agent 是什么？
:::warning
**<font style="color:#ED740C;">Agent 定义：Tools in a loop to achieve a goal.</font>**

:::

<font style="color:#000000;">要了解agent是什么，首先来探索LLM的基本能力。传统上，LLM不过是一个接一个地进行下一个token预测。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186382592-7f5ced2f-7676-4cba-b324-6049fc7a43dd.webp)

<font style="color:#000000;">通过连续采样许多token，可以模拟对话，并使用LLM对问题给出更详细的答案。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383016-dc4a64f5-1a74-4fe0-9e13-ac7b6f9688f4.webp)

然而，当我们继续"对话"时，任何 LLM 都会显示其主要缺点：**<font style="color:#ED740C;">即如果不依赖对话系统将整个对话历史作为上下文传入模型，那么模型就不会记得对话内容。</font>**

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186382986-fe37a3c4-379e-4718-8f09-fdf660d78fcd.webp)

<font style="color:#000000;">还有许多其他任务是LLM经常失败的，包括基本的数学运算，如乘法和除法：</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383015-5d22de8b-0d27-4a31-85e3-640c2ac885c0.webp)

**<font style="color:#000000;">这是否意味着LLM很糟糕？</font>**

**<font style="color:#000000;">当然不是！</font>**

**<font style="color:#ED740C;">没有必要让LLM具备一切能力，因为可以借助外部工具、记忆和检索系统来弥补它们的不足。</font>**

通过外部系统，LLM 的能力可以得到增强。Anthropic 将这称为"**<font style="color:#ED740C;">增强型大模型</font>**"**<font style="color:#ED740C;">（The Augmented LLM）</font>**。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383244-0a04bfa1-1d77-4f63-8836-2f80d8e96eaf.webp)

<font style="color:#000000;">例如，面对一个数学问题时，LLM可能会决定使用适当的工具（计算器）。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383370-8db9b8e1-f7e9-4b9a-99da-76029b366baa.webp)

> <font style="color:#000000;">那么，这种“</font>**<font style="color:#000000;">增强型LLM</font>**<font style="color:#000000;">”是不是agent呢？</font>
>

<font style="color:#2F4BDA;">不完全是，但也可以说有一点。</font>

Agent 的进化路线，从基础的"**<font style="color:#ED740C;">增强型大模型</font>**"**<font style="color:#ED740C;">（The Augmented LLM）</font>**开始，逐步发展到预定义的**<font style="color:#ED740C;">workflow</font>**，最终形成自主的 **<font style="color:#ED740C;">agent</font>**。

![具有不同自主程度的LLM Agent](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383822-9fe6d203-624d-42bd-8559-c948b8a02461.webp)

:::color3
所以你会发现，真正落到开发实践里，其实只要在 LLM 外面接个工具，再加上最基础的循环逻辑，立马就能跑起来，一个基础 Agent 就具备雏形了。至于后面那些复杂的调度、记忆、上下文管理，本质上都是“选配”，目的是让 Agent 更聪明、更稳定，但绝不是构成 Agent 的前提条件。说到底，LLM 和 Tools 才是 Agent 最朴素、也是最不可缺少的“标配”。

:::

:::warning
**<font style="color:#ED740C;">Agent 定义：Tools in a loop to achieve a goal.</font>**

:::



### Agent 的设计框架
接着讨论第二个观点，Agent 的三种设计框架是：The Augmented LLM、Workflow、Agent

1. <font style="color:#000000;">增强型 LLM 是Agent的基本构建单元，它集成了检索、工具和记忆等能力。</font>
2. <font style="color:#000000;">对于可预测、定义明确的任务，需要一致性时使用</font>**<font style="color:#000000;">「工作流」（workflows）</font>**<font style="color:#000000;">：也就是基于</font>**<font style="color:#ED740C;">「预先编排好的提示词与工具路径」</font>**<font style="color:#000000;">构成的工作流智能体，主打稳定可靠！</font>
3. <font style="color:#000000;">相比于 workflow，</font>**<font style="color:#000000;">Agent</font>**<font style="color:#000000;"> 的设计反而是很简单。</font>**<font style="color:#ED740C;">背后依靠强大的推理模型，让模型自己去理解复杂输入、进行推理和规划、使用工具以及从错误中反思</font>**<font style="color:#000000;">，所谓</font>**模型即服务**<font style="color:#000000;">。其核心在于模型主导全部决策过程（模型能够在 in a loop 中自主处理问题，无须人为定制某个step+prompt干预），人类无需预先定义详细流程，只需在特定情况下提供少量干预，如在Deep Research中我们只需要确认关键的信息或在Operator中当输入账号密码等敏感信息需要人类干预。</font>[模型即产品：万字详解RL驱动的AI Agent模型如何巨震AI行业范式](https://www.tsingtaoai.com/newsinfo/8186867.html)<font style="color:#DF2A3F;"> </font>

<font style="color:#DF2A3F;"></font>

**Workflow vs. Agent**

在这里简单聊一聊Workflow 和 Agent 的区别。

:::warning
+ <font style="color:rgb(51, 51, 51);">Agentic Workflows（代理工作流）：</font><u><font style="color:rgb(51, 51, 51);">基于预定义代码路径的应用</font></u><font style="color:rgb(51, 51, 51);">，流程固定且不可变。简单、可预测，适合明确任务。它们依赖硬编码规则，不赋予LLM真正的“自主权”，很多人误将其称为agent。</font>
+ <font style="color:rgb(51, 51, 51);">Agents（代理）：具备</font><u><font style="color:rgb(51, 51, 51);">自主规划、动态调整</font></u><font style="color:rgb(51, 51, 51);">能力，依赖模型驱动决策，灵活且可扩展。</font>

:::

| **框架名称** | **分类** | **工具调用方式** | **编排逻辑** | **示例平台** |
| --- | --- | --- | --- | --- |
| Workflow | 静态 / 动态 | 以工作流编排为核心，将 LLM 对话能力和工具调用作为节点加入到工作流中完成任务<br/>![](https://cdn.nlark.com/yuque/0/2025/png/8420697/1755585941440-26e34f21-a636-4665-84af-5114fbdfca04.png) | 静态流程图或节点规则 | Dify、Coze、Manus |
| Agent | single / multi | 通过自我循环的方式自主规划并调用工具完成任务<br/>![Agent 完全体](https://cdn.nlark.com/yuque/0/2025/png/8420697/1755586203909-9bd41dd5-1525-44fd-9ab4-ec710b5bc296.png) | 无显式编排，由模型自行推理 | Openai Deep Research |




1. **Workflow**

**Workflow**** 本质上是一种白盒系统**，所谓白盒，指的是系统过程完全透明，开发者和用户都清楚每一个节点、每一条路径，结果是怎么一步步推导出来的，基本一目了然。

在传统的 Workflow 里，LLM 通常只是流程中的“工具人”，负责具体的小任务，比如总结、翻译、分类这些相对独立、风险可控的环节。但现在，Workflow 本身也开始分化出两种形态——<font style="background-color:#F9EFCD;">静态和动态</font>。

[Building Effective Agents: 不做全场景、保持简单，像Agent一样思考](https://www.yuque.com/ningshixian/epwt8c/mv3seys80btnszdl)

区别在于，流程的分支选择到底是死板地写死在代码里的，还是交给 LLM 根据实际情况动态决策。

![](https://cdn.nlark.com/yuque/0/2025/svg/8420697/1755587936731-2832078c-7c01-4896-b0f7-9d069e08e854.svg)

这里需要强调一点：**Workflow 并不是一种“低级版”的 Agent**，别小看它。在很多业务里，Workflow 有它非常合理的存在价值，尤其是那些流程路径清晰、需要稳定输出、但又不完全想放弃智能决策能力的场景。



2. **Agent**

**相**比之下，Agent 的结构更为简单：大模型独自承担整个任务流程，自己调用工具，自己做规划，自己完成执行。**简单的背后，对大模型的链路理解和长流程规划能力要求反而更高，需要超高质量的训练数据+极致的端到端强化训练。**所有 if-else 和 workflow 的选择将由模型自身判断完成，而非依赖人工编写的规则代码。

![](https://cdn.nlark.com/yuque/0/2025/svg/8420697/1755588724134-c876f41f-fa5e-4877-bc48-e86a6f02e98f.svg)

+ `单智能体` = `大语言模型`（LLM） + `观察`（obs） + `思考`（thought） + `行动`（act） + `记忆`（mem）
+ `多智能体` = 多个`智能体` + `环境` + `SOP` + `评审` + `通信` + `成本`



### 现状
目前整个 Agent 行业正处在**从传统的 ****Workflow**** 逐步往 Agent 形态演进的阶段**。很多传统的流程、系统都在尝试“加一点 AI”，比如，原本靠正则表达式提取关键词的环节，改用 LLM 来理解语义、提取信息；原本数据分析系统只能出图、出表，结果全靠人去理解；现在加上 LLM，系统可以自动生成摘要、趋势分析、甚至给出风险提示。这些看似只是小改造，实质上也是 AI 介入的一种形式。

很多人觉得，这还算不上真正的 Agent，认为只有多角色、多轮协作那种才叫 Agent。但其实也没必要这么狭隘理解，凡是让 LLM 参与具体任务、赋能原有流程，本质上就是 Agent 思路的一部分，区别只是“含 AI 量”多还是少。就像我们不会因为某台电动车纯电续航短，就说它不是电动车一样，路径可以多元，趋势是一样的。



## Agent 的关键要素
<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">LLM Agent 的三个主要组件：</font>**<font style="color:rgb(15, 76, 129);background-color:#FBF5CB;">记忆</font>**<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">、</font>**<font style="color:rgb(15, 76, 129);background-color:#FBF5CB;">工具</font>**<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">和</font>**<font style="color:rgb(15, 76, 129);background-color:#FBF5CB;">规划</font>**

:::color3
+ 记忆（Memory）
    - 短期记忆：<font style="color:#000000;">（近期）即时上下文的缓冲区</font>
    - 长期记忆：这为代理提供了在长时间保留和回忆（无限）信息的能力，通常通过利用外部向量存储和快速检索来实现。
+ 工具的使用（Tool use）
    - 学会调用外部API获取模型权重中缺失的额外信息 Function call
    - 通过 MCP 创建和管理工具
+ 规划（Planning）
    - 推理 - 子目标与分解
    - <font style="color:rgb(63, 63, 63);">推理 + 行动执行 ReAct</font>
    - 反思与完善（Reflection and refinement）

:::

<font style="color:#000000;">为了选择要采取哪些行动，LLM agent 需要一个</font><u><font style="color:#000000;">至关重要的组件：</font></u>**<u><font style="color:#000000;">规划能力</font></u>**<font style="color:#000000;">。为此，LLM需要能够通过诸如chain of thought等方法进行“推理”和“思考”。所谓</font>**<u><font style="color:#000000;"> Planning through reasoning</font></u>**

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383572-e5b5c03a-566c-478c-b707-d2b5b7939c39.webp)

关于推理型 LLMs，请查看这篇文章《图解推理型 LLM》→ [3 PoLMs for Reasoning](https://www.yuque.com/ningshixian/epwt8c/gmwxwoui0ez9u99d?singleDoc#)

利用这种推理行为，LLM Agent 将规划出必要的行动步骤。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383699-f0a10b35-c100-4438-8169-0fafd22f14fa.webp)

这种规划行为使 Agent 能够理解情况（LLM）、规划下一步（规划）、采取行动（工具）并跟踪已采取的行动（记忆）。

![Agent 核心 3 个组件](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383832-63729729-2469-4eb6-b871-d7b0d7251050.webp)

<font style="background-color:#FBF5CB;">下面</font><font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">，我们将通过 LLM Agent 的三个主要组件：</font>**<font style="color:rgb(15, 76, 129);background-color:#FBF5CB;">记忆</font>**<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">、</font>**<font style="color:rgb(15, 76, 129);background-color:#FBF5CB;">工具</font>**<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">和</font>**<font style="color:rgb(15, 76, 129);background-color:#FBF5CB;">规划</font>**<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;">，讨论各种自主行为方法。</font>

<font style="color:rgb(63, 63, 63);background-color:#FBF5CB;"></font>

## <font style="color:#000000;">记忆Memory </font>
:::color3
+ <font style="color:rgb(51, 51, 51);">全量记忆：不遗忘任何内容</font>
+ <font style="color:rgb(51, 51, 51);">滑动窗口：固定长度的截断</font>
+ <font style="color:rgb(51, 51, 51);">相关性过滤：遗忘次要信息</font>
+ <font style="color:rgb(51, 51, 51);">摘要/压缩：提炼关键信息</font>
+ <font style="color:rgb(51, 51, 51);">向量数据库：语义检索记忆</font>
+ <font style="color:rgb(51, 51, 51);">知识图谱：结构化记忆</font>
+ <font style="color:rgb(51, 51, 51);">分层记忆：短期与长期结合</font>
+ <font style="color:rgb(51, 51, 51);">类OS内存管理：模拟Swap原理</font>

:::

<font style="color:#000000;">LLM是健忘的系统，或者更准确地说，在与它们互动时，它们根本不进行任何记忆。</font>

<font style="color:#000000;">例如，</font>**<font style="color:#000000;">当向LLM提问，然后接着问另一个问题时，它不会记得前者</font>**<font style="color:#000000;">。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186383847-35a0cc38-ca5e-4095-b502-dc6bcd15fcd2.webp)

<font style="color:#000000;">通常将其称为</font>**<font style="color:#ED740C;">短期记忆</font>**<font style="color:#000000;">，也称为工作记忆，它作为（近期）即时上下文的缓冲区。这包括LLM agent最近采取的行动。</font>

<font style="color:#000000;">然而，LLM agent还需要跟踪</font>**<font style="color:#000000;">可能多达数十个步骤</font>**<font style="color:#000000;">，而不仅仅是最近的行动。也就是说，Agent 工作时需要</font>**<font style="color:#ED740C;">长期记忆</font>**<font style="color:#000000;">，因为LLM agent理论上可能需要记住多达数十个甚至数百个步骤。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384135-e2ae77a3-c699-4352-bfe7-f483f6106a04.webp)

### <font style="color:#000000;">短期记忆</font>
实现短期记忆的最直接方法是使用模型的上下文窗口，即 LLM 可以处理的 token 数量。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384245-6610f1be-9a32-42bb-a5c7-f0dd6c6a249e.webp)

上下文窗口通常至少为 8192 个token，有时甚至可以扩展到数十万个 token.

大型上下文窗口，可用于将完整的对话历史作为输入 prompt 的一部分进行跟踪。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384293-290ed553-7961-48e2-919a-d5057de54738.webp)

只要对话历史适合 LLM 的上下文窗口，这种方法就能有效模拟记忆。

但是，这并非真正记住对话，而是在"告诉"LLM这个对话是什么。

对于上下文窗口较小的模型，或者当对话历史较大时，我们可以使用另一个LLM来总结迄今为止发生的对话。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384245-576c4d1a-19ca-4f80-b044-5ad97be26d05.webp)

通过持续总结对话，我们可以保持较小的对话规模。这将减少 token 数量，同时只跟踪最重要的信息。

### <font style="color:#000000;">长期记忆</font>
<font style="color:rgb(63, 63, 63);">LLM Agent 的长期记忆包括需要长期保留的 Agent 过去的行动空间。</font>

> 行动空间：指的是 Agent 过去所有的操作、决策和互动记录，而不仅仅是静态的数据或信息。
>

**<font style="color:rgb(15, 76, 129);">实现长期记忆的常见技术是将所有先前的交互、行动和对话存储在外部向量数据库中。</font>**

<font style="color:rgb(63, 63, 63);">要构建这样的数据库，首先将对话嵌入到能够捕捉其含义的数值表示中。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384495-65e6a6af-17b3-462c-b336-b2e2b3cad561.webp)

构建数据库后，我们可以嵌入任何给定的提示，并通过比较提示嵌入与数据库嵌入来找到向量数据库中最相关的信息。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384470-006e0505-1b9d-4a54-91af-74429ab79332.webp)

这种方法也就是**<font style="color:#ED740C;">检索增强生成</font>**（Retrieval-Augmented Generation，RAG）。

## <font style="color:#000000;">工具Tools </font>
### 工具的类型
工具允许给定的 LLM 与外部环境（如数据库）交互或使用外部应用程序（如运行自定义代码）。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384803-5bcab23d-ea79-4081-a2bc-3a8846835968.webp)

<font style="color:#000000;">工具通常有两种用例：</font>**<font style="color:#ED740C;">获取数据以检索最新信息</font>****<font style="color:#000000;"> 以及 </font>****<font style="color:#ED740C;">采取行动</font>****<font style="color:#000000;">（如安排会议或订购食物）</font>**

### <font style="color:#000000;">工具使用的底层原理</font>
我们常说，大模型可以调用工具，但它到底是怎么“知道”自己该用哪个工具，又是怎么真的把任务一步步完成的？很多人对这个过程是模糊的，只知道“它好像能查天气、算数学、读文献”，但真要落地一个Agent系统，搞清楚它到底**怎么调工具**，是必不可少的第一步。

1. 首先，**你得准备一个工具。**工具可以是任意一个提供服务的API，比如一个可以查询天气的接口。这一步是开发者的活，需要去找一个靠谱的API。
2. 然后，**把工具“翻译”成模型能理解的说明书。**Agent框架会把这个API转换成一段标准的 JSON 结构，里面写明：这个工具是干嘛的、有哪些参数、返回什么结果。就像写了份使用指南发给模型。
3. **用户提问，比如：“长沙的天气怎么样？”**Agent系统在收到这个问题时，并不会马上让模型回答，而是会把**<font style="background-color:rgba(255,246,122,0.8);">用户的问题</font>**<font style="background-color:rgba(255,246,122,0.8);"> + </font>**<font style="background-color:rgba(255,246,122,0.8);">工具说明</font>**一起打包发给模型。
4. **大模型看到后，决定“调用工具”。**它不是乱猜答案，而是读完工具说明后，生成一段“调用请求”，告诉系统：“我要用那个天气查询工具，省份是湖南，城市填长沙。”
5. **Agent框架收到这个请求，就去真正调用****API****了。**它把“湖南”和“长沙”这个两个参数传给天气API，等结果返回，再把这个结果发回给大模型。
6. 最后，**大模型根据返回结果，生成回复。**比如：“长沙当前的天气情况是多云，温度为36度，湿度52%。”

这个过程看起来挺简单，其实背后完成了**一次完整的“模型调用外部工具”流程**。模型不再只是对输入做语言生成，而是通过工具**接入现实世界**，主动执行动作，拿到数据，再做出判断和回应。

<font style="color:#000000;"></font>

<font style="color:#000000;">要实际使用工具，LLM 必须生成符合给定工具 API 的文本。我们通常期望生成可以格式化为 JSON 的字符串，以便它能够轻松地输到代码解释器中。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384695-1b805665-fd2d-423d-b591-cee2da06612b.webp)

<font style="color:#000000;">请注意，这不仅限于JSON，也可以直接在代码中调用工具。</font>

<font style="color:#000000;">还可以生成LLM可以使用的自定义函数，如基本的乘法函数，这通常被称为函数调用，也就是</font>**<font style="color:#ED740C;">Function call</font>**<font style="color:#000000;">。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384841-6df8b712-0f9b-44b3-ad35-93b27d063028.webp)

如果提示词足够准确，一些 LLM 可以使用任何工具。工具使用是大多数当前 LLM 都具备的能力。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384981-be08ba28-095f-40be-aafa-34339c2cacb7.webp)

**<font style="color:#ED740C;">访问工具的更稳定方法是通过微调 LLM</font>**（稍后会详细介绍）。

如果代理框架是固定的，工具可以按照特定顺序使用；

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186384997-02625f96-8553-45d5-b433-a3010ddcafc1.webp)

或者 LLM 可以自主选择使用哪种工具以及何时使用。

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186385083-ffc8eeb3-4bc9-44ed-af7e-a49c09ceac2f.webp)

<font style="color:rgb(63, 63, 63);">LLM 调用序列的中间步骤，会被反馈回 LLM 以继续处理。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1750411731858-67b61749-e338-457c-b582-502e19743604.webp)

<font style="color:rgb(63, 63, 63);">可以认为，</font>**<font style="color:rgb(15, 76, 129);">LLM Agent，本质上是 LLM 调用的序列</font>**<font style="color:rgb(63, 63, 63);">（但具有自主选择行动/工具等的能力）。</font>

### **<font style="color:rgb(38, 38, 38);">自主工具调用能力</font>**
<font style="color:rgba(0, 0, 0, 0.9);">合适的工具调用能够拓展大模型的能力边界，但现有工具调用方式将大模型限制在预设的工具使用框架内，更像是一个被动的「提线木偶」，而非真正具备主动性的智能体。主要体现在以下几个方面：</font>

+ <font style="color:rgba(0, 0, 0, 0.9);"> </font>**<font style="color:rgb(61, 170, 214);">浅层模仿而非深度理解：</font>**<font style="color:rgba(0, 0, 0, 0.9);">SFT 只是学会了特定场景下工具调用的表面模式，而非真正理解工具的功能边界、适用场景和内部工作机制 </font>
+ **<font style="color:rgb(61, 170, 214);">上下文依赖性强：</font>**<font style="color:rgba(0, 0, 0, 0.9);">基于 Prompt 的工具调用方法高度依赖于提示的精确性和完整性。一旦用户描述模糊或提示设计不当，模型就无法正确选择和使用工具 </font>
+ **<font style="color:rgb(61, 170, 214);">工具组合能力受限：</font>**<font style="color:rgba(0, 0, 0, 0.9);">当需要多个工具协同解决复杂问题时，现有方法难以支持模型进行灵活的工具组合</font>

<font style="color:rgba(0, 0, 0, 0.9);">复旦大学知识工场实验室团队在开源项目 </font>[SimpleGRPO](https://github.com/lsdefine/simple_GRPO/tree/main/Auto_Program )<font style="color:rgba(0, 0, 0, 0.9);"> 中开源实现了大模型自主工具调用机制，通过引入大模型的深度思考能力，从根本上重构了大模型工具调用的范式。简单说就是 </font><font style="color:rgba(0, 0, 0, 0.9);background-color:#FBDE28;">CodeAct</font><font style="color:rgba(0, 0, 0, 0.9);">？</font>

> 2025.04.17 [https://zhuanlan.zhihu.com/p/1896148817021215732](https://zhuanlan.zhihu.com/p/1896148817021215732)
>

**<font style="color:rgb(17, 24, 39);">两大核心实现方案</font>**

+ **<font style="color:rgb(17, 24, 39);">【边想边干】（Think-and-Act）</font>**<font style="color:rgb(44, 44, 54);">：</font>
    - <font style="color:rgb(44, 44, 54);">大模型在生成推理链的过程中，当意识到需要计算或分析时，会</font>**<font style="color:rgb(17, 24, 39);">直接生成代码片段</font>**<font style="color:rgb(44, 44, 54);">（如Python）。</font>
    - <font style="color:rgb(44, 44, 54);">系统执行代码，将</font>**<font style="color:rgb(17, 24, 39);">运行结果无缝插入</font>**<font style="color:rgb(44, 44, 54);">到模型的推理流中，供其继续思考和调整后续步骤。</font>
+ **<font style="color:rgb(17, 24, 39);">【专业分工】（Specialized Collaboration）</font>**<font style="color:rgb(44, 44, 54);">：</font>
    - <font style="color:rgb(44, 44, 54);">大模型在推理中</font>**<font style="color:rgb(17, 24, 39);">明确提出需求</font>**<font style="color:rgb(44, 44, 54);">（如“我需要计算标准差”）。</font>
    - <font style="color:rgb(44, 44, 54);">一个</font>**<font style="color:rgb(17, 24, 39);">专门的代码生成模型</font>**<font style="color:rgb(44, 44, 54);">接收需求，生成精确的代码并执行。</font>
    - <font style="color:rgb(44, 44, 54);">执行结果返回给主模型，由其整合并推进推理。这利用了“专业的人做专业的事”的思想。</font>

### <font style="color:rgb(25, 27, 31);">MCP（Model Context Protocol）</font>
> [MCP (Model Context Protocol)](https://www.yuque.com/ningshixian/epwt8c/bsveq9dc28g610ah)
>

:::info
MCP 实现了工具的解耦，<font style="color:rgb(25, 27, 31);">解决了工具调用、Prompt模板、资源访问等标准化问题</font>

:::

<font style="color:rgb(63, 63, 63);">背景：工具是代理框架的重要组成部分，当存在多种不同API时，启用工具使用变得麻烦，因为任何工具都需要：</font>

+ <font style="color:rgb(63, 63, 63);">手动跟踪并输入到LLM中</font>
+ <font style="color:rgb(63, 63, 63);">手动描述（包括其预期的JSON schema）</font>
+ <font style="color:rgb(63, 63, 63);">每当API发生变化时，手动更新</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186385716-4fb4104b-5559-4374-b1f4-24025c1524d4.webp)

<font style="color:#000000;">所以，为了使工具更容易在任何给定的agent框架中实现，Anthropic开发了</font>**<font style="color:#ED740C;">模型上下文协议（MCP）</font>****<font style="color:#000000;">。</font>**

<font style="color:#000000;">它由三个组件组成：</font>

:::info
+ <font style="color:rgb(63, 63, 63);">MCP Host（宿主） — LLM 应用程序（如 Cursor）负责管理连接;</font>
+ <font style="color:rgb(63, 63, 63);">MCP Client（客户端） — 维护与 MCP 服务器的 1:1 连接;</font>
+ <font style="color:rgb(63, 63, 63);">MCP Server（服务器） — 向 LLMs 提供上下文、工具和功能;</font>

:::

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186385666-f9421182-fc98-4132-a02c-40ccbf698e48.webp)

<font style="color:#000000;">例如，假设希望某个LLM应用总结仓库中的最新5次提交。</font>

<font style="color:rgb(63, 63, 63);">MCP Host（与 MCP Client一起）</font><font style="color:#000000;">将首先调用MCP服务器，询问哪些工具可用。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186385779-4ab2e611-91aa-4abe-a3af-41317b3ef168.webp)

<font style="color:#000000;">LLM接收到信息后，可能会选择使用工具。它通过</font><font style="color:rgb(63, 63, 63);">Host</font><font style="color:#000000;">向 </font><font style="color:rgb(63, 63, 63);">MCP Server</font><font style="color:#000000;"> 发送请求，然后接收结果，包括使用的工具。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186385907-726e2b50-4249-4b08-8396-a2e0cdc7114a.webp)

<font style="color:#000000;">最后，LLM接收到结果，并可以向用户解析答案。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386074-d3c27418-dff4-4863-a370-48c7a11a8e9b.webp)

<font style="color:rgb(63, 63, 63);">这个框架通过连接到任何 LLM 应用程序都可以使用的 MCP Servers，使创建工具变得更加简单。因此，当你创建一个与 Github 交互的 MCP Server时，任何支持 MCP 的LLM 应用程序都可以使用它。</font>

## <font style="color:#000000;">规划Planning </font>
<font style="color:rgb(63, 63, 63);">工具使用允许 LLM 增强其能力。它们通常通过类 JSON 请求进行调用。</font>

<font style="color:rgb(63, 63, 63);">但在 Agent 系统中，LLM 如何决定使用哪个工具以及何时使用呢？这就非常考验 LLM 模型的关键能力了，这里主要有三个：</font>

+ **指令遵循能力（ICL）**
+ **工具使用能力（Tool Use）**
+ **规划反思与推理能力（****<font style="color:#000000;">Planning</font>****）**

<font style="color:rgb(63, 63, 63);">规划反思与推理能力总的来说可以理解为模型的智商，有的模型比较笨遇到问题不会尝试其他的办法，进入死循环，有的模型能够灵活运用 各种方法来尝试解决问题。能力上的差距决定了 Agent 的上限，聪明的基座模型可以帮你解决一些关键性的问题，比如生成复杂 SQL，编写代码绘制图表等。</font>

**<font style="color:#000000;">Planning </font>**<font style="color:rgb(63, 63, 63);">涉及：</font>

1. **<font style="color:#DF2A3F;">将给定任务分解为可执行的步骤（CoT、ToT.....）</font>**
2. **<font style="color:#DF2A3F;">自我反思和自我修正（Reflection and Refinement）</font>**

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386136-96f8f730-27b3-41c5-b33d-e33403ef43cb.webp)

<font style="color:#000000;">为了在LLM agent中启用规划，先来看看这种技术的基础，即</font>**<font style="color:#ED740C;">推理能力</font>**<font style="color:#000000;">。</font>

### <font style="color:#000000;">推理 Reasoning </font>
<font style="color:rgb(63, 63, 63);">规划可执行步骤需要复杂的推理行为。因此，LLM 必须能够展示这种行为，然后才能进行任务规划的下一步。</font>

<font style="color:rgb(63, 63, 63);">"推理型"LLM是那些倾向于在回答问题前先"思考"的模型。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386220-9b92e71e-e5e2-4e7b-af11-1a43de9bc165.webp)

_<font style="color:rgb(63, 63, 63);">这里对"推理-reasoning"和"思考-thinking"这两个术语使用得比较宽松，因为我们可以讨论这是否真的类似于人类思考，或者仅仅是将答案分解为结构化步骤。</font>_

<font style="color:rgb(63, 63, 63);">这种推理行为大致可以通过两种选择来实现：微调LLM或特定的提示工程（prompt engineering）。</font>

<font style="color:rgb(63, 63, 63);">通过提示工程，我们可以创建 LLM 应遵循的推理过程示例。提供示例（也称为少样本提示，few-shot prompting）是引导 LLM 行为的一种优秀方法。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386245-a5ea4e48-4481-4d20-a4c1-3d731eb71b64.webp)

<font style="color:rgb(63, 63, 63);">这种提供思考过程示例的方法被称为</font>[<font style="color:rgb(36,91,219);">思维链</font>](https://zhida.zhihu.com/search?content_id=230345068&content_type=Article&match_order=1&q=%E6%80%9D%E7%BB%B4%E9%93%BE&zhida_source=entity)<font style="color:rgb(63, 63, 63);">（</font>[Chain of thought](https://link.zhihu.com/?target=https%3A//lilianweng.github.io/posts/2023-03-15-prompt-engineering/%23chain-of-thought-cot)<font style="color:rgb(63, 63, 63);">，CoT； </font>[Wei et al. 2022](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2201.11903)<font style="color:rgb(63, 63, 63);">），它能够实现更复杂的推理行为。</font>

<font style="color:rgb(63, 63, 63);">思维链也可以在没有任何示例（零样本提示，zero-shot prompting）的情况下实现，只需简单地说明"让我们一步步思考"。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386735-d5785887-cf5f-4a5b-a87d-ea748b63abad.webp)

<font style="color:rgb(63, 63, 63);">在训练 LLM 时，我们可以给它提供足够数量包含思考类示例的数据集，或者 LLM 可以发现自己的思考过程，比如使用强化学习。</font>

**<font style="color:#ED740C;">DeepSeek-R1是一个很好的例子，它使用奖励机制来引导思考过程的使用。</font>**

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386869-83efba8b-c2e2-4682-9981-dc335f3db12d.webp)

### <font style="color:#000000;">推理和行动 </font><font style="color:#DF2A3F;">ReAct</font>
<font style="color:rgb(63, 63, 63);">在LLM中启用推理行为很好，但这并不一定使其能够规划可执行的步骤。</font>

<font style="color:rgb(63, 63, 63);">迄今为止我们关注的技术要么展示推理行为，要么通过工具与环境交互。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386747-58b9ddb1-b1a5-4c61-9703-20015df08caa.webp)

<font style="color:#ED740C;">例如，思维链（Chain-of-Thought）纯粹专注于推理。</font>

<font style="color:#ED740C;">最早将这两个过程结合起来的技术之一被称为 ReAct（Reason and Act）。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186386779-7a0b5bda-fa48-462d-9f2f-03702aea2932.webp)

<font style="color:#DF2A3F;">ReAct</font><font style="color:rgb(63, 63, 63);">（</font>[Yao et al. 2023](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2210.03629)<font style="color:rgb(63, 63, 63);">）通过精心设计的提示工程来实现这一点。</font>

<font style="color:rgb(63, 63, 63);">ReAct并不复杂，它只是一种提示技巧。我们为模型提供工具，然后指导它思考是否应该使用这些工具来完成给定任务。ReAct提示描述了三个步骤：</font>

+ <font style="color:rgb(63, 63, 63);"></font>**<font style="color:rgb(15, 76, 129);">思考（Thought）</font>**<font style="color:rgb(63, 63, 63);"> - 关于当前情况的推理步骤</font>
+ <font style="color:rgb(63, 63, 63);"></font>**<font style="color:rgb(15, 76, 129);">行动（Action）</font>**<font style="color:rgb(63, 63, 63);"> - 要执行的一系列行动（例如，使用工具）</font>
+ <font style="color:rgb(63, 63, 63);"></font>**<font style="color:rgb(15, 76, 129);">观察（Observation）</font>**<font style="color:rgb(63, 63, 63);"> - 关于行动结果的推理步骤</font>

<font style="color:#000000;">提示本身相当简单，如下：</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186387155-ab3efa8b-18e3-40d3-9f26-3c8112242d13.webp)

<font style="color:rgb(63, 63, 63);">LLM使用这个提示（可作为系统提示使用）来引导其行为，在思考、行动和观察的循环中工作。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186387199-7115006a-8d14-4617-9254-c58f0bcb4c4c.webp)

<font style="color:rgb(63, 63, 63);">它会一直保持这种行为，直到某个行动指示返回结果。通过对思考和观察的迭代，LLM 可以规划行动，观察其输出，并相应地进行调整。</font>

<font style="color:rgb(63, 63, 63);">因此，与那些预定义固定步骤的代理相比，这个框架使 LLMs 能够展示更加自主的代理行为。</font>

### <font style="color:#000000;">反思 Reflection</font>
<font style="color:rgb(63, 63, 63);">没有人，甚至采用 ReAct 的LLM，能在每个任务上都表现出色。失败在所难免，关键是从中反思，以推动成长。</font>

<font style="color:rgb(63, 63, 63);">这个过程在 ReAct 中缺失，而这正是 </font><font style="color:#000000;">Reflection</font><font style="color:rgb(63, 63, 63);"> 发挥作用的地方。</font>

**<font style="color:#ED740C;">一些案例：</font>**

+ `<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">Self-refine</font>`<font style="color:rgb(25, 27, 31);">: 利用迭代过程，包括生成、反馈和精炼。在每次生成后，LLM为计划产生反馈，促进基于反馈的调整。</font>
+ <font style="color:rgb(25, 27, 31);"></font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">Reflexion</font>`<font style="color:rgb(25, 27, 31);">: 扩展 </font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">ReAct</font>`<font style="color:rgb(25, 27, 31);"> 方法，通过引入</font>**<font style="color:rgb(25, 27, 31);">评估器</font>**<font style="color:rgb(25, 27, 31);">来评估轨迹。LLM 检测到错误时生成自我反思，帮助纠正错误。</font>
+ `<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">CRITIC</font>`<font style="color:rgb(25, 27, 31);">: 使用</font>**<font style="color:rgb(25, 27, 31);">外部</font>**<font style="color:rgb(25, 27, 31);">工具，如</font>**<font style="color:rgb(25, 27, 31);">知识库</font>**<font style="color:rgb(25, 27, 31);">和</font>**<font style="color:rgb(25, 27, 31);">搜索引擎</font>**<font style="color:rgb(25, 27, 31);">，验证LLM生成的动作。然后利用外部知识进行自我纠正，显著减少事实错误。</font>
+ <font style="color:rgb(25, 27, 31);"></font>`<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">InteRecAgent</font>`<font style="color:rgb(25, 27, 31);">: 使用称为ReChain的自我纠正机制。LLM用于评估由交互推荐代理生成的响应和工具使用计划，总结错误反馈，并决定是否重新规划。</font>
+ `<font style="color:rgb(25, 27, 31);background-color:rgb(248, 248, 250);">LEMA</font>`<font style="color:rgb(25, 27, 31);">: 首先收集错误的规划样本，然后使用更强大的GPT-4进行纠正。纠正后的样本随后用于</font>**<font style="color:rgb(25, 27, 31);">微调LLM代理</font>**<font style="color:rgb(25, 27, 31);">，从而在各种规模的LLaMA模型上实现显著的性能提升。</font>

![Self-refine](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186387243-f2f42557-cf0a-4fa8-91d2-1a0c2078958c.webp)

<font style="color:rgb(63, 63, 63);">同一个LLM负责生成初始输出、精炼后的输出和反馈。</font>

![Self-refine](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1744186387457-ac9aa31c-32ac-4283-be20-aba45b785deb.webp)

<font style="color:rgb(63, 63, 63);">有趣的是，这种自我反思行为，无论是Reflexion还是SELF-REFINE，都与强化学习非常相似，在强化学习中，基于输出质量给予奖励。</font>



## <font style="color:#000000;">模块化框架选型</font>
说到框架，市面上各个公司出的都大差不差，主要是侧重点不同。有的框架很注重角色设计，比如CrewAI。有的框架很在意流程设计，比如LangGraph。**当框架过于在意多智能体的抽象设计时，可能会制约我们创新的想法，毕竟Agent的发展日新月异，我们不能确保框架的某些设计理念永远都是正确的。所以当时在选择框架时我优先考虑的是越轻量越好、要足够的简洁，这样开发和调试的时候理解成本就不会太高，改起来也简单。**

<u>这里对比较流行的几个框架做了对比：</u>[Agent 框架梳理](https://www.yuque.com/ningshixian/epwt8c/ag8zco5a70doksvs)

<font style="color:rgb(63, 63, 63);"></font>

<font style="color:rgb(63, 63, 63);">无论你选择哪种框架创建多智能体系统，这些框架通常由多个要素组成，包括智能体的配置文件、对环境的感知、记忆、规划以及可用的行动。</font>

![](https://cdn.nlark.com/yuque/0/2025/webp/8420697/1750413119579-ae112e9d-62d7-4356-8a80-4329f7ebfcb1.webp)

<font style="color:rgb(63, 63, 63);"></font>

## <font style="color:rgb(38, 38, 38);">挑战</font>
<font style="color:rgb(38, 38, 38);">在经历了构建以LLM为中心的代理人的关键思想和演示之后，我开始看到一些共同的局限性：</font>

+ <font style="color:rgb(38, 38, 38);">有限的上下文长度（Finite context length）：受限的上下文容量限制了历史信息、详细指令、API调用上下文和响应的包含。系统的设计必须在这种有限的通信带宽下运作，而像自我反思这样的机制可以从长或无限的上下文窗口中获益良多。虽然向量存储和检索可以提供对更大知识库的访问，但它们的表示能力不如全注意力强大</font>
+ <font style="color:rgb(38, 38, 38);">长期规划和任务分解中的挑战（Challenges in long-term planning and task decomposition）：在漫长的历史中进行规划并有效地探索解决方案空间仍然具有挑战性。语言模型很难在面对意外错误时调整计划，使其相对于能够通过试错学习的人类来说更加脆弱。</font>
+ <font style="color:rgb(38, 38, 38);">自然语言接口的可靠性（Reliability of natural language interface）：当前的代理系统依赖于自然语言作为LLMs与内存和工具等外部组件之间的接口。然而，模型输出的可靠性值得怀疑，因为LLMs可能会出现格式错误，并偶尔表现出叛逆行为（例如，拒绝遵循指令）。因此，大部分代理演示代码都集中在解析模型输出上。</font>

