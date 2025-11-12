# Module 9: Parlant Conversation Framework

## Overview
Parlant是一个LLM-原生的「会话设计引擎」、「生产级 AI 智能体框架」，带前端UI、可直接部署
——通过行为准则（Guidelines） 和对话旅程（Journeys） 这两个核心抽象——确保复杂对话代理可靠地遵循业务协议。

### 对话旅程 (Conversational Journeys)
- 是什么：用于定义多步骤、结构化的交互流程，引导用户完成特定目标（如预订航班、处理退款、技术支持排障等）
- 如何工作：仍然是构造dialogue flow，但不同于僵硬的对话流程图，Parlant 的旅程允许智能体根据上下文动态调整流程，跳过、重复或提前进入某些步骤，保持对话的自然性。 
- 示例：商品咨询→规格确认→促销说明→订单创建的完整对话流

### 动态上下文管理 (Dynamic Context Management)
- 智能地管理session会话上下文窗口，只加载与当前对话最相关的准则和旅程信息，减轻模型的“认知负担”，确保高效、准确的决策。

### Dialogue Management
- 对话状态跟踪
- Flow control and routing
- 上下文切换
- 对话持久性

### Natural Language Understanding
- Intent classification
- Entity extraction
- Slot filling
- Confidence scoring

### 6. Advanced Features
- 多轮对话
- 对话分支
- 回退处理
- 与其他系统的集成

### 全链路可解释性 (Full Explainability)
- Parlant 为智能体的每一个决策提供详细的追溯和解释。开发者可以清晰看到哪条准则被触发、为什么被触发、以及是如何被遵循的。这极大方便了调试、审计和合规性检查。


## Learning Objectives
- Master conversational agent design principles
- Implement natural dialogue flows
- Handle context and conversation state
- Build domain-specific conversation agents

## Hands-On Activities
1. **Basic Chatbot**: Build a simple conversational agent
2. **Context Management**: Implement persistent conversation memory
3. **Domain-Specific Agent**: Create a customer service bot
4. **Multi-Modal Chat**: Add image/file handling capabilities

## Files in This Module
- `parlant_basics.py` - Core conversation concepts
- `dialogue_management.py` - Conversation flow examples
- `context_handlers.py` - Context and memory management
- `domain_agents.py` - Specialized conversation agents
- `exercises/` - Hands-on coding exercises

## Use Cases Covered
- Customer support chatbots
- Personal assistant agents
- Educational conversation systems
- Entertainment and gaming bots
- Multi-language conversation agents

## Integration Patterns
- Combining with LangChain tools
- CrewAI conversation agents
- API-based conversation services
- Voice interface integration

## Prerequisites
- Completed previous modules
- Understanding of conversation design
- API integration experience

## Next Steps
After completing this module, proceed to Module 10: Integration for combining multiple frameworks effectively.