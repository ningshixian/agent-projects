# Module 7: SmolAgents - Lightweight Agent Framework

## Overview
Learn SmolAgents, HuggingFace's lightweight and flexible framework for building AI agents with minimal dependencies and maximum customization.

## Learning Objectives
- Understand the philosophy of lightweight agent frameworks
- Build custom agents with minimal overhead
- Implement tool integration with SmolAgents
- Create domain-specific agent architectures
- Compare lightweight vs heavy frameworks for different use cases

## Topics Covered

### 1. SmolAgents Philosophy
- Lightweight vs heavyweight frameworks
- Simplicity and customization benefits
- When to choose SmolAgents over alternatives
- Installation and setup

### 2. Core Components
- Agent base classes and interfaces
- Tool system and integration
- Memory management approaches
- Message handling and conversation flow

### 3. Agent Creation Patterns
- Code agents for programming tasks
- Web agents for internet interaction
- Multimodal agents with vision capabilities
- Custom agent development

### 4. Tool Development
- Creating custom tools
- Tool composition and chaining
- Error handling in tools
- Performance optimization

### 5. Advanced Patterns
- Agent orchestration
- Custom execution environments
- Integration with HuggingFace models
- Deployment strategies

## Hands-On Activities
1. **First SmolAgent**: Create a basic agent with tools
2. **Code Agent**: Build programming assistant
3. **Custom Tools**: Develop domain-specific tools
4. **Multi-Agent System**: Orchestrate multiple SmolAgents
5. **Performance Comparison**: Compare with heavier frameworks

## Files in This Module
- `smolagents_basics.py` - Core concepts and first agent
- `code_agents.py` - Programming-focused agents
- `custom_tools.py` - Tool development examples
- `multi_agent_setup.py` - Multi-agent orchestration
- `performance_comparison.py` - Framework benchmarking
- `exercises/` - Hands-on coding exercises

## Key Advantages

### Lightweight Design
- Minimal dependencies
- Fast startup and execution
- Low memory footprint
- Easy deployment and distribution

### Flexibility
- Highly customizable architecture
- Easy to extend and modify
- Clear separation of concerns
- Framework-agnostic tool integration

### Simplicity
- Clean, readable codebase
- Straightforward APIs
- Minimal boilerplate code
- Easy debugging and maintenance

## Agent Types

### Code Agent
```python
from smolagents import CodeAgent
agent = CodeAgent(tools=[python_interpreter, file_editor])
```

### Web Agent
```python
from smolagents import ToolCallingAgent
agent = ToolCallingAgent(tools=[web_search, web_scraper])
```

### Multimodal Agent
```python
from smolagents import MultiModalAgent
agent = MultiModalAgent(tools=[image_analyzer, text_processor])
```

## Tool Ecosystem

### Built-in Tools
- **Python interpreter**: Code execution
- **Web search**: Internet information retrieval
- **Image processing**: Computer vision tasks
- **File operations**: File system interaction

### Custom Tools
```python
from smolagents import Tool

class CustomTool(Tool):
    name = "custom_analyzer"
    description = "Analyze custom data formats"

    def __call__(self, data: str) -> str:
        # Custom logic here
        return f"Analyzed: {data}"
```

## Comparison with Other Frameworks

| Feature | SmolAgents | LangChain | CrewAI | OpenAI API |
|---------|------------|-----------|---------|------------|
| Dependencies | Minimal | Heavy | Medium | None |
| Customization | High | Medium | Low | High |
| Learning Curve | Low | Medium | Medium | Low |
| Performance | Fast | Medium | Medium | Fast |
| Community | Growing | Large | Medium | Large |

## Use Cases

### Ideal For
- **Rapid prototyping**: Quick agent development
- **Resource-constrained environments**: Minimal overhead
- **Custom architectures**: Unique agent designs
- **Educational purposes**: Learning agent concepts
- **Edge deployment**: Lightweight deployment needs

### Consider Alternatives For
- **Complex workflows**: LangGraph might be better
- **Enterprise features**: LangChain ecosystem
- **Multi-agent collaboration**: CrewAI specialization
- **Established patterns**: Well-tested frameworks

## Integration Patterns

### With HuggingFace Models
```python
from transformers import pipeline
from smolagents import Agent

class HFAgent(Agent):
    def __init__(self):
        self.model = pipeline("text-generation", model="gpt2")
```

### With Local Models
```python
import ollama
from smolagents import Agent

class LocalAgent(Agent):
    def __init__(self):
        self.client = ollama.Client()
```

### With Other Frameworks
```python
# SmolAgent as a component in larger systems
from smolagents import ToolCallingAgent
from langchain.tools import Tool

def create_smolagent_tool():
    agent = ToolCallingAgent(tools=[...])

    return Tool(
        name="smolagent",
        description="Lightweight agent for specific tasks",
        func=agent.run
    )
```

## Performance Characteristics

### Startup Time
- **SmolAgents**: ~100ms
- **LangChain**: ~1-2s
- **CrewAI**: ~500ms-1s

### Memory Usage
- **SmolAgents**: 50-100MB
- **LangChain**: 200-500MB
- **CrewAI**: 150-300MB

### Execution Speed
- **Simple tasks**: SmolAgents faster
- **Complex workflows**: Framework-dependent
- **Tool calling**: Comparable performance

## Architecture Patterns

### Single Agent
```python
# Simple, focused agent
agent = CodeAgent(tools=[interpreter])
result = agent.run("Write a function to sort a list")
```

### Multi-Agent Coordination
```python
# Lightweight orchestration
specialist_agents = {
    "coder": CodeAgent(tools=[interpreter]),
    "researcher": WebAgent(tools=[search]),
    "analyzer": DataAgent(tools=[analyzer])
}

def coordinate(task):
    # Simple coordination logic
    return specialist_agents["coder"].run(task)
```

### Tool Composition
```python
# Composable tool chains
tools = [
    PythonInterpreter(),
    FileEditor(),
    WebSearch(),
    DataAnalyzer()
]

agent = ToolCallingAgent(tools=tools)
```

## Prerequisites
- Understanding of agent concepts
- Python programming experience
- HuggingFace ecosystem familiarity (helpful)
- Completed foundational modules

## Next Steps
After completing this module, proceed to Module 8: DSPy to learn about declarative self-improving language programs.

# SmolAgents 项目整理

## 项目概述

本项目是 HuggingFace SmolAgents 框架的学习和实践项目，旨在帮助开发者掌握轻量级 AI Agent 的构建方法。

## 整理后的目录结构

```
07-smolagents/
├── core/                    # 核心模块
│   ├── agent.py            # 主要的 Agent 实现
│   └── tools.py            # 自定义工具
├── examples/               # 示例代码
│   ├── basic_agent.py      # 基础 Agent 示例
│   ├── custom_tools.py     # 自定义工具示例
│   └── advanced_features.py # 高级功能示例
├── tutorials/              # 教程文档
│   ├── building_good_agents.md
│   └── ...
├── exercises/              # 练习代码
│   └── smolagents_comprehensive_exercise.py
├── app.py                  # 主应用
├── requirements.txt        # 依赖列表
└── README.md               # 项目说明
```

## 核心组件说明

### 1. 基础概念
- `CodeAgent`: 基于代码执行的 Agent
- `ToolCallingAgent`: 基于工具调用的 Agent
- `@tool` 装饰器: 用于创建自定义工具

### 2. 模型支持
- HuggingFace Inference API
- OpenAI/Anthropic (通过 LiteLLM)
- 本地模型 (Transformers)

### 3. 安全执行
- E2B 沙箱
- Docker 容器
- 其他安全执行环境

## 使用方法

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 运行示例:
```bash
python examples/basic_agent.py
```

3. 启动应用:
```bash
python app.py
```

