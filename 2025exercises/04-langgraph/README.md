# Module 4: LangGraph - Advanced Workflow Orchestration

## Overview
Learn LangGraph, LangChain's extension for building complex, stateful, multi-actor applications with cycles and branching logic.

## Learning Objectives
- Understand stateful agent workflows and graph-based execution
- Build complex multi-step processes with conditional branching
- Implement human-in-the-loop patterns and approval flows
- Create persistent, resumable agent workflows
- Handle errors and retries in complex agent systems

## Topics Covered

### 1. LangGraph Fundamentals
- Graph-based workflow concepts
- Nodes, edges, and state management
- Difference from simple chains
- Installation and setup

### 2. State Management
- StateGraph and state schemas
- State persistence across nodes
- State transformations and updates
- Conditional state routing

### 3. Node Types and Patterns
- Function nodes vs agent nodes
- Conditional edges and routing
- Parallel execution patterns
- Human-in-the-loop nodes

### 4. Complex Workflows
- Multi-agent orchestration
- Approval and review processes
- Error handling and retry logic
- Workflow checkpointing and resume

### 5. Advanced Features
- Custom state reducers
- Dynamic graph construction
- Streaming and real-time updates
- Integration with external systems

## Hands-On Activities
1. **First Graph**: Build a simple linear workflow
2. **Conditional Workflow**: Implement branching logic based on conditions
3. **Multi-Agent Graph**: Orchestrate multiple specialized agents
4. **Human-in-Loop**: Create approval-based workflow
5. **Error Recovery**: Implement robust error handling

## Files in This Module
- `langgraph_basics.py` - Core concepts and simple graphs
- `stateful_workflows.py` - State management examples
- `multi_agent_graphs.py` - Complex multi-agent orchestration
- `human_in_loop.py` - Interactive workflow patterns
- `error_handling.py` - Robust error recovery patterns
- `exercises/` - Hands-on coding exercises

## Key Concepts

### Graph Components
- **Nodes**: Individual steps in the workflow
- **Edges**: Connections between nodes with routing logic
- **State**: Shared data structure passed between nodes
- **Checkpoints**: Persistence points for workflow resumption

### Workflow Patterns
- **Sequential**: Linear execution through nodes
- **Conditional**: Dynamic routing based on state
- **Parallel**: Concurrent execution of multiple branches
- **Cyclic**: Loops and iterative processes
- **Human-Interactive**: Human approval and input points

## Prerequisites
- Completed Module 3: LangChain
- Understanding of state machines and workflow concepts
- LangGraph package installed

## Next Steps
After completing this module, proceed to Module 5: Google ADK to explore Google's AI development platform.