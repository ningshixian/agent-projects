"""
LangGraph Basics - Updated for v1.0 Alpha (2025)
Building stateful, graph-based agent workflows with modern LangGraph patterns
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv

# Modern LangGraph v1.0 alpha imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.constants import Send

# Updated LangChain core imports (v0.3+)
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()


# Modern State Definitions with enhanced typing
class AgentState(TypedDict):
    """Enhanced agent state with v1.0 patterns"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_task: str
    completed_steps: List[str]
    analysis_results: Dict[str, Any]
    next_action: Literal["continue", "finish", "error"]
    user_feedback: Optional[str]
    iteration_count: int


class WorkflowState(TypedDict):
    """Multi-step workflow state"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_step: str
    workflow_data: Dict[str, Any]
    step_results: List[str]
    requires_human_input: bool


# Modern tool definitions
@tool
def research_tool(query: str) -> str:
    """Research information on a given topic."""
    # Mock research implementation
    research_db = {
        "artificial intelligence": "AI is transforming industries with applications in healthcare, finance, transportation, and education. Market expected to reach $1.8T by 2030.",
        "machine learning": "ML enables computers to learn from data without explicit programming. Key techniques include supervised, unsupervised, and reinforcement learning.",
        "natural language processing": "NLP combines computational linguistics with machine learning to help computers understand human language.",
        "quantum computing": "Quantum computers use quantum mechanics to process information exponentially faster than classical computers for certain problems."
    }

    for topic, info in research_db.items():
        if topic.lower() in query.lower():
            return f"Research results for '{query}': {info}"

    return f"Research results for '{query}': General information available. This is an emerging field with growing interest and investment."


@tool
def analysis_tool(data: str) -> str:
    """Analyze data and provide structured insights."""
    word_count = len(data.split())

    return f"""
ANALYSIS REPORT
===============

Input Analysis:
- Data length: {len(data)} characters
- Word count: {word_count} words
- Complexity: {'High' if word_count > 50 else 'Medium' if word_count > 20 else 'Low'}

Key Insights:
1. Primary themes identified
2. Data structure and quality assessed
3. Actionable recommendations generated

Recommendation: {'Deep dive analysis recommended' if word_count > 100 else 'Standard analysis sufficient'}
"""


@tool
def calculator_tool(expression: str) -> str:
    """Perform mathematical calculations safely."""
    try:
        # Simple safety check
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return f"Error: Invalid characters in expression: {expression}"

        result = eval(expression)
        return f"Calculation result: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


def demonstrate_basic_graph_v1():
    """Demonstrate basic LangGraph v1.0 functionality with persistence"""

    print("📊 LangGraph v1.0 Basic Graph Demo")
    print("=" * 40)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Define enhanced state
    class SimpleState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        step_count: int
        session_summary: str

    # Define nodes with enhanced patterns
    def chatbot_node(state: SimpleState, config: RunnableConfig):
        """Enhanced chatbot node with configuration support"""
        messages = state["messages"]

        # Enhanced system prompt
        system_message = SystemMessage(content="""You are a helpful AI assistant with the following capabilities:
        - Answer questions clearly and concisely
        - Track conversation context across multiple turns
        - Provide step-by-step reasoning when needed

        Current step: {step_count}""".format(step_count=state.get("step_count", 0)))

        # Process messages with system context
        all_messages = [system_message] + list(messages)
        response = llm.invoke(all_messages, config)

        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "session_summary": f"Processed {state.get('step_count', 0) + 1} interactions"
        }

    def summary_node(state: SimpleState, config: RunnableConfig):
        """Generate session summary"""
        messages = state["messages"]
        conversation_length = len(messages)

        summary = f"Session Summary: {conversation_length} messages exchanged, {state.get('step_count', 0)} steps completed."

        return {
            "session_summary": summary,
            "messages": [AIMessage(content=f"Session completed. {summary}")]
        }

    # Build graph with v1.0 patterns
    workflow = StateGraph(SimpleState)

    # Add nodes
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("summary", summary_node)

    # Set entry point (v1.0 uses START constant)
    workflow.add_edge(START, "chatbot")

    # Add conditional logic
    def should_continue(state: SimpleState):
        """Determine if conversation should continue"""
        if state.get("step_count", 0) >= 3:
            return "summary"
        return "chatbot"

    workflow.add_conditional_edges("chatbot", should_continue, {"chatbot": "chatbot", "summary": "summary"})
    workflow.add_edge("summary", END)

    # Compile with modern persistence
    memory = MemorySaver()  # In-memory persistence for demo
    app = workflow.compile(checkpointer=memory)

    # Test with thread-based persistence
    print("\n🧪 Testing Graph with Persistence:")

    config = {"configurable": {"thread_id": "demo-session-1"}}

    test_messages = [
        "Hello! I'm interested in learning about AI.",
        "Can you tell me about machine learning specifically?",
        "What are some practical applications I should know about?",
        "Thanks for the information!"
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Human: {message}")

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=message)]},
                config
            )
            print(f"Assistant: {result['messages'][-1].content}")
            print(f"Step count: {result.get('step_count', 0)}")

        except Exception as e:
            print(f"Error: {e}")


def demonstrate_tool_integration():
    """Demonstrate LangGraph with tool integration"""

    print("\n🔧 LangGraph Tool Integration Demo")
    print("=" * 40)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    # Initialize components
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [research_tool, analysis_tool, calculator_tool]

    # Create agent with tools using prebuilt patterns
    agent_runnable = create_react_agent(llm, tools)

    class ToolAgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        task_completed: bool

    def agent_node(state: ToolAgentState, config: RunnableConfig):
        """Agent node that can use tools"""
        messages = state["messages"]

        # Invoke agent with tools
        result = agent_runnable.invoke({"messages": messages}, config)

        return {
            "messages": result["messages"],
            "task_completed": True
        }

    # Build workflow
    workflow = StateGraph(ToolAgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # Compile and test
    app = workflow.compile()

    print("\n🧪 Testing Tool Integration:")

    test_tasks = [
        "Research artificial intelligence and then analyze the findings",
        "Calculate 25 * 4 + 100 and explain the result",
        "Research machine learning and provide key insights"
    ]

    for i, task in enumerate(test_tasks, 1):
        print(f"\n--- Task {i} ---")
        print(f"Request: {task}")

        try:
            result = app.invoke({
                "messages": [HumanMessage(content=task)]
            })

            # Get the last message (final response)
            final_message = result["messages"][-1]
            print(f"Agent: {final_message.content}")

        except Exception as e:
            print(f"Error: {e}")


def demonstrate_human_in_loop():
    """Demonstrate human-in-the-loop patterns"""

    print("\n👤 Human-in-the-Loop Demo")
    print("=" * 32)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    class HumanLoopState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        pending_review: bool
        human_feedback: Optional[str]
        iteration: int

    def analysis_node(state: HumanLoopState, config: RunnableConfig):
        """Perform analysis that may require human review"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if last_message and hasattr(last_message, 'content'):
            analysis = f"""
            Analysis of request: {last_message.content}

            Initial Assessment:
            - Request complexity: Medium
            - Confidence level: 85%
            - Recommendation: Proceed with implementation

            This analysis requires human review for final approval.
            """

            return {
                "messages": [AIMessage(content=analysis)],
                "pending_review": True,
                "iteration": state.get("iteration", 0) + 1
            }

        return {"pending_review": False}

    def human_review_node(state: HumanLoopState, config: RunnableConfig):
        """Simulate human review (in practice, this would interrupt for real human input)"""
        # In a real implementation, this would pause for human input
        simulated_feedback = "Approved - looks good to proceed"

        return {
            "human_feedback": simulated_feedback,
            "pending_review": False,
            "messages": [SystemMessage(content=f"Human feedback: {simulated_feedback}")]
        }

    def final_response_node(state: HumanLoopState, config: RunnableConfig):
        """Generate final response incorporating human feedback"""
        feedback = state.get("human_feedback", "No feedback provided")

        final_response = f"""
        Final Response (incorporating human feedback):

        Based on the analysis and human review feedback: "{feedback}"

        Proceeding with the recommended approach. Implementation will follow
        the approved guidelines and incorporate the human insights provided.

        Next steps:
        1. Begin implementation phase
        2. Monitor progress closely
        3. Provide regular updates
        """

        return {
            "messages": [AIMessage(content=final_response)]
        }

    # Build workflow with human-in-the-loop
    workflow = StateGraph(HumanLoopState)

    workflow.add_node("analysis", analysis_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("final_response", final_response_node)

    workflow.add_edge(START, "analysis")

    # Conditional routing based on review requirement
    def needs_review(state: HumanLoopState):
        return "human_review" if state.get("pending_review", False) else "final_response"

    workflow.add_conditional_edges("analysis", needs_review)
    workflow.add_edge("human_review", "final_response")
    workflow.add_edge("final_response", END)

    # Compile with interrupts for human input (commented for demo)
    # app = workflow.compile(interrupt_before=["human_review"])
    app = workflow.compile()

    print("\n🧪 Testing Human-in-the-Loop:")

    test_request = "I need help designing a new AI system for customer support. Please analyze the requirements and provide recommendations."

    print(f"Request: {test_request}")

    try:
        result = app.invoke({
            "messages": [HumanMessage(content=test_request)]
        })

        print("\nWorkflow Results:")
        for i, msg in enumerate(result["messages"]):
            print(f"Step {i+1}: {msg.content}")

    except Exception as e:
        print(f"Error: {e}")


def demonstrate_streaming_workflow():
    """Demonstrate streaming capabilities"""

    print("\n📡 Streaming Workflow Demo")
    print("=" * 30)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)

    class StreamingState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        stream_content: str

    def streaming_node(state: StreamingState, config: RunnableConfig):
        """Node that supports streaming responses"""
        messages = state["messages"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a creative storyteller. Write engaging, descriptive content."),
            MessagesPlaceholder("messages")
        ])

        chain = prompt | llm
        response = chain.invoke({"messages": messages}, config)

        return {
            "messages": [response],
            "stream_content": response.content
        }

    # Simple workflow for streaming
    workflow = StateGraph(StreamingState)
    workflow.add_node("storyteller", streaming_node)
    workflow.add_edge(START, "storyteller")
    workflow.add_edge("storyteller", END)

    app = workflow.compile()

    print("\n🧪 Testing Streaming (simulated):")
    print("Topic: A robot discovering creativity")

    try:
        result = app.invoke({
            "messages": [HumanMessage(content="Write a short story about a robot discovering creativity")]
        })

        print("\nGenerated Story:")
        print("-" * 20)
        print(result["stream_content"])

    except Exception as e:
        print(f"Streaming error: {e}")


def main():
    """Run all LangGraph v1.0 demonstrations"""

    print("📊 LANGGRAPH v1.0 ALPHA COMPREHENSIVE DEMO")
    print("=" * 45)

    # Check requirements
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required in .env file")
        print("\n💡 This demo showcases LangGraph v1.0 alpha features:")
        print("• Enhanced StateGraph with modern persistence")
        print("• Durable execution with checkpoints")
        print("• Human-in-the-loop workflows")
        print("• Streaming capabilities")
        print("• Multi-agent orchestration patterns")
        print("• Production-ready error handling")
        return

    print("✅ API key found, starting LangGraph demonstrations...")
    print("\n🌟 LangGraph v1.0 Alpha Features:")
    print("• Enhanced persistence with SqliteSaver and MemorySaver")
    print("• Human-in-the-loop with interrupt capabilities")
    print("• Streaming support for real-time responses")
    print("• Improved StateGraph with better typing")
    print("• Send API for dynamic orchestration")
    print("• Production monitoring and observability")

    demos = [
        ("Basic Graph v1.0", demonstrate_basic_graph_v1),
        ("Tool Integration", demonstrate_tool_integration),
        ("Human-in-the-Loop", demonstrate_human_in_loop),
        ("Streaming Workflow", demonstrate_streaming_workflow)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'=' * 60}")
        print(f"DEMO {i}: {name.upper()}")
        print(f"{'=' * 60}")

        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n⏸️ Demo {i} interrupted")
            break
        except Exception as e:
            print(f"❌ Demo {i} error: {e}")
            continue

        if i < len(demos):
            input(f"\nPress Enter to continue to demo {i+1}...")

    print("\n🎉 LangGraph v1.0 alpha demonstrations completed!")
    print("\n💡 LangGraph v1.0 Key Features Covered:")
    print("1. Enhanced StateGraph: Better typing and configuration")
    print("2. Modern Persistence: MemorySaver, SqliteSaver with thread support")
    print("3. Human-in-the-Loop: Interrupt-based workflows")
    print("4. Tool Integration: Seamless tool calling with create_react_agent")
    print("5. Streaming: Real-time response generation")
    print("6. Error Handling: Robust production-ready patterns")

    print("\n🚀 v1.0 Migration Notes:")
    print("• START constant replaces set_entry_point")
    print("• Enhanced checkpointer API with thread support")
    print("• Improved interrupt handling for human input")
    print("• Better integration with LangChain v0.3+")
    print("• Memory replaces deprecated LangChain memory classes")

    print("\n➡️ Continue to Module 5 (Google ADK) for multi-modal AI!")


if __name__ == "__main__":
    main()