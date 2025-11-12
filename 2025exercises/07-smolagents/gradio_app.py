"""
Gradio UI for SmolAgents
A web interface for interacting with SmolAgents
"""

import gradio as gr
import os
from core.agent import agent_manager
from core.tools import *
from smolagents import InferenceClientModel, OpenAIServerModel
# from smolagents import HfApiModel  # smolagents库旧版本中的类名
from dotenv import load_dotenv
load_dotenv()

from typing import Generator, Any

# Initialize the agent
def initialize_agent():
    """Initialize the agent for the Gradio app."""
    model = OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL_ID", ""),
        api_key=os.getenv("LLM_API_KEY", ""), 
        api_base=os.getenv("LLM_BASE_URL", "")
    )

    # Web Search Agent
    web_agent = agent_manager.create_tool_calling_agent(
        tools=[search_tool, visitwebpage_tool],
        model=model,
        name="SearchAgent",
        description="Runs web searches for you.",
        max_steps=2,  # Allow more steps for complex tasks
    )

    # knowledge retrieval agent
    retrieval_agent = agent_manager.create_tool_calling_agent(
        tools=[retriever_tool],
        model=model,
        name="RetrievalAgent",
        description="Retrieves relevant knowledge information from customer service KB.",
        max_steps=1,  # Allow more steps for complex tasks
    )

    # Manager Agent
    manager_agent = agent_manager.create_code_agent(
        tools=[
            final_answer_tool, python_tool, 
            user_input_tool, get_weather, calculator, get_time
        ],
        model=model,
        name="GradioAgent",
        description="Agent for Gradio interface",
        max_steps=3,  # Allow more steps for complex tasks
        # planning_interval=3  # 这是你激活规划的地方！
        verbosity_level=1,
        managed_agents=[web_agent, retrieval_agent],
        stream_outputs=True, 
        additional_authorized_imports=["time", "numpy", "pandas", "markdownify", "requests", "json"],
        # executor_type="e2b",    # Type of code executor. default "local"
    )
    
    return manager_agent


# Create the agent
agent = initialize_agent()

from smolagents import GradioUI, stream_to_gradio
gradio_ui = GradioUI(agent, file_upload_folder="uploads", reset_agent_memory=True)
gradio_ui.launch(
    server_name="0.0.0.0",
    server_port=7860,
    root_path="",
    inbrowser=True, 
    share=False
)

# def run_agent(task: str) -> Generator[str, None, None]:
#     """Run the agent with the given task and stream the output."""
#     try:
#         # Stream the agent's response
#         stream_result = agent.run(task, stream=True)
#         # 确保我们正确处理流式输出
#         for step in stream_result:  # type: ignore
#             yield f"{step}\n"
        
#         # 获取并返回最终结果
#         # 注意：如果agent.run(stream=True)已经返回了完整结果，则不需要再次调用agent.run(task)
#         # 这里假设stream模式不会返回最终结果，因此再调用一次非stream模式获取最终答案
#         final_result = agent.run(task, stream=False)
#         if hasattr(final_result, 'output'):
#             # 如果返回的是 RunResult 对象，获取其 output 属性
#             yield f"\nFinal Answer: {final_result.output}"
#         else:
#             # 如果返回的是直接的结果
#             yield f"\nFinal Answer: {final_result}"
#     except Exception as e:
#         yield f"Error: {str(e)}"

