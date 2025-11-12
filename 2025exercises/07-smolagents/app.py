"""
Main Application
A Gradio UI for interacting with SmolAgents
"""

import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.agent import agent_manager
from core.tools import *
from smolagents import E2BExecutor
from smolagents import InferenceClientModel, OpenAIServerModel
# from smolagents import HfApiModel  # smolagents库旧版本中的类名
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="SmolAgents API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    task: str
    max_iterations: int = 5


def create_main_agent():
    """Create the main agent for the application."""
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

    # Manager Agent
    manager_agent = agent_manager.create_code_agent(
        tools=[
            final_answer_tool, python_tool, 
            user_input_tool, get_weather, calculator, get_time
        ],
        model=model,
        name="MainAgent",
        description="A versatile agent with access to various tools",
        max_steps=3,  # Allow more steps for complex tasks
        # planning_interval=3 # 这是你激活规划的地方！
        verbosity_level=1,
        managed_agents=[web_agent],
        # stream_outputs=True, 
        additional_authorized_imports=["time", "numpy", "pandas"],
        executor_type="e2b",    # Type of code executor. default "local"
    )
    
    return manager_agent

print("Creating main agent...")
agent = create_main_agent()
# agent.push_to_hub('sergiopaniego/AlfredAgent')
print("Agent created successfully!")


def run_task(task: str):
    """Run a task with the main agent."""
    try:
        agent = agent_manager.get_agent("code_agent")
        if agent is None:
            agent = create_main_agent()
            
        result = agent.run(task)
        print(f"Result: {result}")
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@app.post("/solve")
async def solve_task(request: TaskRequest):
    try:
        print(f"Processing task: {request.task}")
        result = agent.run(request.task)
        print("Task completed successfully")
        return {"result": result, "status": "success"}
    except Exception as e:
        print(f"Task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # For now, just demonstrate the agent creation
    print("Creating main agent...")
    agent = create_main_agent()
    print("Agent created successfully!")
    
    # Example task
    # task = "What time is it and what is 123 * 456?"
    task = "Li Auto 公司的最新股价是多少？超过 20 吗？"
    print(f"\nRunning example task: {task}")
    run_task(task)
