"""
SmolAgents Comprehensive Exercise - Master HuggingFace SmolAgents

This exercise guides you through building sophisticated applications using
the real HuggingFace SmolAgents framework. You'll learn to create custom tools,
handle multi-modal inputs, and deploy production-ready agents.

Learning Objectives:
- Master the SmolAgents CodeAgent API
- Create custom tools using the @tool decorator
- Handle different model providers (HF, OpenAI, local)
- Implement secure code execution patterns
- Build production-ready agent applications

Prerequisites:
- pip install "smolagents[toolkit]"
- API keys for model providers (HF_TOKEN, OPENAI_API_KEY, etc.)
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for SmolAgents availability
try:
    from smolagents import CodeAgent, InferenceClientModel, tool
    SMOLAGENTS_AVAILABLE = True
except ImportError:
    SMOLAGENTS_AVAILABLE = False
    print("⚠️  SmolAgents not installed. Install with: pip install \"smolagents[toolkit]\"")


class SmolAgentsExercise:
    """Complete SmolAgents exercise with progressive difficulty"""

    def __init__(self):
        self.completed_tasks = []
        self.agents = {}

    def task_1_basic_agent(self):
        """
        Task 1: Create a basic SmolAgents CodeAgent

        TODO:
        1. Import required components from smolagents
        2. Create an InferenceClientModel instance
        3. Create a CodeAgent with the model
        4. Test with a simple mathematical task
        """
        print("📚 Task 1: Basic SmolAgents Setup")
        print("=" * 40)

        if not SMOLAGENTS_AVAILABLE:
            print("❌ SmolAgents not available. Showing expected structure:")
            self._show_basic_example()
            return False

        try:
            # TODO: Implement basic agent creation
            # Your code here:
            # model = ...
            # agent = ...

            print("✅ Complete this task by implementing the basic agent setup")
            return True

        except Exception as e:
            print(f"❌ Error in Task 1: {e}")
            return False

    def task_2_custom_tools(self):
        """
        Task 2: Create custom tools for your agent

        TODO:
        1. Create a custom weather tool using @tool decorator
        2. Create a custom calculator tool
        3. Add both tools to a CodeAgent
        4. Test the agent with tasks that use these tools
        """
        print("\n📚 Task 2: Custom Tools")
        print("=" * 40)

        if not SMOLAGENTS_AVAILABLE:
            self._show_custom_tools_example()
            return False

        try:
            # TODO: Implement custom tools
            # Example structure:
            # @tool
            # def your_custom_tool(param: type) -> type:
            #     """Tool description"""
            #     # Tool implementation
            #     return result

            print("✅ Complete this task by creating custom tools")
            return True

        except Exception as e:
            print(f"❌ Error in Task 2: {e}")
            return False

    def task_3_model_providers(self):
        """
        Task 3: Work with different model providers

        TODO:
        1. Set up InferenceClientModel (HuggingFace)
        2. Set up LiteLLMModel (OpenAI/Anthropic)
        3. Try local model integration
        4. Compare performance and capabilities
        """
        print("\n📚 Task 3: Model Providers")
        print("=" * 40)

        if not SMOLAGENTS_AVAILABLE:
            self._show_model_providers_example()
            return False

        try:
            # TODO: Implement different model providers
            print("✅ Complete this task by setting up different model providers")
            return True

        except Exception as e:
            print(f"❌ Error in Task 3: {e}")
            return False

    def task_4_secure_execution(self):
        """
        Task 4: Implement secure code execution

        TODO:
        1. Set up E2B executor for secure sandboxing
        2. Configure Docker-based execution
        3. Test with potentially unsafe code
        4. Compare security approaches
        """
        print("\n📚 Task 4: Secure Execution")
        print("=" * 40)

        if not SMOLAGENTS_AVAILABLE:
            self._show_security_example()
            return False

        try:
            # TODO: Implement secure execution patterns
            print("✅ Complete this task by setting up secure execution")
            return True

        except Exception as e:
            print(f"❌ Error in Task 4: {e}")
            return False

    def task_5_production_app(self):
        """
        Task 5: Build a production application

        TODO:
        1. Create a FastAPI application with SmolAgents
        2. Add rate limiting and authentication
        3. Implement error handling and logging
        4. Add monitoring and metrics
        """
        print("\n📚 Task 5: Production Application")
        print("=" * 40)

        if not SMOLAGENTS_AVAILABLE:
            self._show_production_example()
            return False

        try:
            # TODO: Implement production application
            print("✅ Complete this task by building a production app")
            return True

        except Exception as e:
            print(f"❌ Error in Task 5: {e}")
            return False

    def _show_basic_example(self):
        """Show basic SmolAgents example structure"""
        print("💡 Expected implementation:")
        print("""
from smolagents import CodeAgent, InferenceClientModel

# Create model
model = InferenceClientModel()

# Create agent
agent = CodeAgent(tools=[], model=model)

# Test agent
result = agent.run("Calculate the sum of numbers from 1 to 100")
print(result)
""")

    def _show_custom_tools_example(self):
        """Show custom tools example"""
        print("💡 Expected custom tools structure:")
        print("""
from smolagents import CodeAgent, InferenceClientModel, tool
import requests

@tool
def get_weather(city: str) -> str:
    '''Get weather information for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information as string
    '''
    # Implement weather API call
    return f"Weather in {city}: Sunny, 25°C"

@tool
def advanced_calculator(expression: str) -> float:
    '''Evaluate mathematical expressions safely.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        Result of the calculation
    '''
    # Implement safe evaluation
    try:
        return eval(expression)  # In production, use safer evaluation
    except:
        return 0.0

# Create agent with tools
model = InferenceClientModel()
agent = CodeAgent(tools=[get_weather, advanced_calculator], model=model)

# Test with custom tools
result = agent.run("What's the weather like in Paris and calculate 25 * 4")
""")

    def _show_model_providers_example(self):
        """Show model providers example"""
        print("💡 Model provider options:")
        print("""
# Option 1: HuggingFace Inference API
from smolagents import InferenceClientModel
model1 = InferenceClientModel()

# Option 2: OpenAI via LiteLLM
from smolagents import LiteLLMModel
model2 = LiteLLMModel(model_id="gpt-4")

# Option 3: Anthropic via LiteLLM
model3 = LiteLLMModel(model_id="claude-3-sonnet-20240229")

# Option 4: Local model
from smolagents import InferenceClientModel
model4 = InferenceClientModel(model_id="microsoft/DialoGPT-medium")

# Create agents with different models
agent1 = CodeAgent(tools=[], model=model1)
agent2 = CodeAgent(tools=[], model=model2)
""")

    def _show_security_example(self):
        """Show security implementation example"""
        print("💡 Secure execution patterns:")
        print("""
# E2B Sandbox (recommended for production)
from smolagents import CodeAgent, InferenceClientModel, E2BExecutor

executor = E2BExecutor()
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, executor=executor)

# Docker-based execution
from smolagents import DockerExecutor
docker_executor = DockerExecutor(image="python:3.9-slim")
agent_docker = CodeAgent(tools=[], model=model, executor=docker_executor)

# Test with potentially unsafe code
unsafe_task = "Create a file and write some data to it"
result = agent.run(unsafe_task)  # Executes safely in sandbox
""")

    def _show_production_example(self):
        """Show production application example"""
        print("💡 Production application structure:")
        print("""
# app.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from smolagents import CodeAgent, InferenceClientModel, E2BExecutor
import logging
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolAgents API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
executor = E2BExecutor()
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, executor=executor)

class TaskRequest(BaseModel):
    task: str
    max_iterations: int = 5

@app.post("/solve")
async def solve_task(request: TaskRequest):
    try:
        logger.info(f"Processing task: {request.task}")
        result = agent.run(request.task)
        logger.info("Task completed successfully")
        return {"result": result, "status": "success"}
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
""")

    def run_all_tasks(self):
        """Run all exercise tasks"""
        print("🐭 SmolAgents Comprehensive Exercise")
        print("=" * 50)

        tasks = [
            self.task_1_basic_agent,
            self.task_2_custom_tools,
            self.task_3_model_providers,
            self.task_4_secure_execution,
            self.task_5_production_app,
        ]

        for i, task in enumerate(tasks, 1):
            try:
                success = task()
                if success:
                    self.completed_tasks.append(f"Task {i}")
                print()
            except Exception as e:
                print(f"❌ Task {i} failed: {e}")

        self._show_summary()

    def _show_summary(self):
        """Show exercise completion summary"""
        print("\n📊 Exercise Summary")
        print("=" * 30)
        print(f"Completed tasks: {len(self.completed_tasks)}/5")

        if self.completed_tasks:
            print("✅ Completed:")
            for task in self.completed_tasks:
                print(f"   - {task}")

        print("\n🎯 Next Steps:")
        if not SMOLAGENTS_AVAILABLE:
            print("1. Install SmolAgents: pip install \"smolagents[toolkit]\"")
            print("2. Set up API keys (HF_TOKEN, OPENAI_API_KEY, etc.)")
            print("3. Complete the implementation tasks")
        else:
            print("1. Complete any remaining TODO items")
            print("2. Experiment with more complex use cases")
            print("3. Deploy your production application")
            print("4. Explore advanced features like custom executors")

        print("\n📚 Resources:")
        print("- GitHub: https://github.com/huggingface/smolagents")
        print("- Docs: https://huggingface.co/docs/smolagents")
        print("- E2B: https://e2b.dev/ (for secure execution)")
        print("- Examples: https://huggingface.co/spaces (search for smolagents)")


def main():
    """Main exercise function"""
    exercise = SmolAgentsExercise()
    exercise.run_all_tasks()


if __name__ == "__main__":
    main()