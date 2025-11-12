"""
Core Agent implementations for SmolAgents project.
"""

from smolagents import CodeAgent, ToolCallingAgent
from smolagents import InferenceClientModel, LiteLLMModel, OpenAIServerModel
from typing import List, Optional, Any

# # HuggingFace Inference API (default)
# model = InferenceClientModel()
# # OpenAI via LiteLLM
# # Note: Requires OPENAI_API_KEY environment variable
# model_openai = LiteLLMModel(model_id="gpt-3.5-turbo")
# # Anthropic via LiteLLM
# # Note: Requires ANTHROPIC_API_KEY environment variable
# model_anthropic = LiteLLMModel(model_id="claude-3-haiku-20240307")


class AgentManager:
    """Manager class for different types of agents."""
    
    def __init__(self):
        self.agents = {}
    
    def create_code_agent(self, tools: Optional[List[Any]] = None, model=None, **kwargs):
        """Create a CodeAgent instance."""
        if tools is None:
            tools = []
        
        if model is None:
            model = InferenceClientModel()
            
        agent = CodeAgent(tools=tools, model=model, **kwargs)
        self.agents['code_agent'] = agent
        return agent
    
    def create_tool_calling_agent(self, tools: Optional[List[Any]] = None, model=None, **kwargs):
        """Create a ToolCallingAgent instance."""
        if tools is None:
            tools = []
            
        if model is None:
            model = InferenceClientModel()
            
        agent = ToolCallingAgent(tools=tools, model=model, **kwargs)
        self.agents['tool_calling_agent'] = agent
        return agent
    
    def get_agent(self, agent_type: str):
        """Get an agent by type."""
        return self.agents.get(agent_type)
    
    def run_agent(self, agent_type: str, task: str, **kwargs):
        """Run an agent with a specific task."""
        agent = self.get_agent(agent_type)
        if agent is None:
            raise ValueError(f"Agent '{agent_type}' not found.")
        return agent.run(task, **kwargs)

# Default agent manager instance
agent_manager = AgentManager()