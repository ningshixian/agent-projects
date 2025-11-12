"""
LangChain with Local Models
Demonstrates how to use local models with LangChain for cost-effective agents
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
# from langchain.schema import BaseMessage, HumanMessage, AIMessage  # Unused imports

# Local model imports
import requests
import json

# Load environment variables
load_dotenv()


class OllamaLLM(LLM):
    """Custom LangChain LLM wrapper for Ollama"""

    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Ollama API to generate text"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }

        if stop:
            payload["options"]["stop"] = stop

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error calling Ollama: {e}"


class LocalModelAgent:
    """LangChain agent using local models"""

    def __init__(self, model_name: str = "llama3.2:3b"):
        # Initialize local LLM
        self.llm = OllamaLLM(model_name=model_name)

        # Set up memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Define tools
        self.tools = self._create_tools()

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""

        def calculator(expression: str) -> str:
            """Simple calculator tool"""
            try:
                # Only allow safe mathematical operations
                allowed_chars = set('0123456789+-*/.()')
                if not all(c in allowed_chars or c.isspace() for c in expression):
                    return "Invalid expression: only basic math operations allowed"

                result = eval(expression)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {e}"

        def search_knowledge(query: str) -> str:
            """Mock knowledge search tool"""
            knowledge_base = {
                "python": "Python is a high-level programming language known for its simplicity and readability.",
                "ai": "Artificial Intelligence is the simulation of human intelligence in machines.",
                "langchain": "LangChain is a framework for developing applications powered by language models.",
                "local models": "Local models run on your own hardware, providing privacy and cost benefits."
            }

            query_lower = query.lower()
            for key, value in knowledge_base.items():
                if key in query_lower:
                    return f"Knowledge: {value}"

            return f"No specific knowledge found for '{query}'. This is a mock knowledge base."

        def text_analyzer(text: str) -> str:
            """Simple text analysis tool"""
            words = text.split()
            sentences = text.split('.')

            analysis = {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "character_count": len(text)
            }

            return f"Text Analysis: {json.dumps(analysis, indent=2)}"

        return [
            Tool(
                name="Calculator",
                func=calculator,
                description="Useful for mathematical calculations. Input should be a mathematical expression."
            ),
            Tool(
                name="Knowledge Search",
                func=search_knowledge,
                description="Search for information in the knowledge base. Input should be a search query."
            ),
            Tool(
                name="Text Analyzer",
                func=text_analyzer,
                description="Analyze text for word count, sentences, etc. Input should be the text to analyze."
            )
        ]

    def chat(self, message: str) -> str:
        """Chat with the local agent"""
        try:
            response = self.agent.run(message)
            return response
        except Exception as e:
            return f"Error: {e}"

    def reset_memory(self):
        """Clear conversation memory"""
        self.memory.clear()


class HybridAgent:
    """Agent that uses local models with cloud fallback"""

    def __init__(self,
                 local_model: str = "llama3.2:3b",
                 fallback_to_cloud: bool = True):
        self.local_model = local_model
        self.fallback_to_cloud = fallback_to_cloud

        # Initialize local agent
        self.local_agent = LocalModelAgent(local_model)

        # Initialize cloud agent (if available)
        self.cloud_agent = None
        if fallback_to_cloud and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain.llms import OpenAI
                from langchain.agents import initialize_agent, AgentType

                cloud_llm = OpenAI(temperature=0.7)
                self.cloud_agent = initialize_agent(
                    tools=self.local_agent.tools,
                    llm=cloud_llm,
                    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    ),
                    verbose=False
                )
            except Exception as e:
                print(f"Could not initialize cloud agent: {e}")

    def chat(self, message: str, prefer_local: bool = True) -> Dict[str, Any]:
        """Chat with hybrid agent (local first, cloud fallback)"""

        if prefer_local:
            # Try local first
            try:
                local_response = self.local_agent.chat(message)

                # Simple heuristic to check if local response is good
                if (not local_response.startswith("Error") and
                    len(local_response) > 10 and
                    not "timeout" in local_response.lower()):
                    return {
                        "response": local_response,
                        "source": "local",
                        "model": self.local_model
                    }
            except Exception as e:
                print(f"Local model failed: {e}")

        # Fallback to cloud if available
        if self.cloud_agent and self.fallback_to_cloud:
            try:
                cloud_response = self.cloud_agent.run(message)
                return {
                    "response": cloud_response,
                    "source": "cloud",
                    "model": "openai"
                }
            except Exception as e:
                print(f"Cloud model failed: {e}")

        # If everything fails
        return {
            "response": "I'm sorry, I'm having trouble processing your request right now.",
            "source": "error",
            "model": "none"
        }


def cost_comparison_demo():
    """Demonstrate cost comparison between local and cloud models"""

    print("💰 Cost Comparison: Local vs Cloud Models")
    print("=" * 50)

    # Simulate costs for different scenarios
    scenarios = [
        {"name": "Small App", "monthly_requests": 1000, "avg_tokens": 100},
        {"name": "Medium Business", "monthly_requests": 50000, "avg_tokens": 200},
        {"name": "Large Enterprise", "monthly_requests": 1000000, "avg_tokens": 300},
    ]

    # Approximate costs (as of 2024)
    openai_cost_per_token = 0.00002  # GPT-4 approximate
    local_setup_cost = 2000  # GPU setup cost
    local_monthly_electricity = 50  # Monthly electricity

    print(f"{'Scenario':<15} {'Requests/mo':<12} {'OpenAI Cost/mo':<15} {'Local Cost/mo':<15} {'Break-even':<12}")
    print("-" * 80)

    for scenario in scenarios:
        requests = scenario["monthly_requests"]
        tokens = scenario["avg_tokens"]

        openai_monthly = requests * tokens * openai_cost_per_token
        local_monthly = local_monthly_electricity

        break_even_months = local_setup_cost / max(openai_monthly - local_monthly, 1)

        print(f"{scenario['name']:<15} {requests:<12,} ${openai_monthly:<14.2f} ${local_monthly:<14.2f} {break_even_months:<11.1f}mo")


def demonstrate_local_langchain():
    """Demonstrate LangChain with local models"""

    print("🏠 LangChain + Local Models Demonstration")
    print("=" * 50)

    # Check if Ollama is available
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama not responding")
    except:
        print("❌ Ollama not available. Please install and run Ollama:")
        print("   1. Install from https://ollama.ai")
        print("   2. Run 'ollama serve' in terminal")
        print("   3. Pull a model: 'ollama pull llama3.2:3b'")
        return

    print("✅ Ollama is running")

    # Test local agent
    print("\\n🤖 Testing Local Agent")
    print("-" * 30)

    agent = LocalModelAgent("llama3.2:3b")

    test_queries = [
        "Calculate 15 * 24 + 100",
        "Search for information about Python",
        "Analyze this text: 'Local models are becoming increasingly popular for AI applications due to privacy and cost benefits.'"
    ]

    for query in test_queries:
        print(f"\\n👤 User: {query}")
        response = agent.chat(query)
        print(f"🤖 Local Agent: {response}")
        print("-" * 50)

    # Test hybrid agent
    print("\\n🔄 Testing Hybrid Agent (Local + Cloud Fallback)")
    print("-" * 50)

    hybrid_agent = HybridAgent()

    hybrid_query = "What are the key advantages of using local AI models in production?"
    print(f"\\n👤 User: {hybrid_query}")

    result = hybrid_agent.chat(hybrid_query)
    print(f"🤖 Hybrid Agent ({result['source']}): {result['response']}")
    print(f"   Model used: {result['model']}")

    # Cost comparison
    print("\\n")
    cost_comparison_demo()


if __name__ == "__main__":
    demonstrate_local_langchain()