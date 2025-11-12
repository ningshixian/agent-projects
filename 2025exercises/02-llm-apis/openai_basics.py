"""
OpenAI API Basics
Comprehensive examples of using OpenAI's API for agentic applications
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Message:
    role: str
    content: str


class OpenAIAgent:
    """Basic agent using OpenAI API with function calling capabilities"""

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.conversation_history: List[Dict] = []
        self.available_tools = self._setup_tools()

    def _setup_tools(self) -> List[Dict]:
        """Define available tools for the agent"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search for information in a knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "domain": {
                                "type": "string",
                                "description": "Domain to search in (e.g., 'science', 'technology', 'history')",
                                "enum": ["science", "technology", "history", "general"]
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_information",
                    "description": "Save important information for later reference",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title for the saved information"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to save"
                            },
                            "category": {
                                "type": "string",
                                "description": "Category for organization"
                            }
                        },
                        "required": ["title", "content"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool function"""
        try:
            if tool_name == "calculate":
                expression = arguments.get("expression", "")
                # Simple eval for demo - in production, use a safe math evaluator
                result = eval(expression)
                return f"Calculation result: {expression} = {result}"

            elif tool_name == "search_knowledge":
                query = arguments.get("query", "")
                domain = arguments.get("domain", "general")
                # Simulate knowledge search
                return f"Knowledge search results for '{query}' in {domain} domain: [Sample results would appear here]"

            elif tool_name == "save_information":
                title = arguments.get("title", "")
                content = arguments.get("content", "")
                category = arguments.get("category", "general")
                # Simulate saving information
                return f"Information saved successfully: '{title}' in category '{category}'"

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Have a conversation with the agent"""

        # Add system message if provided
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            # Make API call with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )

            assistant_message = response.choices[0].message

            # Handle tool calls
            if assistant_message.tool_calls:
                # Add assistant message to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls]
                })

                # Execute tools and collect results
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    result = self.execute_tool(function_name, function_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": result
                    })

                # Add tool results to conversation
                self.conversation_history.extend(tool_results)

                # Get final response from assistant
                final_messages = messages + [
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [tc.model_dump() for tc in assistant_message.tool_calls]
                    }
                ] + tool_results

                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=final_messages,
                    temperature=0.7,
                    max_tokens=1000
                )

                final_content = final_response.choices[0].message.content

            else:
                final_content = assistant_message.content

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": final_content})

            return final_content

        except Exception as e:
            return f"Error in conversation: {str(e)}"

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history"

        summary_messages = [
            {
                "role": "system",
                "content": "Summarize the following conversation in 2-3 sentences:"
            },
            {
                "role": "user",
                "content": "\\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_history])
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_messages,
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"


def demonstrate_openai_basics():
    """Demonstrate OpenAI API capabilities"""

    print("=== OpenAI API Demonstration ===\\n")

    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        return

    # Create agent
    agent = OpenAIAgent()

    # System prompt for the agent
    system_prompt = """You are a helpful AI assistant with access to tools for calculation,
    knowledge search, and information storage. Use these tools when appropriate to help
    users with their requests."""

    # Test conversations
    test_conversations = [
        "Hello! Can you calculate 15 * 24 + 100?",
        "Search for information about artificial intelligence in the technology domain",
        "Save the calculation result from earlier as 'Daily Revenue Calculation'",
        "What's the difference between machine learning and deep learning?"
    ]

    print("Starting conversation with OpenAI agent...\\n")

    for i, message in enumerate(test_conversations, 1):
        print(f"👤 User: {message}")
        response = agent.chat(message, system_prompt if i == 1 else None)
        print(f"🤖 Assistant: {response}\\n")
        print("-" * 60 + "\\n")

    # Show conversation summary
    print("📋 Conversation Summary:")
    summary = agent.get_conversation_summary()
    print(summary)


def compare_models():
    """Compare different OpenAI models"""
    print("\\n=== Model Comparison ===\\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found.")
        return

    models = ["gpt-3.5-turbo", "gpt-4-turbo-preview"]
    prompt = "Explain the concept of recursion in programming in exactly 50 words."

    for model in models:
        print(f"Testing {model}:")
        try:
            agent = OpenAIAgent(model=model)
            response = agent.chat(prompt)
            word_count = len(response.split())
            print(f"Response ({word_count} words): {response}\\n")
        except Exception as e:
            print(f"Error with {model}: {e}\\n")


if __name__ == "__main__":
    demonstrate_openai_basics()
    compare_models()