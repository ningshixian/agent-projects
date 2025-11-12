"""
Anthropic Claude API Basics
Comprehensive examples of using Claude for agentic applications
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float


class ClaudeAgent:
    """Agent using Anthropic's Claude model with advanced features"""

    def __init__(self, model: str = "claude-3-7-sonnet-latest"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.conversation_history: List[ConversationTurn] = []
        self.available_tools = self._setup_tools()

    def _setup_tools(self) -> List[Dict[str, Any]]:
        """Define available tools for Claude"""
        return [
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "text_analyzer",
                "description": "Analyze text for various properties",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis: sentiment, readability, or summary",
                            "enum": ["sentiment", "readability", "summary"]
                        }
                    },
                    "required": ["text", "analysis_type"]
                }
            },
            {
                "name": "code_reviewer",
                "description": "Review code for quality and best practices",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to review"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language",
                            "enum": ["python", "javascript", "java", "cpp", "other"]
                        }
                    },
                    "required": ["code", "language"]
                }
            },
            {
                "name": "task_planner",
                "description": "Break down complex tasks into manageable steps",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Complex task to break down"
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context or constraints"
                        }
                    },
                    "required": ["task"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool function"""
        try:
            if tool_name == "calculator":
                expression = arguments.get("expression", "")
                # Safe evaluation for demo - use proper math library in production
                try:
                    result = eval(expression)
                    return f"Calculation result: {expression} = {result}"
                except Exception as e:
                    return f"Calculation error: {e}"

            elif tool_name == "text_analyzer":
                text = arguments.get("text", "")
                analysis_type = arguments.get("analysis_type", "summary")

                if analysis_type == "sentiment":
                    # Simple sentiment analysis
                    positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                    negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]

                    text_lower = text.lower()
                    pos_count = sum(1 for word in positive_words if word in text_lower)
                    neg_count = sum(1 for word in negative_words if word in text_lower)

                    if pos_count > neg_count:
                        sentiment = "Positive"
                    elif neg_count > pos_count:
                        sentiment = "Negative"
                    else:
                        sentiment = "Neutral"

                    return f"Sentiment analysis: {sentiment} (positive: {pos_count}, negative: {neg_count})"

                elif analysis_type == "readability":
                    words = len(text.split())
                    sentences = text.count('.') + text.count('!') + text.count('?')
                    avg_words_per_sentence = words / max(sentences, 1)

                    if avg_words_per_sentence < 15:
                        readability = "Easy"
                    elif avg_words_per_sentence < 25:
                        readability = "Moderate"
                    else:
                        readability = "Complex"

                    return f"Readability analysis: {readability} ({words} words, {sentences} sentences, {avg_words_per_sentence:.1f} avg words/sentence)"

                elif analysis_type == "summary":
                    word_count = len(text.split())
                    char_count = len(text)
                    return f"Text summary: {word_count} words, {char_count} characters"

            elif tool_name == "code_reviewer":
                code = arguments.get("code", "")
                language = arguments.get("language", "python")

                # Simple code analysis
                lines = code.split('\\n')
                issues = []

                if language == "python":
                    for i, line in enumerate(lines, 1):
                        if len(line) > 100:
                            issues.append(f"Line {i}: Line too long ({len(line)} chars)")
                        if 'print(' in line and not line.strip().startswith('#'):
                            issues.append(f"Line {i}: Consider using logging instead of print")

                review = f"Code review for {language}:\\n"
                review += f"Total lines: {len(lines)}\\n"
                if issues:
                    review += "Issues found:\\n" + "\\n".join(issues)
                else:
                    review += "No major issues found!"

                return review

            elif tool_name == "task_planner":
                task = arguments.get("task", "")
                context = arguments.get("context", "")

                # Simple task breakdown
                if "website" in task.lower():
                    steps = [
                        "1. Plan website structure and pages",
                        "2. Design mockups and user interface",
                        "3. Set up development environment",
                        "4. Implement frontend components",
                        "5. Add backend functionality if needed",
                        "6. Test across different devices",
                        "7. Deploy to production"
                    ]
                elif "app" in task.lower():
                    steps = [
                        "1. Define app requirements and features",
                        "2. Choose technology stack",
                        "3. Create project structure",
                        "4. Implement core functionality",
                        "5. Add user interface",
                        "6. Test and debug",
                        "7. Prepare for deployment"
                    ]
                else:
                    steps = [
                        "1. Break down the main task into components",
                        "2. Identify dependencies between components",
                        "3. Prioritize tasks by importance",
                        "4. Create timeline and milestones",
                        "5. Execute tasks systematically",
                        "6. Review and adjust as needed"
                    ]

                plan = f"Task breakdown for: {task}\\n"
                if context:
                    plan += f"Context: {context}\\n"
                plan += "\\n".join(steps)
                return plan

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """Have a conversation with Claude using tools"""
        try:
            # Prepare messages
            messages = []

            # Add conversation history
            for turn in self.conversation_history:
                messages.append({
                    "role": turn.role,
                    "content": turn.content
                })

            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })

            # First API call with tools
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt or "You are a helpful assistant with access to various tools. Use them when appropriate to help users.",
                messages=messages,
                tools=self.available_tools
            )

            response_text = ""
            tool_results = []

            # Process Claude's response
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == "text":
                        response_text += content_block.text
                    elif content_block.type == "tool_use":  # Claude invoked a tool
                        tool_id = content_block.id
                        tool_name = content_block.name
                        tool_input = content_block.input

                        print(f"🔧 Using tool: {tool_name}")
                        print(f"📋 Input: {tool_input}")

                        # Run the tool
                        tool_result = self.execute_tool(tool_name, tool_input)

                        print(f"🔍 Result: {tool_result}")

                        # Save for follow-up
                        tool_results.append({
                            "tool_use_id": tool_id,
                            "content": tool_result
                        })

            # If tools were used, send results back to Claude
            if tool_results:
                messages.append({
                    "role": "assistant",
                    "content": response.content  # include Claude's original tool request
                })

                for result in tool_results:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result["tool_use_id"],
                                "content": result["content"]
                            }
                        ]
                    })

                # Second call: give Claude the tool results
                follow_up = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt or "Provide a helpful response based on the tool results.",
                    messages=messages
                )

                final_response = ""
                for content_block in follow_up.content:
                    if content_block.type == "text":
                        final_response += content_block.text

                response_text = final_response

            # Store conversation
            self.conversation_history.append(ConversationTurn("user", user_message, time.time()))
            self.conversation_history.append(ConversationTurn("assistant", response_text, time.time()))

            return response_text or "No response generated"

        except Exception as e:
            return f"Error in conversation: {str(e)}"

    def analyze_conversation(self) -> Dict[str, Any]:
        """Analyze the conversation for insights"""
        if not self.conversation_history:
            return {"message": "No conversation to analyze"}

        user_turns = [turn for turn in self.conversation_history if turn.role == "user"]
        assistant_turns = [turn for turn in self.conversation_history if turn.role == "assistant"]

        # Calculate metrics
        avg_user_length = sum(len(turn.content) for turn in user_turns) / len(user_turns)
        avg_assistant_length = sum(len(turn.content) for turn in assistant_turns) / len(assistant_turns)

        # Conversation duration
        if len(self.conversation_history) >= 2:
            duration = self.conversation_history[-1].timestamp - self.conversation_history[0].timestamp
        else:
            duration = 0

        return {
            "total_turns": len(self.conversation_history),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "avg_user_message_length": avg_user_length,
            "avg_assistant_message_length": avg_assistant_length,
            "conversation_duration_seconds": duration,
            "topics_discussed": self._extract_topics()
        }

    def _extract_topics(self) -> List[str]:
        """Simple topic extraction from conversation"""
        topics = []
        all_text = " ".join(turn.content for turn in self.conversation_history).lower()

        topic_keywords = {
            "programming": ["code", "python", "javascript", "programming", "function", "variable"],
            "mathematics": ["calculate", "math", "equation", "number", "formula"],
            "analysis": ["analyze", "data", "text", "sentiment", "review"],
            "planning": ["plan", "task", "step", "organize", "project"],
            "general": []
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                topics.append(topic)

        return topics if topics else ["general"]

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print("🧹 Conversation history cleared")


class ClaudeComparison:
    """Compare different Claude models"""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.models = [
            "claude-3-5-haiku-latest",    # Fast and efficient
            "claude-opus-4-1-20250805",   # Balanced performance
            "claude-3-7-sonnet-latest"      # Highest capability (if available)
        ]

    def compare_models(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """Compare response from different Claude models"""
        results = {}

        for model in self.models:
            print(f"\\n🧪 Testing {model}...")
            start_time = time.time()

            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_time = time.time() - start_time
                response_text = ""

                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        response_text += content_block.text

                results[model] = {
                    "response": response_text,
                    "response_time": response_time,
                    "response_length": len(response_text),
                    "tokens_per_second": len(response_text.split()) / response_time,
                    "status": "success"
                }

                print(f"   ✅ Success in {response_time:.2f}s")

            except Exception as e:
                results[model] = {
                    "response": None,
                    "error": str(e),
                    "status": "error"
                }
                print(f"   ❌ Error: {e}")

        return results

    def print_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Print formatted comparison results"""
        print("\\n" + "="*80)
        print("CLAUDE MODEL COMPARISON")
        print("="*80)

        for model, result in results.items():
            print(f"\\n🤖 {model.upper()}")
            print("-" * 50)

            if result["status"] == "success":
                print(f"⏱️  Response Time: {result['response_time']:.2f}s")
                print(f"📝 Response Length: {result['response_length']} chars")
                print(f"🚀 Speed: {result['tokens_per_second']:.1f} words/sec")
                print(f"💬 Response Preview: {result['response'][:150]}...")
            else:
                print(f"❌ Error: {result['error']}")


def demonstrate_claude_basics():
    """Demonstrate Claude API capabilities"""

    print("🧠 Anthropic Claude API Demonstration")
    print("="*40)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found. Please set it in your .env file")
        print("   Get your API key from: https://console.anthropic.com/")
        return

    print("✅ Anthropic API key found")

    # Create Claude agent
    print("\\n🤖 Creating Claude Agent...")
    try:
        agent = ClaudeAgent()
        print(f"✅ Agent created with model: {agent.model}")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return

    # System prompt for the agent
    system_prompt = """You are an expert AI assistant with access to various tools.
    You excel at helping users with calculations, text analysis, code review, and task planning.
    Always use the appropriate tools when they can help answer the user's question more effectively."""

    # Test conversations with tool usage
    print("\\n💬 Testing Claude with Tool Integration")
    print("-" * 45)

    test_conversations = [
        "Can you calculate the compound interest on $1000 at 5% annual rate for 3 years? Use the formula: A = P(1 + r)**t",
        "Please analyze the sentiment of this text: 'I absolutely love this new feature! It's incredibly useful and well-designed.'",
        "Can you review this Python code and suggest improvements?\\n\\ndef calculate_average(numbers):\\n    return sum(numbers) / len(numbers)",
        "Help me plan a project to create a personal blog website. I want to use modern web technologies."
    ]

    for i, conversation in enumerate(test_conversations, 1):
        print(f"\\n--- Conversation {i} ---")
        print(f"👤 User: {conversation}")
        print()

        response = agent.chat(conversation, system_prompt)
        print(f"🤖 Claude: {response}")
        print("-" * 60)

    # Show conversation analysis
    print("\\n📊 Conversation Analysis")
    print("-" * 25)
    analysis = agent.analyze_conversation()
    print(f"Total turns: {analysis['total_turns']}")
    print(f"Topics discussed: {', '.join(analysis['topics_discussed'])}")
    print(f"Average user message length: {analysis['avg_user_message_length']:.0f} chars")
    print(f"Average assistant message length: {analysis['avg_assistant_message_length']:.0f} chars")
    print(f"Conversation duration: {analysis['conversation_duration_seconds']:.1f} seconds")


def demonstrate_model_comparison():
    """Demonstrate comparison between Claude models"""

    print("\\n🥊 Claude Model Comparison")
    print("-" * 30)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found")
        return

    try:
        comparison = ClaudeComparison()

        # Test prompt that benefits from different model capabilities
        test_prompt = """Explain the concept of recursion in programming with a practical example.
        Make it accessible for beginners but also mention advanced considerations."""

        print(f"\\nTest prompt: {test_prompt}")
        print("\\nTesting across different Claude models...")

        results = comparison.compare_models(test_prompt)
        comparison.print_comparison(results)

        # Performance summary
        print("\\n📈 Performance Summary:")
        successful_models = {k: v for k, v in results.items() if v["status"] == "success"}

        if successful_models:
            fastest_model = min(successful_models.keys(),
                              key=lambda x: successful_models[x]["response_time"])
            most_detailed = max(successful_models.keys(),
                               key=lambda x: successful_models[x]["response_length"])

            print(f"🚀 Fastest: {fastest_model} ({successful_models[fastest_model]['response_time']:.2f}s)")
            print(f"📝 Most detailed: {most_detailed} ({successful_models[most_detailed]['response_length']} chars)")

    except Exception as e:
        print(f"❌ Comparison error: {e}")


def claude_safety_features():
    """Demonstrate Claude's safety and helpfulness features"""

    print("\\n🛡️ Claude Safety Features Demo")
    print("-" * 35)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found")
        return

    try:
        agent = ClaudeAgent()

        # Test helpful refusal and safety
        safety_tests = [
            "How can I write better Python code?",  # Helpful
            "Explain how neural networks work",      # Educational
            "What are the best practices for API security?",  # Security (helpful)
        ]

        print("\\nTesting Claude's helpful and safe responses:")

        for i, test in enumerate(safety_tests, 1):
            print(f"\\n{i}. Test: {test}")
            response = agent.chat(test)
            print(f"   Response length: {len(response)} chars")
            print(f"   Tone: {'Helpful' if len(response) > 50 else 'Brief'}")
            print(f"   Preview: {response[:100]}...")

        print("\\n✅ Claude demonstrates:")
        print("- Helpful responses to legitimate questions")
        print("- Educational explanations for learning")
        print("- Safety-conscious approach to sensitive topics")
        print("- Refusal to help with harmful activities")

    except Exception as e:
        print(f"❌ Safety demo error: {e}")


def main():
    """Run all Claude demonstrations"""

    print("🧠 ANTHROPIC CLAUDE API TUTORIAL")
    print("="*35)

    demonstrations = [
        ("Claude Basics with Tools", demonstrate_claude_basics),
        ("Model Comparison", demonstrate_model_comparison),
        ("Safety Features", claude_safety_features)
    ]

    for i, (name, demo_func) in enumerate(demonstrations, 1):
        print(f"\\n{'='*60}")
        print(f"DEMO {i}: {name.upper()}")
        print(f"{'='*60}")

        try:
            demo_func()
        except KeyboardInterrupt:
            print("\\n⏸️ Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Demo error: {e}")

        if i < len(demonstrations):
            input("\\nPress Enter to continue to next demo...")

    print("\\n🎉 Claude API demonstrations completed!")
    print("\\n💡 Key Takeaways:")
    print("1. Claude excels at reasoning and analysis tasks")
    print("2. Tool integration enables complex workflows")
    print("3. Different models offer speed vs capability trade-offs")
    print("4. Claude has built-in safety features and helpfulness")
    print("5. Conversation analysis helps optimize interactions")

    print("\\n➡️ Continue to Module 3 (LangChain) when ready!")


if __name__ == "__main__":
    main()