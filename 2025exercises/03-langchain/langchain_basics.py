"""
LangChain Basics - Updated for v0.3+
Comprehensive introduction to building agents with modern LangChain patterns
"""

import os
from typing import List, Dict, Any, Optional, Sequence
from dotenv import load_dotenv

# Updated LangChain v0.3+ imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import StdOutCallbackHandler

# Modern tool patterns
from langchain_core.tools import tool, Tool
from langchain.agents import create_react_agent, create_structured_chat_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain import hub

# Load environment variables
load_dotenv()


class NumberOutputParser(BaseOutputParser[float]):
    """Parse the output of an LLM call to a number using v0.3+ patterns."""

    def parse(self, text: str) -> float:
        """Parse the output of an LLM call to extract a number."""
        try:
            import re
            numbers = re.findall(r'[-+]?\d*\.?\d+', text)
            if numbers:
                return float(numbers[0])
            else:
                return 0.0
        except:
            return 0.0


def demonstrate_lcel_chains():
    """Demonstrate LangChain Expression Language (LCEL) - the modern way"""

    print("🔗 LangChain Expression Language (LCEL) Demo")
    print("=" * 45)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    print("\n1. Simple LCEL Chain")
    print("-" * 20)

    # Create a simple chain using LCEL
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions concisely."),
        ("human", "{question}")
    ])

    # Chain using the pipe operator (LCEL)
    chain = prompt | llm | StrOutputParser()

    questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the benefits of using AI agents?"
    ]

    for question in questions:
        result = chain.invoke({"question": question})
        print(f"Q: {question}")
        print(f"A: {result}\n")

    print("\n2. Complex LCEL Chain with Multiple Steps")
    print("-" * 42)

    # Step 1: Generate topic
    topic_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a specific, interesting topic about the given subject."),
        ("human", "Subject: {subject}")
    ])

    # Step 2: Create outline
    outline_prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a detailed outline for an article about the given topic."),
        ("human", "Topic: {topic}")
    ])

    # Step 3: Write introduction
    intro_prompt = ChatPromptTemplate.from_messages([
        ("system", "Write an engaging introduction based on the topic and outline."),
        ("human", "Topic: {topic}\nOutline: {outline}")
    ])

    # Complex chain using LCEL
    complex_chain = (
        RunnablePassthrough()
        | {"topic": topic_prompt | llm | StrOutputParser()}
        | {"topic": RunnableLambda(lambda x: x["topic"]),
           "outline": outline_prompt | llm | StrOutputParser()}
        | {"topic": RunnableLambda(lambda x: x["topic"]),
           "outline": lambda x: x["outline"],
           "introduction": intro_prompt | llm | StrOutputParser()}
    )

    # Run the complex chain
    result = complex_chain.invoke("artificial intelligence")

    print("Generated Topic:", result["topic"])
    print("\nOutline:", result["outline"])
    print("\nIntroduction:", result["introduction"])


def demonstrate_modern_agents():
    """Demonstrate modern LangChain agents using create_react_agent"""

    print("\n🤖 Modern LangChain Agents Demo")
    print("=" * 35)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    from langchain_core.tools import tool, Tool
    # Define tools using the modern @tool decorator
    @tool
    def python_calculator(expression: str) -> str:
        """Calculate mathematical expressions safely."""
        try:
            # Simple safety check
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression.replace(' ', '')):
                return "Error: Invalid characters in expression"

            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def word_counter(text: str) -> str:
        """Count words in the given text."""
        words = text.split()
        return f"Word count: {len(words)} words"

    @tool
    def text_reverser(text: str) -> str:
        """Reverse the given text."""
        return f"Reversed text: {text[::-1]}"

    @tool
    def pig_latin_translator(text: str) -> str:
        """Translate text to Pig Latin."""
        words = text.split()
        pig_latin_words = []
        for word in words:
            if word:
                if word[0].lower() in 'aeiou':
                    pig_latin_words.append(word + 'way')
                else:
                    pig_latin_words.append(word[1:] + word[0] + 'ay')
        return f"Pig Latin: {' '.join(pig_latin_words)}"

    # Create tools list
    tools = [python_calculator, word_counter, text_reverser, pig_latin_translator]

    # Get a pre-built prompt from hub (modern approach)
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Fallback prompt if hub is not available
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the tools available to answer questions.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

    # Create agent using modern patterns
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )

    print("\nAgent initialized with tools:")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")

    # Test agent with various tasks
    test_tasks = [
        "Calculate the result of 25 * 4 + 17",
        "Count how many words are in this sentence: 'LangChain makes it easy to build AI applications with language models'",
        "Reverse the text 'Hello World'",
        "Translate 'Hello World' to Pig Latin"
    ]

    print("\n🧪 Testing Modern Agent:")
    for i, task in enumerate(test_tasks, 1):
        print(f"\n--- Task {i} ---")
        print(f"Human: {task}")

        try:
            response = agent_executor.invoke({"input": task})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 50)


def demonstrate_structured_output():
    """Demonstrate structured output with modern patterns"""

    print("\n📊 Structured Output Demo (Modern)")
    print("=" * 35)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    from langchain_core.pydantic_v1 import BaseModel, Field
    from typing import List

    # Define output structure
    class PersonAnalysis(BaseModel):
        name: str = Field(description="The person's name")
        age_estimate: int = Field(description="Estimated age")
        profession: str = Field(description="Likely profession")
        personality_traits: List[str] = Field(description="List of personality traits")
        summary: str = Field(description="Brief summary")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Use modern structured output method
    structured_llm = llm.with_structured_output(PersonAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the following person description and extract structured information."),
        ("human", "{description}")
    ])

    chain = prompt | structured_llm

    test_description = """
    Sarah is a 28-year-old software engineer who loves hiking and reading science fiction novels.
    She's known for being analytical, creative, and always eager to learn new technologies.
    She works at a tech startup and enjoys solving complex problems.
    """

    result = chain.invoke({"description": test_description})

    print("Structured Analysis Result:")
    print(f"Name: {result.name}")
    print(f"Age Estimate: {result.age_estimate}")
    print(f"Profession: {result.profession}")
    print(f"Personality Traits: {', '.join(result.personality_traits)}")
    print(f"Summary: {result.summary}")


def demonstrate_streaming():
    """Demonstrate streaming capabilities"""

    print("\n📡 Streaming Demo")
    print("=" * 17)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative storyteller."),
        ("human", "Write a short story about {topic}")
    ])

    chain = prompt | llm | StrOutputParser()

    print("🎭 Streaming story generation:")
    print("Topic: A robot learning to paint")
    print("\nStory:")
    print("-" * 20)

    try:
        for chunk in chain.stream({"topic": "a robot learning to paint"}):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Streaming error: {e}")


def demonstrate_research_agent():
    """Demonstrate a modern research agent"""

    print("\n🔍 Modern Research Agent Demo")
    print("=" * 32)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # Research tools using modern patterns
    @tool
    def web_search(query: str) -> str:
        """Search the web for information about a topic."""
        # Mock search results
        mock_results = {
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from data. Key applications include image recognition, natural language processing, and predictive analytics.",
            "climate change": "Climate change refers to global warming and its effects on weather patterns. Recent studies show increasing temperatures and extreme weather events worldwide.",
            "quantum computing": "Quantum computing uses quantum mechanics to process information. Companies like IBM, Google, and Microsoft are developing quantum computers.",
            "default": f"Search results for '{query}': Various articles and papers discuss this topic from multiple perspectives."
        }

        for key, result in mock_results.items():
            if key.lower() in query.lower():
                return result
        return mock_results["default"]

    @tool
    def academic_search(query: str) -> str:
        """Search academic databases for research papers."""
        return f"Academic papers on '{query}': Found 23 relevant papers published in the last 5 years. Key themes include methodology improvements and practical applications."

    @tool
    def fact_checker(claim: str) -> str:
        """Verify the accuracy of factual claims."""
        if "earth is flat" in claim.lower():
            return "FALSE: Scientific consensus confirms Earth is roughly spherical."
        elif "water boils at 100" in claim.lower():
            return "TRUE: Water boils at 100°C (212°F) at standard atmospheric pressure."
        else:
            return f"Fact check for '{claim}': This claim requires verification from reliable sources."

    tools = [web_search, academic_search, fact_checker]

    # Create modern research agent
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a thorough researcher. Use available tools to provide comprehensive, accurate information."),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    research_tasks = [
        "Research the current state of quantum computing",
        "Fact-check: Water boils at 100 degrees Celsius",
        "Find academic research on machine learning applications"
    ]

    print("\n📚 Research Agent Tasks:")
    for i, task in enumerate(research_tasks, 1):
        print(f"\n--- Research Task {i} ---")
        print(f"Request: {task}")

        try:
            result = agent_executor.invoke({"input": task})
            print(f"Research Result: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 60)


def main():
    """Run all modern LangChain demonstrations"""

    print("🦜 LANGCHAIN v0.3+ COMPREHENSIVE DEMO")
    print("=" * 40)

    # Check requirements
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required in .env file")
        print("\n💡 This demo showcases modern LangChain v0.3+ patterns:")
        print("• LCEL (LangChain Expression Language)")
        print("• Modern agent patterns (create_react_agent)")
        print("• Structured output with Pydantic")
        print("• Streaming capabilities")
        print("• Updated import patterns")
        return

    print("✅ API key found, starting modern LangChain demonstrations...")

    demos = [
        ("LCEL Chains", demonstrate_lcel_chains),
        ("Modern Agents", demonstrate_modern_agents),
        ("Structured Output", demonstrate_structured_output),
        ("Streaming", demonstrate_streaming),
        ("Research Agent", demonstrate_research_agent)
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

    print("\n🎉 Modern LangChain demonstrations completed!")
    print("\n💡 Modern LangChain v0.3+ Features Covered:")
    print("1. LCEL: Modern chain composition with | operator")
    print("2. Modern Agents: create_react_agent patterns")
    print("3. Structured Output: with_structured_output method")
    print("4. Streaming: Real-time response generation")
    print("5. Updated Imports: langchain_core, langchain_community")
    print("6. Tool Decorators: @tool for easy tool creation")

    print("\n🚀 Key Migration Notes:")
    print("• Deprecated: initialize_agent, LLMChain, ConversationBufferMemory")
    print("• Modern: create_react_agent, LCEL, LangGraph for memory")
    print("• Tools: Use @tool decorator instead of Tool class")
    print("• Prompts: Use ChatPromptTemplate with MessagesPlaceholder")

    print("\n➡️ Continue to Module 4 (LangGraph) for advanced workflows!")


if __name__ == "__main__":
    main()