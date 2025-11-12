"""
LangChain Advanced Agents
Complex agent patterns and real-world implementations
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentType, initialize_agent, Tool, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools.base import BaseTool
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


class TaskManagerTool(BaseTool):
    """Advanced tool for task management"""

    name: str = "task_manager"
    description: str = "Manage tasks: create, list, update status, or delete tasks. Input format: 'action:task_description' or 'action:task_id:new_status'"

    def __init__(self):
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 1
        super().__init__()

    def _run(self, query: str) -> str:
        """Execute task management operations"""
        try:
            parts = query.split(':', 2)
            action = parts[0].lower().strip()

            if action == "create" and len(parts) > 1:
                task_desc = parts[1].strip()
                task_id = self.next_id
                self.tasks[task_id] = {
                    "id": task_id,
                    "description": task_desc,
                    "status": "pending",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                self.next_id += 1
                return f"Task created with ID {task_id}: {task_desc}"

            elif action == "list":
                if not self.tasks:
                    return "No tasks found."

                result = "Current tasks:\n"
                for task_id, task in self.tasks.items():
                    result += f"  [{task_id}] {task['description']} - Status: {task['status']} (Created: {task['created']})\n"
                return result.strip()

            elif action == "update" and len(parts) > 2:
                task_id = int(parts[1].strip())
                new_status = parts[2].strip()

                if task_id in self.tasks:
                    self.tasks[task_id]["status"] = new_status
                    return f"Task {task_id} status updated to: {new_status}"
                else:
                    return f"Task {task_id} not found."

            elif action == "delete" and len(parts) > 1:
                task_id = int(parts[1].strip())
                if task_id in self.tasks:
                    desc = self.tasks[task_id]["description"]
                    del self.tasks[task_id]
                    return f"Deleted task {task_id}: {desc}"
                else:
                    return f"Task {task_id} not found."

            else:
                return "Invalid action. Use: create:description, list, update:id:status, or delete:id"

        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class DataAnalyzerTool(BaseTool):
    """Tool for analyzing data and generating insights"""

    name: str = "data_analyzer"
    description: str = "Analyze numerical data and provide statistics. Input: comma-separated numbers or 'demo' for sample data"

    def _run(self, data_input: str) -> str:
        """Analyze numerical data"""
        try:
            if data_input.lower().strip() == "demo":
                numbers = [23, 45, 67, 89, 34, 56, 78, 90, 12, 43]
                data_source = "demo dataset"
            else:
                # Parse comma-separated numbers
                numbers = [float(x.strip()) for x in data_input.split(',') if x.strip()]
                data_source = "provided data"

            if not numbers:
                return "No valid numbers found in input."

            # Calculate statistics
            count = len(numbers)
            total = sum(numbers)
            mean = total / count
            sorted_nums = sorted(numbers)

            # Median
            mid = count // 2
            if count % 2 == 0:
                median = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
            else:
                median = sorted_nums[mid]

            # Standard deviation
            variance = sum((x - mean) ** 2 for x in numbers) / count
            std_dev = variance ** 0.5

            # Range
            min_val, max_val = min(numbers), max(numbers)

            analysis = f"""Data Analysis Results ({data_source}):

📊 Basic Statistics:
   • Count: {count} values
   • Sum: {total:.2f}
   • Mean: {mean:.2f}
   • Median: {median:.2f}
   • Range: {min_val:.2f} - {max_val:.2f}
   • Standard Deviation: {std_dev:.2f}

📈 Distribution:
   • Minimum: {min_val:.2f}
   • Maximum: {max_val:.2f}
   • Spread: {max_val - min_val:.2f}

💡 Insights:
   • Data variability: {'High' if std_dev > mean * 0.3 else 'Moderate' if std_dev > mean * 0.1 else 'Low'}
   • Distribution: {'Skewed' if abs(mean - median) > std_dev * 0.5 else 'Relatively normal'}
   • Outliers: {'Possible outliers detected' if any(abs(x - mean) > 2 * std_dev for x in numbers) else 'No significant outliers'}
            """

            return analysis

        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    async def _arun(self, data_input: str) -> str:
        return self._run(data_input)


class CodeReviewTool(BaseTool):
    """Tool for reviewing and analyzing code"""

    name: str = "code_reviewer"
    description: str = "Review code for quality, bugs, and improvements. Input: programming language and code separated by '|' (e.g., 'python|print(hello)')"

    def _run(self, input_str: str) -> str:
        """Review code for quality and issues"""
        try:
            if '|' not in input_str:
                return "Please provide input as 'language|code'"

            language, code = input_str.split('|', 1)
            language = language.strip().lower()
            code = code.strip()

            if not code:
                return "No code provided for review."

            issues = []
            suggestions = []
            good_practices = []

            # Language-specific analysis
            if language == "python":
                lines = code.split('\n')

                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()

                    # Check line length
                    if len(line) > 100:
                        issues.append(f"Line {i}: Line too long ({len(line)} chars). PEP 8 recommends max 79 chars.")

                    # Check for print statements
                    if 'print(' in line and not line_stripped.startswith('#'):
                        suggestions.append(f"Line {i}: Consider using logging instead of print() for production code.")

                    # Check for proper naming
                    if 'def ' in line_stripped:
                        func_name = line_stripped.split('def ')[1].split('(')[0]
                        if not func_name.islower() or '-' in func_name:
                            issues.append(f"Line {i}: Function name '{func_name}' should be lowercase with underscores.")

                    # Check for missing docstrings
                    if line_stripped.startswith('def ') and i < len(lines) - 1:
                        next_line = lines[i].strip()
                        if not next_line.startswith('"""') and not next_line.startswith("'''"):
                            suggestions.append(f"Line {i}: Consider adding docstring to function.")

                    # Check for bare except
                    if 'except:' in line_stripped:
                        issues.append(f"Line {i}: Avoid bare except clauses. Specify exception types.")

                    # Check for good practices
                    if 'if __name__ == "__main__"' in line_stripped:
                        good_practices.append(f"Line {i}: Good practice using main guard.")

                    if line_stripped.startswith('"""') or line_stripped.startswith("'''"):
                        good_practices.append(f"Line {i}: Good documentation with docstring.")

            elif language in ["javascript", "js"]:
                lines = code.split('\n')

                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()

                    # Check for var usage
                    if line_stripped.startswith('var '):
                        suggestions.append(f"Line {i}: Consider using 'let' or 'const' instead of 'var'.")

                    # Check for console.log
                    if 'console.log(' in line:
                        suggestions.append(f"Line {i}: Remove console.log() statements before production.")

                    # Check for == vs ===
                    if '==' in line and '===' not in line and '!=' in line:
                        suggestions.append(f"Line {i}: Consider using strict equality (=== or !==).")

            # Generate review summary
            review = f"""Code Review Results ({language.upper()}):

📄 Code Analysis:
   • Lines of code: {len(code.split())}
   • Language: {language.upper()}
   • Complexity: {'Simple' if len(code.split()) < 50 else 'Moderate' if len(code.split()) < 200 else 'Complex'}

"""

            if issues:
                review += "🚨 Issues Found:\n"
                for issue in issues:
                    review += f"   • {issue}\n"
                review += "\n"

            if suggestions:
                review += "💡 Suggestions:\n"
                for suggestion in suggestions:
                    review += f"   • {suggestion}\n"
                review += "\n"

            if good_practices:
                review += "✅ Good Practices:\n"
                for practice in good_practices:
                    review += f"   • {practice}\n"
                review += "\n"

            if not issues and not suggestions:
                review += "✅ No major issues found! Code looks good.\n\n"

            review += "🎯 Overall Rating: "
            if len(issues) == 0:
                review += "Excellent ⭐⭐⭐⭐⭐"
            elif len(issues) <= 2:
                review += "Good ⭐⭐⭐⭐"
            elif len(issues) <= 5:
                review += "Needs Improvement ⭐⭐⭐"
            else:
                review += "Needs Major Revision ⭐⭐"

            return review

        except Exception as e:
            return f"Error reviewing code: {str(e)}"

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)


class ProjectPlannerTool(BaseTool):
    """Tool for creating and managing project plans"""

    name: str = "project_planner"
    description: str = "Create detailed project plans with timelines and milestones. Input: project description"

    def _run(self, project_description: str) -> str:
        """Create a detailed project plan"""
        try:
            # Analyze project type
            desc_lower = project_description.lower()

            if any(word in desc_lower for word in ['website', 'web app', 'frontend', 'backend']):
                project_type = "web_development"
            elif any(word in desc_lower for word in ['mobile app', 'ios', 'android', 'app']):
                project_type = "mobile_app"
            elif any(word in desc_lower for word in ['ai', 'ml', 'machine learning', 'data science']):
                project_type = "ai_ml"
            elif any(word in desc_lower for word in ['api', 'service', 'microservice']):
                project_type = "api_service"
            else:
                project_type = "general_software"

            # Generate appropriate plan
            if project_type == "web_development":
                phases = [
                    ("Planning & Design", "1-2 weeks", [
                        "Define requirements and scope",
                        "Create wireframes and mockups",
                        "Choose technology stack",
                        "Set up development environment"
                    ]),
                    ("Frontend Development", "2-3 weeks", [
                        "Implement UI components",
                        "Add responsive design",
                        "Integrate with backend APIs",
                        "Add client-side validation"
                    ]),
                    ("Backend Development", "2-3 weeks", [
                        "Design database schema",
                        "Implement API endpoints",
                        "Add authentication/authorization",
                        "Implement business logic"
                    ]),
                    ("Testing & Deployment", "1 week", [
                        "Write unit and integration tests",
                        "Perform user acceptance testing",
                        "Deploy to production environment",
                        "Set up monitoring and logging"
                    ])
                ]

            elif project_type == "mobile_app":
                phases = [
                    ("Research & Planning", "1 week", [
                        "Market research and user analysis",
                        "Define app features and scope",
                        "Choose platform and framework",
                        "Create app architecture"
                    ]),
                    ("UI/UX Design", "1-2 weeks", [
                        "Design user interface mockups",
                        "Create user experience flows",
                        "Design app icons and assets",
                        "Review and iterate on designs"
                    ]),
                    ("Core Development", "3-4 weeks", [
                        "Implement main app features",
                        "Add navigation and screens",
                        "Integrate device capabilities",
                        "Implement data persistence"
                    ]),
                    ("Testing & Launch", "1 week", [
                        "Perform device testing",
                        "Submit to app stores",
                        "Create marketing materials",
                        "Plan launch strategy"
                    ])
                ]

            elif project_type == "ai_ml":
                phases = [
                    ("Problem Definition", "1 week", [
                        "Define ML problem and objectives",
                        "Identify required data sources",
                        "Choose appropriate algorithms",
                        "Set up development environment"
                    ]),
                    ("Data Preparation", "2-3 weeks", [
                        "Collect and clean training data",
                        "Perform exploratory data analysis",
                        "Feature engineering and selection",
                        "Split data into train/validation/test"
                    ]),
                    ("Model Development", "2-3 weeks", [
                        "Implement baseline models",
                        "Experiment with different algorithms",
                        "Hyperparameter tuning",
                        "Model evaluation and validation"
                    ]),
                    ("Deployment & Monitoring", "1-2 weeks", [
                        "Deploy model to production",
                        "Implement monitoring and logging",
                        "Create inference API",
                        "Set up retraining pipeline"
                    ])
                ]

            else:  # general_software
                phases = [
                    ("Planning & Analysis", "1 week", [
                        "Gather and analyze requirements",
                        "Create system architecture",
                        "Choose development tools",
                        "Set up project structure"
                    ]),
                    ("Core Development", "3-4 weeks", [
                        "Implement core functionality",
                        "Add user interface",
                        "Integrate external services",
                        "Handle error cases"
                    ]),
                    ("Testing & Quality", "1 week", [
                        "Write comprehensive tests",
                        "Perform code review",
                        "Document the system",
                        "Optimize performance"
                    ]),
                    ("Deployment & Maintenance", "1 week", [
                        "Deploy to production",
                        "Set up monitoring",
                        "Create user documentation",
                        "Plan maintenance strategy"
                    ])
                ]

            # Format the plan
            plan = f"""📋 PROJECT PLAN: {project_description}

🎯 Project Type: {project_type.replace('_', ' ').title()}
⏱️ Estimated Duration: {sum(range(1, len(phases) + 1))} - {sum(range(2, len(phases) + 2))} weeks

"""

            for i, (phase_name, duration, tasks) in enumerate(phases, 1):
                plan += f"## Phase {i}: {phase_name} ({duration})\n"
                for task in tasks:
                    plan += f"   ✓ {task}\n"
                plan += f"\n"

            plan += """🎯 SUCCESS METRICS:
   • All planned features implemented
   • Code quality standards met
   • Performance requirements satisfied
   • User acceptance criteria passed

📊 RISK MITIGATION:
   • Regular progress reviews
   • Continuous integration/testing
   • Stakeholder feedback loops
   • Buffer time for unexpected issues

🤝 RECOMMENDED TEAM:
   • Project Manager (planning & coordination)
   • Developer(s) (implementation)
   • Designer (UI/UX)
   • QA Tester (quality assurance)
            """

            return plan

        except Exception as e:
            return f"Error creating project plan: {str(e)}"

    async def _arun(self, project_description: str) -> str:
        return self._run(project_description)


def demonstrate_advanced_agent():
    """Demonstrate advanced agent with multiple sophisticated tools"""

    print("🚀 Advanced Multi-Tool Agent Demo")
    print("="*35)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    # Initialize LLM with specific settings for better reasoning
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo",
        max_tokens=1000
    )

    # Create advanced tools
    tools = [
        TaskManagerTool(),
        DataAnalyzerTool(),
        CodeReviewTool(),
        ProjectPlannerTool()
    ]

    # Advanced memory with window to keep recent context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10  # Keep last 10 exchanges
    )

    # Initialize advanced agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

    print("\n🤖 Advanced Agent initialized with tools:")
    for tool in tools:
        print(f"   • {tool.name}: {tool.description}")

    # Complex multi-step tasks
    advanced_tasks = [
        "I'm starting a new web development project for an e-commerce site. Can you create a project plan for this?",

        "Now create some initial tasks for the first phase of this project",

        "Here's some sample data from our current sales: 150, 200, 175, 300, 250, 180, 220, 190, 280, 210. Can you analyze this data and give me insights?",

        """Please review this Python code for our user authentication system:
        python|
def login_user(username, password):
    users = get_users_from_database()
    for user in users:
        if user.username == username and user.password == password:
            return True
    return False

def register_user(username, password, email):
    if len(password) < 6:
        return "Password too short"
    new_user = User(username=username, password=password, email=email)
    save_user(new_user)
    return "User registered successfully"
        """,

        "List all the tasks we created earlier and update the first one to 'in_progress'",

        "Based on the data analysis and code review, what should be our priorities for the e-commerce project?"
    ]

    print("\n🧪 Testing Advanced Agent with Complex Tasks:")
    for i, task in enumerate(advanced_tasks, 1):
        print(f"\n{'='*60}")
        print(f"TASK {i}")
        print(f"{'='*60}")
        print(f"Human: {task}")
        print()

        try:
            start_time = time.time()
            response = agent.run(task)
            end_time = time.time()

            print(f"\n🤖 Agent Response:")
            print(response)
            print(f"\n⏱️ Response time: {end_time - start_time:.2f} seconds")

        except Exception as e:
            print(f"❌ Error: {e}")

        if i < len(advanced_tasks):
            input("\nPress Enter to continue to next task...")

    print(f"\n📊 Final Memory State:")
    print("Recent conversation context maintained across all interactions.")


def demonstrate_agent_customization():
    """Show how to customize agent behavior and prompts"""

    print("\n🎨 Agent Customization Demo")
    print("="*30)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    # Custom system message for specialized behavior
    custom_system_message = """You are a Senior Software Engineering Consultant AI. You have extensive experience in:
    - Software architecture and design patterns
    - Code quality and best practices
    - Project management and planning
    - Data analysis and insights

    Your responses should be:
    - Professional and detailed
    - Include specific recommendations
    - Provide actionable insights
    - Consider both technical and business perspectives

    Always explain your reasoning and provide concrete next steps.
    """

    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-3.5-turbo",
        max_tokens=1500
    )

    # Specialized tools for software consulting
    consulting_tools = [
        ProjectPlannerTool(),
        CodeReviewTool(),
        DataAnalyzerTool()
    ]

    # Create consultant agent
    consultant_memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5
    )

    consultant_agent = initialize_agent(
        tools=consulting_tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=consultant_memory,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": custom_system_message
        }
    )

    # Consulting scenarios
    consulting_scenarios = [
        "Our startup is building a SaaS platform for small businesses. We have a team of 3 developers and 6 months to launch. What's your recommended approach?",

        "We're seeing performance issues with our Python API. Here's a critical function - can you review it?\npython|def process_user_data(user_list):\n    results = []\n    for user in user_list:\n        for item in user.items:\n            for tag in item.tags:\n                if tag.active:\n                    results.append(calculate_score(user, item, tag))\n    return results",

        "Our monthly revenue data shows: 45000, 48000, 52000, 49000, 55000, 58000. As our technical consultant, what insights can you provide?"
    ]

    print("\n💼 Software Engineering Consultant Agent:")
    for i, scenario in enumerate(consulting_scenarios, 1):
        print(f"\n--- Consulting Session {i} ---")
        print(f"Client: {scenario}")

        try:
            response = consultant_agent.run(scenario)
            print(f"\n🎯 Consultant: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

        if i < len(consulting_scenarios):
            input("\nPress Enter for next consultation...")


def main():
    """Run all advanced agent demonstrations"""

    print("🚀 LANGCHAIN ADVANCED AGENTS")
    print("="*35)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for advanced agent demos")
        return

    print("✅ Starting advanced LangChain agent demonstrations...")

    demos = [
        ("Multi-Tool Agent", demonstrate_advanced_agent),
        ("Customized Consultant Agent", demonstrate_agent_customization)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'='*60}")
        print(f"ADVANCED DEMO {i}: {name.upper()}")
        print(f"{'='*60}")

        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n⏸️ Advanced demo {i} interrupted")
            break
        except Exception as e:
            print(f"❌ Advanced demo {i} error: {e}")
            continue

        if i < len(demos):
            input(f"\nPress Enter to continue to advanced demo {i+1}...")

    print("\n🎉 Advanced LangChain agent demonstrations completed!")
    print("\n💡 Advanced Concepts Covered:")
    print("1. Custom Tool Development: Build specialized tools for specific domains")
    print("2. Complex Multi-Step Tasks: Handle interconnected operations")
    print("3. Memory Management: Maintain context across complex interactions")
    print("4. Agent Customization: Tailor behavior for specific roles")
    print("5. Error Handling: Robust error management in complex scenarios")
    print("6. Performance Monitoring: Track and optimize agent performance")


if __name__ == "__main__":
    main()