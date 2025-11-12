"""
CrewAI Basics - Updated for 2025 Standalone Architecture
Introduction to building multi-agent systems with modern CrewAI patterns
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Modern CrewAI standalone imports (no LangChain dependency)
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from crewai.llm import LLM

load_dotenv()


# Modern CrewAI tool definitions using @tool decorator
@tool("research_web")
def research_web(topic: str) -> str:
    """Search and research information about a given topic from web sources."""
    # Mock research implementation
    research_results = {
        "artificial intelligence": "AI is revolutionizing industries through machine learning, deep learning, and neural networks. Key applications include natural language processing, computer vision, and autonomous systems. Market size expected to reach $1.8 trillion by 2030.",
        "renewable energy": "Renewable energy sources like solar, wind, and hydroelectric power are becoming increasingly cost-effective. Global investment in renewables reached $1.8 trillion in 2023. Solar costs dropped 77% since 2010.",
        "quantum computing": "Quantum computing promises exponential speedups for certain computational problems. Major companies like IBM, Google, and Microsoft are investing heavily. Current focus on NISQ (Noisy Intermediate-Scale Quantum) devices.",
        "blockchain": "Blockchain technology provides decentralized, secure transaction recording. Applications extend beyond cryptocurrency to supply chain management, smart contracts, and digital identity. Enterprise adoption growing by 87% annually."
    }

    for key, result in research_results.items():
        if key.lower() in topic.lower():
            return f"Research findings on {topic}: {result}"

    return f"Research findings on {topic}: Emerging field with significant potential for innovation and growth. Multiple stakeholders actively developing solutions and applications."


@tool("analyze_data")
def analyze_data(data: str) -> str:
    """Analyze data and provide insights, recommendations, and structured conclusions."""
    word_count = len(data.split())

    analysis = f"""
ANALYSIS REPORT
===============

Data Overview:
- Content Length: {len(data)} characters
- Word Count: {word_count} words
- Complexity Level: {"High" if word_count > 100 else "Medium" if word_count > 50 else "Low"}

Key Insights:
1. Primary themes and patterns identified in the content
2. Data quality and completeness assessment
3. Potential opportunities and challenges highlighted

Strategic Recommendations:
1. Focus on high-impact initiatives based on data insights
2. Leverage existing strengths and competitive advantages
3. Address identified gaps with targeted interventions
4. Implement continuous monitoring and optimization

Confidence Level: High (based on comprehensive data analysis)

Next Steps:
- Validate findings with stakeholders and domain experts
- Develop detailed implementation timeline and milestones
- Establish KPIs and success metrics
- Begin execution with regular progress reviews
"""
    return analysis


@tool("create_content")
def create_content(content_brief: str) -> str:
    """Create well-structured, professional content based on provided requirements."""

    if "report" in content_brief.lower():
        return f"""
EXECUTIVE REPORT
===============

EXECUTIVE SUMMARY
Based on: {content_brief}

This report presents comprehensive findings and strategic recommendations based on thorough research and analysis.

KEY FINDINGS
• Market dynamics show strong growth potential
• Technology adoption accelerating across sectors
• Competitive landscape evolving rapidly
• Regulatory environment increasingly supportive

STRATEGIC RECOMMENDATIONS
1. Immediate Actions (0-3 months)
   - Establish cross-functional task force
   - Conduct detailed market assessment
   - Develop pilot program framework

2. Medium-term Initiatives (3-12 months)
   - Scale successful pilot programs
   - Build strategic partnerships
   - Enhance operational capabilities

3. Long-term Vision (12+ months)
   - Market leadership positioning
   - Platform expansion and diversification
   - International market entry

CONCLUSION
The analysis indicates significant opportunity for growth and market expansion with proper strategic execution and resource allocation.
"""

    elif "article" in content_brief.lower():
        return f"""
FEATURED ARTICLE
===============

Title: Insights from Recent Analysis
Brief: {content_brief}

In today's rapidly evolving landscape, organizations must adapt quickly to changing market conditions and emerging opportunities.

The current market dynamics present both challenges and opportunities for forward-thinking organizations. Recent research indicates that companies leveraging data-driven decision making are 23% more likely to acquire customers and 19% more likely to be profitable.

Key trends shaping the industry:
• Digital transformation acceleration
• Increased focus on sustainability
• Remote work normalization
• AI and automation adoption
• Customer experience prioritization

Organizations that proactively address these trends while maintaining operational excellence will be best positioned for long-term success.

Looking ahead, the emphasis on innovation, agility, and customer-centricity will continue to differentiate market leaders from followers.
"""

    else:
        return f"""
CONTENT CREATION OUTPUT
======================

Based on: {content_brief}

This content has been crafted to meet the specified requirements while maintaining high quality and professional standards.

The content addresses key stakeholder needs and provides actionable insights that drive meaningful outcomes.

Key elements incorporated:
- Clear structure and organization
- Evidence-based recommendations
- Stakeholder-focused messaging
- Actionable next steps
- Professional presentation

This approach ensures maximum impact and value for the intended audience.
"""


@tool("quality_review")
def quality_review(content: str) -> str:
    """Review content for quality, accuracy, and completeness."""

    word_count = len(content.split())

    return f"""
QUALITY REVIEW REPORT
====================

Content Metrics:
- Length: {len(content)} characters
- Word Count: {word_count} words
- Structure: {"Well-organized" if "=" in content else "Needs improvement"}

Quality Assessment:
✓ Content clarity and readability
✓ Professional tone and style
✓ Logical flow and structure
✓ Completeness of information
✓ Accuracy of statements

Recommendations:
- Content meets professional standards
- Information appears accurate and relevant
- Structure supports readability
- Tone appropriate for target audience

Overall Score: {95 if word_count > 100 else 85}/100

STATUS: APPROVED FOR PUBLICATION
"""


def demonstrate_basic_crew():
    """Demonstrate basic CrewAI crew functionality"""

    print("🤖 CrewAI Basic Crew Demo")
    print("=" * 28)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ API key required (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        return

    # Create agents with modern CrewAI patterns
    researcher = Agent(
        role='Senior Researcher',
        goal='Conduct thorough research on assigned topics and provide comprehensive, accurate information',
        backstory="""You are an experienced researcher with expertise in multiple domains.
        You excel at finding reliable sources, synthesizing information, and presenting clear,
        actionable insights. Your research is always thorough, accurate, and well-documented.""",
        tools=[research_web],
        verbose=True,
        allow_delegation=False
    )

    analyst = Agent(
        role='Data Analyst',
        goal='Analyze information and provide strategic insights and recommendations',
        backstory="""You are a skilled data analyst with a strong background in strategy consulting.
        You specialize in transforming raw information into actionable insights and strategic
        recommendations that drive business value.""",
        tools=[analyze_data],
        verbose=True,
        allow_delegation=False
    )

    writer = Agent(
        role='Content Writer',
        goal='Create high-quality, engaging content based on research and analysis',
        backstory="""You are a professional content writer with expertise in business communication.
        You excel at transforming complex information into clear, compelling content that resonates
        with diverse audiences and drives engagement.""",
        tools=[create_content, quality_review],
        verbose=True,
        allow_delegation=False
    )

    # Define tasks
    research_task = Task(
        description="""Research the current state of artificial intelligence in healthcare.
        Focus on recent developments, key applications, challenges, and future opportunities.
        Provide comprehensive findings that can inform strategic planning.""",
        expected_output="""A detailed research report covering AI in healthcare including:
        - Current market state and key players
        - Major applications and use cases
        - Technical and regulatory challenges
        - Future opportunities and trends
        - Strategic recommendations""",
        agent=researcher,
        output_file="research_findings.md"
    )

    analysis_task = Task(
        description="""Analyze the research findings to identify strategic opportunities
        and provide actionable recommendations for organizations looking to leverage
        AI in healthcare.""",
        expected_output="""Strategic analysis including:
        - Market opportunity assessment
        - Competitive landscape analysis
        - Risk and challenge evaluation
        - Strategic recommendations with priorities
        - Implementation roadmap""",
        agent=analyst,
        context=[research_task],
        output_file="strategic_analysis.md"
    )

    writing_task = Task(
        description="""Create a comprehensive executive brief based on the research
        and analysis. The content should be professional, engaging, and suitable
        for senior leadership decision-making.""",
        expected_output="""Executive brief including:
        - Executive summary with key insights
        - Market overview and opportunities
        - Strategic recommendations
        - Implementation priorities
        - Risk mitigation strategies""",
        agent=writer,
        context=[research_task, analysis_task],
        output_file="executive_brief.md"
    )

    # Create and run crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=True,
        memory=True
    )

    print("\n🚀 Starting crew execution...")
    print("Agents: Researcher → Analyst → Writer")
    print("Process: Sequential with context sharing")

    try:
        result = crew.kickoff()

        print("\n✅ Crew execution completed!")
        print("\nFinal Output:")
        print("-" * 40)
        print(result)

    except Exception as e:
        print(f"❌ Crew execution error: {e}")


def demonstrate_parallel_crew():
    """Demonstrate parallel crew execution"""

    print("\n🔄 CrewAI Parallel Crew Demo")
    print("=" * 32)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("❌ API key required")
        return

    # Create specialized agents for parallel processing
    market_researcher = Agent(
        role='Market Research Specialist',
        goal='Research market trends and competitive landscape',
        backstory="""Expert in market research with deep knowledge of industry trends,
        competitive dynamics, and consumer behavior patterns.""",
        tools=[research_web, analyze_data],
        verbose=True
    )

    tech_researcher = Agent(
        role='Technology Research Specialist',
        goal='Research technical developments and innovations',
        backstory="""Technology expert specializing in emerging tech trends,
        innovation patterns, and technical feasibility analysis.""",
        tools=[research_web, analyze_data],
        verbose=True
    )

    content_creator = Agent(
        role='Content Strategist',
        goal='Synthesize research into compelling content',
        backstory="""Content strategy expert skilled at transforming research
        into engaging, actionable content for diverse audiences.""",
        tools=[create_content, quality_review],
        verbose=True
    )

    # Parallel tasks
    market_task = Task(
        description="Research renewable energy market trends and opportunities",
        expected_output="Market trends analysis with key insights and opportunities",
        agent=market_researcher
    )

    tech_task = Task(
        description="Research latest renewable energy technology innovations",
        expected_output="Technology innovation report with emerging solutions",
        agent=tech_researcher
    )

    content_task = Task(
        description="""Create executive summary combining market and technology research.
        Synthesize findings into strategic recommendations.""",
        expected_output="Executive summary with integrated insights and recommendations",
        agent=content_creator,
        context=[market_task, tech_task]
    )

    # Parallel crew
    parallel_crew = Crew(
        agents=[market_researcher, tech_researcher, content_creator],
        tasks=[market_task, tech_task, content_task],
        process=Process.hierarchical,  # Manager coordinates parallel execution
        verbose=True,
        memory=True
    )

    print("\n🚀 Starting parallel crew execution...")
    print("Process: Hierarchical with parallel research → synthesis")

    try:
        result = parallel_crew.kickoff()
        print("\n✅ Parallel crew completed!")
        print("\nIntegrated Results:")
        print("-" * 40)
        print(result)

    except Exception as e:
        print(f"❌ Parallel crew error: {e}")


def demonstrate_crew_with_flows():
    """Demonstrate CrewAI Flows for deterministic workflows"""

    print("\n🌊 CrewAI Flows Demo (2025 Feature)")
    print("=" * 35)

    print("💡 CrewAI Flows Concept:")
    print("Flows provide deterministic, event-driven orchestration")
    print("Perfect for production workflows requiring precise control")

    print("\n🏗️ Example Flow Structure:")
    print("""
    Flow: Content Production Pipeline
    ├── Event: Content Request Received
    ├── Step 1: Research Phase (Crew execution)
    ├── Step 2: Analysis Phase (Crew execution)
    ├── Step 3: Content Creation (Crew execution)
    ├── Step 4: Quality Review (Crew execution)
    └── Event: Content Published
    """)

    print("\n🎯 Flow Benefits:")
    print("• Predictable execution paths")
    print("• Event-driven triggers")
    print("• State management and persistence")
    print("• Integration with external systems")
    print("• Production monitoring and logging")

    print("\n📝 Implementation Example:")
    print("""
    # CrewAI Flow (Conceptual - 2025 Feature)
    from crewai.flows import Flow, FlowStep

    content_flow = Flow("content_production")

    @content_flow.step("research")
    def research_step(request):
        crew = create_research_crew()
        return crew.kickoff(inputs=request)

    @content_flow.step("analyze")
    def analysis_step(research_results):
        crew = create_analysis_crew()
        return crew.kickoff(inputs=research_results)

    @content_flow.step("create")
    def creation_step(analysis_results):
        crew = create_writing_crew()
        return crew.kickoff(inputs=analysis_results)

    # Execute flow
    result = content_flow.run({
        "topic": "AI in Healthcare",
        "output_format": "executive_brief"
    })
    """)


def main():
    """Run all CrewAI demonstrations"""

    print("👥 CREWAI 2025 COMPREHENSIVE DEMO")
    print("=" * 35)

    # Check requirements
    api_available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

    if not api_available:
        print("❌ API keys required (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        print("\n💡 This demo showcases modern CrewAI 2025 patterns:")
        print("• Standalone architecture (no LangChain dependency)")
        print("• Modern @tool decorators for tool creation")
        print("• Crews for autonomous collaboration")
        print("• Flows for deterministic orchestration")
        print("• Enhanced performance and scalability")
        return

    print("✅ API keys found, starting CrewAI demonstrations...")
    print("\n🌟 CrewAI 2025 Features:")
    print("• 5.76x faster execution than LangGraph")
    print("• Standalone architecture for better performance")
    print("• Dual workflow management (Crews + Flows)")
    print("• Enterprise-ready with deep customization")
    print("• 30.5K GitHub stars, 1M+ monthly downloads")

    demos = [
        ("Basic Crew Execution", demonstrate_basic_crew),
        ("Parallel Crew Processing", demonstrate_parallel_crew),
        ("CrewAI Flows (2025)", demonstrate_crew_with_flows)
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

    print("\n🎉 CrewAI demonstrations completed!")
    print("\n💡 Modern CrewAI 2025 Concepts Covered:")
    print("1. Standalone Architecture: No external framework dependencies")
    print("2. Modern Tools: @tool decorators for easy integration")
    print("3. Agent Collaboration: Role-based autonomous teams")
    print("4. Dual Workflows: Crews (autonomous) + Flows (deterministic)")
    print("5. Performance: Significant speed improvements over alternatives")
    print("6. Enterprise Ready: Deep customization and production features")

    print("\n🚀 Migration from Legacy CrewAI:")
    print("• Remove LangChain tool dependencies")
    print("• Use @tool decorator instead of BaseTool classes")
    print("• Leverage new Flows for production workflows")
    print("• Enable memory and embedder configurations")

    print("\n➡️ Continue to Module 7 (SmolAgents) for lightweight agents!")


if __name__ == "__main__":
    main()