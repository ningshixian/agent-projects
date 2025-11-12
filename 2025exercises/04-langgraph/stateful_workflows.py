"""
LangGraph Stateful Workflows
Advanced examples of building complex, stateful agent workflows
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from datetime import datetime, timedelta

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class ProjectState(TypedDict):
    """State for project management workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    project_name: str
    requirements: List[str]
    tasks: List[Dict[str, Any]]
    current_phase: str
    completed_phases: List[str]
    team_assignments: Dict[str, List[str]]
    timeline: Dict[str, str]
    risks: List[Dict[str, Any]]
    status_report: str


class DocumentProcessingState(TypedDict):
    """State for document processing workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_type: str
    content_chunks: List[str]
    extracted_entities: Dict[str, List[str]]
    summary: str
    key_insights: List[str]
    action_items: List[Dict[str, Any]]
    processing_stage: str
    metadata: Dict[str, Any]


def demonstrate_project_management_workflow():
    """Demonstrate a complex project management workflow with multiple phases"""

    print("📋 LangGraph Project Management Workflow Demo")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # Phase 1: Requirements Analysis
    def requirements_analysis(state: ProjectState):
        """Analyze project requirements and create initial structure"""
        project_name = state.get("project_name", "Unknown Project")
        messages = state.get("messages", [])

        # Get initial requirements from user input
        user_input = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break

        prompt = f"""
        Analyze the following project request and extract key requirements:

        Project: {project_name}
        Description: {user_input}

        Please provide:
        1. A list of 5-7 main requirements
        2. Suggested team roles needed
        3. Estimated timeline phases

        Format your response as a structured analysis.
        """

        response = llm.invoke([HumanMessage(content=prompt)])

        # Parse response into requirements (simplified extraction)
        requirements = [
            "Define project scope and objectives",
            "Gather stakeholder requirements",
            "Create technical architecture",
            "Develop implementation plan",
            "Set up testing framework",
            "Plan deployment strategy"
        ]

        return {
            "messages": [response],
            "requirements": requirements,
            "current_phase": "planning",
            "completed_phases": ["requirements"],
            "timeline": {"requirements": "Week 1"},
            "team_assignments": {},
            "risks": []
        }

    # Phase 2: Project Planning
    def project_planning(state: ProjectState):
        """Create detailed project plan with tasks and assignments"""
        requirements = state.get("requirements", [])
        project_name = state.get("project_name", "Project")

        prompt = f"""
        Based on these requirements for {project_name}:
        {chr(10).join(f'- {req}' for req in requirements)}

        Create a detailed project plan including:
        1. Breakdown of tasks for each requirement
        2. Dependencies between tasks
        3. Risk assessment
        4. Team roles and responsibilities

        Focus on creating actionable tasks.
        """

        response = llm.invoke([HumanMessage(content=prompt)])

        # Create mock tasks based on requirements
        tasks = []
        for i, req in enumerate(requirements):
            tasks.append({
                "id": f"task_{i+1}",
                "name": req,
                "status": "pending",
                "assignee": "TBD",
                "estimated_hours": 8 + (i * 2),
                "dependencies": []
            })

        # Mock team assignments
        team_assignments = {
            "Project Manager": ["task_1", "task_2"],
            "Lead Developer": ["task_3", "task_4"],
            "QA Engineer": ["task_5"],
            "DevOps Engineer": ["task_6"]
        }

        # Mock risks
        risks = [
            {"type": "Technical", "description": "Integration complexity", "probability": "Medium", "impact": "High"},
            {"type": "Resource", "description": "Team availability", "probability": "Low", "impact": "Medium"},
            {"type": "Timeline", "description": "Scope creep", "probability": "High", "impact": "Medium"}
        ]

        updated_timeline = state.get("timeline", {})
        updated_timeline.update({
            "planning": "Week 2",
            "development": "Weeks 3-8",
            "testing": "Weeks 9-10",
            "deployment": "Week 11"
        })

        return {
            "messages": [response],
            "tasks": tasks,
            "current_phase": "execution",
            "completed_phases": state.get("completed_phases", []) + ["planning"],
            "team_assignments": team_assignments,
            "timeline": updated_timeline,
            "risks": risks
        }

    # Phase 3: Execution Monitoring
    def execution_monitoring(state: ProjectState):
        """Monitor project execution and update status"""
        tasks = state.get("tasks", [])
        team_assignments = state.get("team_assignments", {})
        risks = state.get("risks", [])

        # Simulate task progress
        completed_tasks = 0
        in_progress_tasks = 0

        for task in tasks:
            if task["status"] == "completed":
                completed_tasks += 1
            elif task["status"] == "in_progress":
                in_progress_tasks += 1

        # Update some tasks to show progress
        if tasks:
            tasks[0]["status"] = "completed"
            if len(tasks) > 1:
                tasks[1]["status"] = "in_progress"

        progress_percentage = (completed_tasks / len(tasks)) * 100 if tasks else 0

        status_prompt = f"""
        Generate a project status report with the following information:

        Total tasks: {len(tasks)}
        Completed tasks: {completed_tasks + 1}  # +1 for the task we just marked complete
        In progress tasks: {in_progress_tasks + 1}  # +1 for the task we marked in progress
        Progress: {progress_percentage:.1f}%

        Active risks: {len(risks)}
        Team members: {len(team_assignments)}

        Provide a concise status summary and any recommendations.
        """

        response = llm.invoke([HumanMessage(content=status_prompt)])

        return {
            "messages": [response],
            "tasks": tasks,
            "current_phase": "monitoring",
            "completed_phases": state.get("completed_phases", []) + ["execution"],
            "status_report": response.content
        }

    # Phase 4: Project Review
    def project_review(state: ProjectState):
        """Final project review and recommendations"""
        status_report = state.get("status_report", "")
        completed_phases = state.get("completed_phases", [])
        risks = state.get("risks", [])

        review_prompt = f"""
        Conduct a project review based on:

        Completed phases: {', '.join(completed_phases)}
        Latest status: {status_report[:200]}...
        Risk count: {len(risks)}

        Provide:
        1. Overall project assessment
        2. Lessons learned
        3. Recommendations for future projects
        4. Next steps
        """

        response = llm.invoke([HumanMessage(content=review_prompt)])

        return {
            "messages": [response],
            "current_phase": "completed",
            "completed_phases": completed_phases + ["review"]
        }

    # Routing logic
    def route_project_workflow(state: ProjectState):
        """Route to next phase based on current state"""
        current_phase = state.get("current_phase", "requirements")
        completed_phases = state.get("completed_phases", [])

        if "requirements" not in completed_phases:
            return "requirements"
        elif "planning" not in completed_phases:
            return "planning"
        elif "execution" not in completed_phases:
            return "execution"
        elif "review" not in completed_phases:
            return "review"
        else:
            return END

    # Build the workflow
    workflow = StateGraph(ProjectState)

    # Add nodes
    workflow.add_node("requirements", requirements_analysis)
    workflow.add_node("planning", project_planning)
    workflow.add_node("execution", execution_monitoring)
    workflow.add_node("review", project_review)

    # Set entry point
    workflow.set_entry_point("requirements")

    # Add conditional edges
    workflow.add_conditional_edges(
        "requirements",
        route_project_workflow,
        {
            "planning": "planning",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "planning",
        route_project_workflow,
        {
            "execution": "execution",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "execution",
        route_project_workflow,
        {
            "review": "review",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "review",
        route_project_workflow,
        {
            END: END
        }
    )

    # Compile and test
    app = workflow.compile()

    print("\n🚀 Testing Project Management Workflow:")

    project_scenarios = [
        {
            "name": "E-commerce Platform",
            "description": "Build a modern e-commerce platform with user authentication, product catalog, shopping cart, and payment processing"
        },
        {
            "name": "AI Chatbot Service",
            "description": "Develop an AI-powered customer service chatbot with natural language understanding and integration with existing CRM systems"
        }
    ]

    for i, scenario in enumerate(project_scenarios, 1):
        print(f"\n--- Project Scenario {i}: {scenario['name']} ---")

        try:
            initial_state = {
                "messages": [HumanMessage(content=scenario["description"])],
                "project_name": scenario["name"],
                "requirements": [],
                "tasks": [],
                "current_phase": "requirements",
                "completed_phases": [],
                "team_assignments": {},
                "timeline": {},
                "risks": [],
                "status_report": ""
            }

            result = app.invoke(initial_state)

            print(f"✅ Project completed!")
            print(f"Final phase: {result.get('current_phase', 'unknown')}")
            print(f"Completed phases: {', '.join(result.get('completed_phases', []))}")
            print(f"Total tasks created: {len(result.get('tasks', []))}")
            print(f"Team roles: {len(result.get('team_assignments', {}))}")
            print(f"Identified risks: {len(result.get('risks', []))}")

            # Show final recommendation
            final_messages = result.get("messages", [])
            if final_messages:
                final_response = final_messages[-1].content[:200] + "..."
                print(f"Final recommendation: {final_response}")

        except Exception as e:
            print(f"❌ Error in project workflow: {e}")

        print("-" * 70)


def demonstrate_document_processing_workflow():
    """Demonstrate document processing with entity extraction and analysis"""

    print("\n📄 LangGraph Document Processing Workflow Demo")
    print("=" * 55)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this demo")
        return

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

    # Stage 1: Document Classification
    def classify_document(state: DocumentProcessingState):
        """Classify the document type and extract basic metadata"""
        messages = state.get("messages", [])

        # Get document content from user input
        content = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content
                break

        classify_prompt = f"""
        Analyze this document and determine its type and key characteristics:

        Content: {content[:500]}...

        Please identify:
        1. Document type (contract, report, email, proposal, etc.)
        2. Key sections or structure
        3. Primary language and tone
        4. Estimated complexity level
        """

        response = llm.invoke([HumanMessage(content=classify_prompt)])

        # Simple document type classification
        doc_type = "business_document"
        if "contract" in content.lower() or "agreement" in content.lower():
            doc_type = "legal_contract"
        elif "report" in content.lower() or "analysis" in content.lower():
            doc_type = "analytical_report"
        elif "proposal" in content.lower() or "project" in content.lower():
            doc_type = "project_proposal"

        # Split content into chunks for processing
        chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

        metadata = {
            "length": len(content),
            "chunks": len(chunks),
            "classification_timestamp": datetime.now().isoformat()
        }

        return {
            "messages": [response],
            "document_type": doc_type,
            "content_chunks": chunks,
            "processing_stage": "entity_extraction",
            "metadata": metadata
        }

    # Stage 2: Entity Extraction
    def extract_entities(state: DocumentProcessingState):
        """Extract key entities from the document"""
        chunks = state.get("content_chunks", [])
        doc_type = state.get("document_type", "unknown")

        # Process first chunk for entities (in real implementation, would process all)
        chunk_to_process = chunks[0] if chunks else ""

        entity_prompt = f"""
        Extract key entities from this {doc_type} content:

        {chunk_to_process}

        Please identify:
        1. People/Organizations
        2. Dates and deadlines
        3. Financial amounts
        4. Key terms and concepts
        5. Action items or requirements
        """

        response = llm.invoke([HumanMessage(content=entity_prompt)])

        # Mock entity extraction results
        entities = {
            "people_orgs": ["Company A", "John Smith", "Legal Department"],
            "dates": ["2024-12-01", "Q4 2024"],
            "financial": ["$50,000", "$2.5M budget"],
            "key_terms": ["Service Level Agreement", "Data Protection", "Compliance"],
            "actions": ["Review contract terms", "Submit proposal", "Schedule meeting"]
        }

        return {
            "messages": [response],
            "extracted_entities": entities,
            "processing_stage": "summarization"
        }

    # Stage 3: Summarization
    def create_summary(state: DocumentProcessingState):
        """Create document summary and key insights"""
        chunks = state.get("content_chunks", [])
        entities = state.get("extracted_entities", {})
        doc_type = state.get("document_type", "document")

        main_content = chunks[0] if chunks else ""

        summary_prompt = f"""
        Create a comprehensive summary of this {doc_type}:

        Content: {main_content[:800]}

        Extracted entities: {json.dumps(entities, indent=2)}

        Provide:
        1. Executive summary (2-3 sentences)
        2. Key points (3-5 bullet points)
        3. Important insights or recommendations
        4. Critical action items with priorities
        """

        response = llm.invoke([HumanMessage(content=summary_prompt)])

        # Generate insights and action items
        insights = [
            "Document requires attention to compliance requirements",
            "Timeline appears aggressive for stated deliverables",
            "Budget allocation needs clarification",
            "Key stakeholders need to be aligned on expectations"
        ]

        action_items = [
            {"item": "Review legal terms", "priority": "High", "deadline": "Within 3 days"},
            {"item": "Validate budget estimates", "priority": "Medium", "deadline": "End of week"},
            {"item": "Schedule stakeholder meeting", "priority": "High", "deadline": "Within 5 days"},
            {"item": "Prepare implementation timeline", "priority": "Low", "deadline": "Next week"}
        ]

        return {
            "messages": [response],
            "summary": response.content,
            "key_insights": insights,
            "action_items": action_items,
            "processing_stage": "reporting"
        }

    # Stage 4: Report Generation
    def generate_report(state: DocumentProcessingState):
        """Generate final processing report"""
        doc_type = state.get("document_type", "document")
        entities = state.get("extracted_entities", {})
        insights = state.get("key_insights", [])
        action_items = state.get("action_items", [])
        metadata = state.get("metadata", {})

        report_prompt = f"""
        Generate a final document processing report:

        Document Type: {doc_type}
        Processing Metadata: {json.dumps(metadata, indent=2)}

        Entity Summary:
        - People/Organizations: {len(entities.get('people_orgs', []))}
        - Dates: {len(entities.get('dates', []))}
        - Financial items: {len(entities.get('financial', []))}
        - Key terms: {len(entities.get('key_terms', []))}

        Insights: {len(insights)}
        Action items: {len(action_items)}

        Provide a final summary of the document processing results and next steps.
        """

        response = llm.invoke([HumanMessage(content=report_prompt)])

        return {
            "messages": [response],
            "processing_stage": "completed"
        }

    # Routing logic
    def route_document_workflow(state: DocumentProcessingState):
        """Route document processing based on current stage"""
        stage = state.get("processing_stage", "classification")

        if stage == "entity_extraction":
            return "extract_entities"
        elif stage == "summarization":
            return "summarize"
        elif stage == "reporting":
            return "report"
        else:
            return END

    # Build workflow
    workflow = StateGraph(DocumentProcessingState)

    # Add nodes
    workflow.add_node("classify", classify_document)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("summarize", create_summary)
    workflow.add_node("report", generate_report)

    # Set entry point
    workflow.set_entry_point("classify")

    # Add conditional edges
    workflow.add_conditional_edges(
        "classify",
        route_document_workflow,
        {
            "extract_entities": "extract_entities",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "extract_entities",
        route_document_workflow,
        {
            "summarize": "summarize",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "summarize",
        route_document_workflow,
        {
            "report": "report",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "report",
        route_document_workflow,
        {
            END: END
        }
    )

    # Compile and test
    app = workflow.compile()

    print("\n📊 Testing Document Processing Workflow:")

    # Sample documents to process
    sample_documents = [
        """
        PROJECT PROPOSAL: AI-Powered Customer Service Enhancement

        Executive Summary:
        We propose implementing an AI-powered customer service enhancement system to improve response times and customer satisfaction. The project will integrate with existing CRM systems and provide 24/7 automated support capabilities.

        Budget: $250,000
        Timeline: 6 months
        Team: 4 developers, 2 QA engineers, 1 project manager

        Key Requirements:
        - Natural language processing capabilities
        - Integration with Salesforce CRM
        - Multi-channel support (chat, email, phone)
        - Real-time analytics dashboard
        - Compliance with data protection regulations

        Expected Outcomes:
        - 40% reduction in response time
        - 25% improvement in customer satisfaction scores
        - 60% reduction in routine inquiry handling time

        Next Steps:
        1. Stakeholder approval by December 1st
        2. Team assembly and kickoff by December 15th
        3. Phase 1 delivery by February 28th
        """,

        """
        QUARTERLY BUSINESS REPORT - Q3 2024

        Revenue Performance:
        Total revenue for Q3 2024 reached $2.8M, representing a 15% increase over Q2 2024 and 22% year-over-year growth.

        Key Metrics:
        - New customer acquisition: 1,247 customers
        - Customer retention rate: 94%
        - Average contract value: $12,500
        - Monthly recurring revenue: $890K

        Operational Highlights:
        - Launched new product features in September
        - Expanded team by 12 employees
        - Achieved 99.7% system uptime
        - Completed SOC 2 Type II compliance audit

        Challenges:
        - Supply chain delays affecting product delivery
        - Increased competition in core market segments
        - Rising customer acquisition costs

        Q4 Outlook:
        We project continued growth with revenue targets of $3.2M for Q4, driven by holiday season demand and new enterprise partnerships.

        Action Items:
        1. Address supply chain issues with alternative vendors
        2. Optimize marketing spend for better CAC ratio
        3. Prepare for annual board meeting in January
        """
    ]

    for i, document in enumerate(sample_documents, 1):
        print(f"\n--- Document Processing Test {i} ---")

        try:
            initial_state = {
                "messages": [HumanMessage(content=document)],
                "document_type": "",
                "content_chunks": [],
                "extracted_entities": {},
                "summary": "",
                "key_insights": [],
                "action_items": [],
                "processing_stage": "classification",
                "metadata": {}
            }

            result = app.invoke(initial_state)

            print(f"✅ Document processing completed!")
            print(f"Document type: {result.get('document_type', 'unknown')}")
            print(f"Content chunks: {len(result.get('content_chunks', []))}")
            print(f"Extracted entities: {len(result.get('extracted_entities', {}))}")
            print(f"Key insights: {len(result.get('key_insights', []))}")
            print(f"Action items: {len(result.get('action_items', []))}")
            print(f"Final stage: {result.get('processing_stage', 'unknown')}")

            # Show action items if available
            action_items = result.get('action_items', [])
            if action_items:
                print("\nGenerated Action Items:")
                for item in action_items[:3]:  # Show first 3
                    print(f"  • {item.get('item', 'N/A')} (Priority: {item.get('priority', 'N/A')})")

        except Exception as e:
            print(f"❌ Error in document processing: {e}")

        print("-" * 60)


def main():
    """Run all stateful workflow demonstrations"""

    print("🔄 LANGGRAPH STATEFUL WORKFLOWS DEMO")
    print("=" * 45)

    # Check requirements
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        return

    print("✅ API keys found, starting stateful workflow demonstrations...")

    demos = [
        ("Project Management Workflow", demonstrate_project_management_workflow),
        ("Document Processing Workflow", demonstrate_document_processing_workflow)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n{'=' * 70}")
        print(f"DEMO {i}: {name.upper()}")
        print(f"{'=' * 70}")

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

    print("\n🎉 Stateful workflow demonstrations completed!")
    print("\n💡 Key Concepts Demonstrated:")
    print("1. Multi-phase workflows with conditional routing")
    print("2. Complex state management across workflow stages")
    print("3. Dynamic task creation and assignment")
    print("4. Entity extraction and document analysis")
    print("5. Automated report generation")
    print("6. Error handling and workflow recovery")

    print("\n🔧 Advanced Features:")
    print("  • State persistence across workflow phases")
    print("  • Conditional branching based on content analysis")
    print("  • Dynamic task generation and prioritization")
    print("  • Multi-stage document processing pipeline")
    print("  • Automated insights and action item generation")

    print("\n➡️ These workflows can be extended with:")
    print("  • Real database integration")
    print("  • External API calls")
    print("  • File upload and processing")
    print("  • Notification systems")
    print("  • Advanced NLP and ML models")


if __name__ == "__main__":
    main()