"""
CrewAI Enterprise Consulting Exercise
Build a Complete Enterprise Consulting Crew using CrewAI

This exercise demonstrates building a sophisticated multi-agent consulting team that can:
1. Conduct comprehensive business analysis
2. Develop strategic recommendations
3. Create implementation roadmaps
4. Generate executive presentations
5. Provide financial modeling and analysis
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()


class EnterpriseAnalysisTool(BaseTool):
    """Enterprise business analysis tool"""
    name: str = "enterprise_analysis_tool"
    description: str = "Analyzes enterprise business data, processes, and performance metrics"

    def _run(self, analysis_request: str) -> str:
        """Execute enterprise analysis"""

        # Mock comprehensive enterprise analysis
        analysis_framework = {
            "financial_health": {
                "revenue_growth": "15% YoY",
                "profit_margin": "18%",
                "cash_flow": "Positive $2.3M",
                "debt_ratio": "0.25"
            },
            "operational_efficiency": {
                "productivity_metrics": "Above industry average",
                "automation_level": "65% of processes",
                "quality_scores": "4.2/5.0",
                "cycle_times": "Reduced by 22%"
            },
            "market_position": {
                "market_share": "12% in primary segment",
                "competitive_ranking": "Top 3 in region",
                "brand_recognition": "Strong in B2B segment",
                "customer_satisfaction": "Net Promoter Score: 58"
            },
            "technology_infrastructure": {
                "digital_maturity": "Intermediate level",
                "system_integration": "70% integrated",
                "data_quality": "Good with gaps",
                "security_posture": "Compliant with standards"
            }
        }

        return f"""
ENTERPRISE ANALYSIS REPORT: {analysis_request}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
The enterprise demonstrates strong fundamentals with opportunities for digital transformation and operational optimization.

DETAILED ANALYSIS:

1. FINANCIAL PERFORMANCE:
• Revenue Growth: {analysis_framework['financial_health']['revenue_growth']}
• Profit Margin: {analysis_framework['financial_health']['profit_margin']}
• Cash Flow: {analysis_framework['financial_health']['cash_flow']}
• Debt Management: Debt-to-equity ratio of {analysis_framework['financial_health']['debt_ratio']}

2. OPERATIONAL EXCELLENCE:
• Productivity: {analysis_framework['operational_efficiency']['productivity_metrics']}
• Automation: {analysis_framework['operational_efficiency']['automation_level']}
• Quality: {analysis_framework['operational_efficiency']['quality_scores']}
• Efficiency: {analysis_framework['operational_efficiency']['cycle_times']}

3. MARKET DYNAMICS:
• Market Position: {analysis_framework['market_position']['market_share']}
• Competition: {analysis_framework['market_position']['competitive_ranking']}
• Brand Strength: {analysis_framework['market_position']['brand_recognition']}
• Customer Loyalty: {analysis_framework['market_position']['customer_satisfaction']}

4. TECHNOLOGY ASSESSMENT:
• Digital Readiness: {analysis_framework['technology_infrastructure']['digital_maturity']}
• Integration Level: {analysis_framework['technology_infrastructure']['system_integration']}
• Data Management: {analysis_framework['technology_infrastructure']['data_quality']}
• Cybersecurity: {analysis_framework['technology_infrastructure']['security_posture']}

KEY FINDINGS:
• Strong financial foundation provides investment capacity
• Operational processes show efficiency but need modernization
• Market position is solid with growth potential
• Technology infrastructure requires strategic upgrades

RISK ASSESSMENT:
• Market disruption from digital-first competitors
• Talent gaps in technology and digital skills
• Legacy system dependencies creating technical debt
• Regulatory compliance in evolving landscape

OPPORTUNITY IDENTIFICATION:
• Digital transformation initiatives
• Market expansion into adjacent segments
• Strategic partnerships and acquisitions
• Operational automation and AI integration
"""

    async def _arun(self, analysis_request: str) -> str:
        return self._run(analysis_request)


class StrategicPlanningTool(BaseTool):
    """Strategic planning and roadmap development tool"""
    name: str = "strategic_planning_tool"
    description: str = "Develops comprehensive strategic plans, initiatives, and implementation roadmaps"

    def _run(self, planning_scope: str) -> str:
        """Execute strategic planning"""

        strategic_plan = f"""
STRATEGIC PLAN: {planning_scope}
Planning Horizon: 3 Years (2024-2027)
Generated: {datetime.now().strftime('%Y-%m-%d')}

STRATEGIC VISION:
"To become the leading digital-first enterprise in our industry while maintaining operational excellence and customer-centric growth."

STRATEGIC OBJECTIVES:

1. DIGITAL TRANSFORMATION (Priority: High)
Objective: Achieve 90% digital process automation by 2026
Key Initiatives:
• Cloud migration and infrastructure modernization
• AI/ML integration across operations
• Customer experience digitalization
• Data-driven decision making implementation

Timeline: 18 months
Investment Required: $3.2M
Expected ROI: 240% over 3 years

2. MARKET EXPANSION (Priority: High)
Objective: Increase market share from 12% to 20% by 2027
Key Initiatives:
• Geographic expansion to 3 new markets
• Product line extension and innovation
• Strategic partnerships and alliances
• Customer acquisition and retention programs

Timeline: 24 months
Investment Required: $2.8M
Expected ROI: 180% over 3 years

3. OPERATIONAL EXCELLENCE (Priority: Medium)
Objective: Reduce operational costs by 25% while improving quality
Key Initiatives:
• Lean process optimization
• Supply chain digitalization
• Quality management system enhancement
• Workforce development and training

Timeline: 12 months
Investment Required: $1.5M
Expected ROI: 320% over 2 years

4. INNOVATION & R&D (Priority: Medium)
Objective: Launch 5 new products/services by 2026
Key Initiatives:
• Innovation lab establishment
• Customer co-creation programs
• Technology scouting and evaluation
• Intellectual property development

Timeline: 30 months
Investment Required: $2.1M
Expected ROI: 150% over 4 years

IMPLEMENTATION ROADMAP:

PHASE 1: FOUNDATION (Months 1-6)
• Leadership alignment and change management
• Technology infrastructure assessment
• Team building and capability development
• Quick wins identification and execution

PHASE 2: ACCELERATION (Months 7-18)
• Core digital transformation initiatives
• Market expansion pilot programs
• Process optimization implementation
• Performance monitoring system deployment

PHASE 3: SCALE (Months 19-36)
• Full-scale digital operations
• Multi-market presence establishment
• Innovation pipeline commercialization
• Continuous improvement culture embedding

SUCCESS METRICS:

Financial Metrics:
• Revenue growth: 25% CAGR
• EBITDA improvement: 35% by year 3
• Market capitalization increase: 150%
• Return on invested capital: >15%

Operational Metrics:
• Process automation: 90%
• Customer satisfaction: NPS >70
• Employee engagement: >85%
• Time-to-market reduction: 40%

Market Metrics:
• Market share growth: 12% to 20%
• Brand recognition: Top 2 in industry
• Customer retention: >95%
• New product revenue: 30% of total

RESOURCE REQUIREMENTS:

Human Capital:
• Digital transformation team: 15 specialists
• Market expansion team: 12 professionals
• Operations optimization team: 8 experts
• Innovation team: 10 researchers

Financial Investment:
• Total 3-year investment: $9.6M
• Year 1: $4.2M (44%)
• Year 2: $3.1M (32%)
• Year 3: $2.3M (24%)

Technology Infrastructure:
• Cloud platform migration
• AI/ML development environment
• Data analytics and BI systems
• Cybersecurity enhancement

RISK MITIGATION:

Strategic Risks:
• Market disruption: Continuous monitoring and agility
• Technology obsolescence: Regular assessment and updates
• Competitive response: Differentiation and innovation focus
• Regulatory changes: Compliance team and legal monitoring

Operational Risks:
• Implementation delays: Agile project management
• Budget overruns: Financial controls and milestone gates
• Talent shortages: Early recruitment and development
• Change resistance: Comprehensive change management

Financial Risks:
• ROI shortfall: Phased investment approach
• Cash flow constraints: Working capital optimization
• Currency fluctuations: Hedging strategies
• Economic downturns: Scenario planning and flexibility

This strategic plan provides a comprehensive roadmap for transformation while ensuring sustainable growth and competitive advantage.
"""

        return strategic_plan

    async def _arun(self, planning_scope: str) -> str:
        return self._run(planning_scope)


class FinancialModelingTool(BaseTool):
    """Financial modeling and analysis tool"""
    name: str = "financial_modeling_tool"
    description: str = "Creates comprehensive financial models, projections, and investment analysis"

    def _run(self, modeling_request: str) -> str:
        """Execute financial modeling"""

        financial_model = f"""
COMPREHENSIVE FINANCIAL MODEL: {modeling_request}
Model Date: {datetime.now().strftime('%Y-%m-%d')}
Projection Period: 5 Years (2024-2029)

REVENUE MODEL AND PROJECTIONS:

Base Case Scenario:
Year 1 (2024): $12.5M (+15% YoY)
Year 2 (2025): $15.6M (+25% growth from digital initiatives)
Year 3 (2026): $19.8M (+27% growth from market expansion)
Year 4 (2027): $24.2M (+22% growth from new products)
Year 5 (2029): $28.1M (+16% mature growth rate)

Revenue Stream Breakdown:
• Core Business (60%): Steady growth with optimization
• New Markets (25%): Aggressive expansion in Years 2-4
• Innovation Products (15%): Launched in Year 3

COST STRUCTURE ANALYSIS:

Operating Expenses Projection:
Year 1: $9.8M (78% of revenue)
Year 2: $11.4M (73% of revenue - efficiency gains)
Year 3: $13.7M (69% of revenue - scale benefits)
Year 4: $16.0M (66% of revenue - automation impact)
Year 5: $17.5M (62% of revenue - mature operations)

Cost Categories:
• Personnel (45%): $5.6M Year 1 → $7.9M Year 5
• Technology (20%): $2.5M Year 1 → $3.5M Year 5
• Marketing (15%): $1.9M Year 1 → $2.6M Year 5
• Operations (12%): $1.5M Year 1 → $2.1M Year 5
• General & Admin (8%): $1.0M Year 1 → $1.4M Year 5

PROFITABILITY ANALYSIS:

Gross Margin Evolution:
Year 1: 78% → Year 5: 85% (process optimization)

EBITDA Progression:
Year 1: $2.7M (22% margin)
Year 2: $4.2M (27% margin)
Year 3: $6.1M (31% margin)
Year 4: $8.2M (34% margin)
Year 5: $10.6M (38% margin)

Net Income Projection:
Year 1: $1.8M (14% net margin)
Year 2: $3.1M (20% net margin)
Year 3: $4.8M (24% net margin)
Year 4: $6.5M (27% net margin)
Year 5: $8.7M (31% net margin)

CASH FLOW ANALYSIS:

Operating Cash Flow:
Year 1: $2.1M
Year 2: $3.8M
Year 3: $5.6M
Year 4: $7.4M
Year 5: $9.5M

Free Cash Flow:
Year 1: $0.8M (after $1.3M CapEx)
Year 2: $2.2M (after $1.6M CapEx)
Year 3: $3.7M (after $1.9M CapEx)
Year 4: $5.4M (after $2.0M CapEx)
Year 5: $7.4M (after $2.1M CapEx)

Cumulative FCF (5 years): $19.5M

INVESTMENT ANALYSIS:

Capital Requirements:
• Digital Transformation: $3.2M (Years 1-2)
• Market Expansion: $2.8M (Years 2-3)
• Operational Excellence: $1.5M (Year 1)
• Innovation & R&D: $2.1M (Years 2-4)
Total Investment: $9.6M

ROI Analysis:
• Payback Period: 2.8 years
• Net Present Value (10% discount): $14.2M
• Internal Rate of Return: 34.7%
• Return on Investment: 285% over 5 years

VALUATION METRICS:

Revenue Multiple Valuation:
Current Industry Average: 2.8x revenue
Projected Year 5 Revenue: $28.1M
Estimated Valuation: $78.7M

EBITDA Multiple Valuation:
Current Industry Average: 12x EBITDA
Projected Year 5 EBITDA: $10.6M
Estimated Valuation: $127.2M

Discounted Cash Flow Valuation:
Terminal Value (3% growth): $158.3M
Present Value of Cash Flows: $42.1M
Enterprise Value: $200.4M
Equity Value (after debt): $185.2M

SENSITIVITY ANALYSIS:

Revenue Growth Scenarios:
• Conservative (-20%): NPV = $8.9M, IRR = 28.2%
• Base Case (0%): NPV = $14.2M, IRR = 34.7%
• Optimistic (+20%): NPV = $21.8M, IRR = 42.1%

Cost Management Scenarios:
• Poor Control (+10% costs): NPV = $7.3M, IRR = 25.8%
• Base Case (0%): NPV = $14.2M, IRR = 34.7%
• Excellent Control (-10% costs): NPV = $23.4M, IRR = 45.2%

FINANCIAL RATIOS AND METRICS:

Liquidity Ratios:
• Current Ratio: 2.1 → 2.8 (improving)
• Quick Ratio: 1.6 → 2.2 (strengthening)
• Cash Ratio: 0.8 → 1.4 (excellent)

Efficiency Ratios:
• Asset Turnover: 1.4 → 1.8 (improving)
• Inventory Turnover: 8.2 → 12.5 (optimization)
• Receivables Turnover: 6.8 → 8.4 (better collection)

Profitability Ratios:
• Gross Margin: 78% → 85% (process improvement)
• Operating Margin: 22% → 38% (scale benefits)
• Net Margin: 14% → 31% (comprehensive optimization)

Leverage Ratios:
• Debt-to-Equity: 0.25 → 0.15 (deleveraging)
• Interest Coverage: 12.4x → 28.6x (strong)
• Debt Service Coverage: 3.2x → 7.8x (excellent)

FUNDING REQUIREMENTS AND STRATEGY:

Total Funding Needed: $9.6M over 3 years
Recommended Funding Mix:
• Internal Cash Generation: $4.8M (50%)
• Bank Term Loan: $2.4M (25%)
• Strategic Investor/PE: $2.4M (25%)

Funding Timeline:
• Year 1: $4.2M (internal + loan)
• Year 2: $3.1M (investor capital)
• Year 3: $2.3M (cash flow funded)

RISK ANALYSIS AND MITIGATION:

Financial Risks:
• Revenue shortfall: Diversification and agility
• Cost overruns: Rigorous budget controls
• Cash flow gaps: Credit facility and monitoring
• Market volatility: Scenario planning

Mitigation Strategies:
• Quarterly financial reviews and adjustments
• Conservative cash management
• Flexible cost structure development
• Multiple revenue stream cultivation

This comprehensive financial model provides a robust framework for investment decisions and strategic planning with clear risk assessment and mitigation strategies.
"""

        return financial_model

    async def _arun(self, modeling_request: str) -> str:
        return self._run(modeling_request)


class PresentationTool(BaseTool):
    """Executive presentation creation tool"""
    name: str = "presentation_tool"
    description: str = "Creates professional executive presentations and reports"

    def _run(self, presentation_brief: str) -> str:
        """Create executive presentation"""

        presentation = f"""
EXECUTIVE PRESENTATION: {presentation_brief}
Presentation Date: {datetime.now().strftime('%Y-%m-%d')}
Duration: 45 minutes + 15 minutes Q&A

SLIDE DECK STRUCTURE:

=== SLIDE 1: TITLE SLIDE ===
"Strategic Transformation Initiative"
"Roadmap to Digital Leadership and Sustainable Growth"

Presented by: Enterprise Consulting Team
Date: {datetime.now().strftime('%B %d, %Y')}
Confidential & Proprietary

=== SLIDE 2: EXECUTIVE SUMMARY ===
"Three-Year Strategic Initiative Overview"

Key Message: "Transform into a digital-first market leader while achieving 25% revenue CAGR"

Summary Points:
• Total Investment: $9.6M over 3 years
• Expected ROI: 285% with 2.8-year payback
• Market Share Growth: 12% → 20%
• EBITDA Margin Expansion: 22% → 38%

=== SLIDE 3: CURRENT STATE ANALYSIS ===
"Where We Stand Today"

Strengths:
✓ Strong financial foundation (15% revenue growth)
✓ Market-leading position in core segment
✓ Experienced management team
✓ Loyal customer base (NPS: 58)

Challenges:
• Limited digital capabilities (35% manual processes)
• Geographic concentration risk
• Aging technology infrastructure
• Emerging competitive threats

=== SLIDE 4: STRATEGIC VISION & OBJECTIVES ===
"Our Vision for 2027"

Vision Statement:
"To be the leading digital-first enterprise in our industry"

Strategic Pillars:
1. Digital Transformation Excellence
2. Market Leadership Expansion
3. Operational Efficiency Mastery
4. Innovation-Driven Growth

Success Metrics:
• Revenue: $24.2M by 2027 (25% CAGR)
• Market Share: 20% (from 12%)
• Digital Processes: 90% automated
• Customer Satisfaction: NPS >70

=== SLIDE 5: INITIATIVE ROADMAP ===
"Three-Phase Implementation Strategy"

PHASE 1: Foundation (Months 1-6)
• Leadership alignment & change management
• Technology infrastructure assessment
• Quick wins identification and delivery
• Team capability building

PHASE 2: Acceleration (Months 7-18)
• Core digital transformation rollout
• Market expansion pilot programs
• Process optimization implementation
• Performance monitoring deployment

PHASE 3: Scale (Months 19-36)
• Full digital operations establishment
• Multi-market presence consolidation
• Innovation pipeline commercialization
• Continuous improvement culture

=== SLIDE 6: FINANCIAL PROJECTIONS ===
"Compelling Financial Returns"

Revenue Growth Trajectory:
2024: $12.5M → 2027: $24.2M (94% total growth)

Profitability Improvement:
• EBITDA: $2.7M → $8.2M (204% increase)
• Net Margin: 14% → 27% (13 point improvement)
• ROI: 285% over 5 years

Investment Summary:
• Total Investment: $9.6M
• Payback Period: 2.8 years
• NPV (10% discount): $14.2M
• IRR: 34.7%

=== SLIDE 7: MARKET OPPORTUNITY ===
"Significant Growth Potential"

Market Expansion Strategy:
• Current Market: $104M total addressable
• Target Markets: 3 new geographic regions
• Addressable Opportunity: $280M by 2027

Competitive Positioning:
• Current: #3 in region (12% share)
• Target: #2 in expanded market (20% share)
• Differentiation: Digital-first capabilities

Customer Value Proposition:
• 40% faster service delivery
• 25% cost reduction for clients
• 95% reliability guarantee
• 24/7 digital accessibility

=== SLIDE 8: DIGITAL TRANSFORMATION ===
"Technology as Competitive Advantage"

Digital Initiatives:
• Cloud Infrastructure Migration (6 months)
• AI/ML Operations Integration (12 months)
• Customer Experience Digitalization (18 months)
• Data-Driven Decision Platform (24 months)

Expected Benefits:
• Process Automation: 65% → 90%
• Operational Efficiency: +35%
• Customer Response Time: -60%
• Decision Speed: +50%

Technology Investment:
• Infrastructure: $1.8M
• Software & Platforms: $1.2M
• Implementation & Training: $0.8M

=== SLIDE 9: RISK MANAGEMENT ===
"Comprehensive Risk Mitigation Strategy"

Strategic Risks & Mitigation:
• Market Disruption → Continuous innovation & agility
• Technology Obsolescence → Regular assessment & updates
• Competitive Response → Differentiation & first-mover advantage
• Talent Shortages → Early recruitment & development programs

Financial Risks & Controls:
• Budget Overruns → Milestone-based funding & controls
• ROI Shortfall → Phased approach with go/no-go gates
• Cash Flow → Conservative planning & credit facilities
• Economic Downturn → Scenario planning & flexibility

Implementation Risks & Solutions:
• Change Resistance → Comprehensive change management
• Execution Delays → Agile methodology & expert partners
• Integration Challenges → Proven technology partners
• Performance Gaps → Regular monitoring & course correction

=== SLIDE 10: IMPLEMENTATION PLAN ===
"Detailed Execution Framework"

Governance Structure:
• Steering Committee: CEO, CFO, CTO, COO
• Project Management Office: Dedicated PMO team
• Workstream Leads: Subject matter experts
• External Partners: Implementation specialists

Resource Allocation:
• Project Team: 45 dedicated professionals
• Budget Distribution: 44% Year 1, 32% Year 2, 24% Year 3
• Technology Partners: 3 strategic vendors
• Change Management: Organization-wide program

Success Tracking:
• Weekly progress reviews
• Monthly steering committee reports
• Quarterly board updates
• Annual strategy refresh

=== SLIDE 11: NEXT STEPS ===
"Immediate Actions Required"

Decision Points:
1. Strategic Initiative Approval
2. Budget Authorization ($9.6M)
3. Resource Allocation Confirmation
4. Timeline Agreement

Immediate Actions (Next 30 Days):
• Project team establishment
• Technology partner selection
• Change management planning
• Detailed project scheduling

First 90 Days Milestones:
• Infrastructure assessment completion
• Quick wins identification and initiation
• Market expansion feasibility studies
• Performance baseline establishment

Success Measures:
• Board approval by [Date + 2 weeks]
• Project kickoff by [Date + 6 weeks]
• First quarterly review by [Date + 14 weeks]
• Year 1 targets achievement

=== SLIDE 12: Q&A ===
"Questions & Discussion"

Anticipated Questions & Responses:
• ROI Confidence? → Based on conservative assumptions with 34.7% IRR
• Implementation Risk? → Proven methodology with expert partners
• Market Timing? → First-mover advantage in digital transformation
• Resource Requirements? → Detailed capability assessment completed
• Competitive Response? → Differentiated approach with innovation focus

Contact Information:
Enterprise Consulting Team
[Contact details]

PRESENTATION DELIVERY NOTES:

Opening (5 minutes):
• Welcome and team introductions
• Presentation overview and objectives
• Key message reinforcement

Main Content (30 minutes):
• 2-3 minutes per slide with focus on visuals
• Pause for clarifying questions only
• Emphasize financial returns and strategic benefits

Conclusion (10 minutes):
• Summary of key recommendations
• Clear call to action
• Next steps timeline

Q&A Session (15 minutes):
• Address concerns directly and confidently
• Refer to detailed appendix materials
• Commit to follow-up on complex questions

SUPPORTING MATERIALS:
• Detailed financial model (Excel)
• Risk analysis matrix
• Implementation timeline (Gantt chart)
• Technology assessment report
• Market research summary
• Competitive analysis deep-dive
• Change management plan

This presentation is designed to secure executive approval and funding for the strategic transformation initiative while demonstrating thorough planning and compelling returns.
"""

        return presentation

    async def _arun(self, presentation_brief: str) -> str:
        return self._run(presentation_brief)


def run_enterprise_consulting_exercise():
    """Run the comprehensive enterprise consulting crew exercise"""

    print("🏢 ENTERPRISE CONSULTING CREW EXERCISE")
    print("Complete Business Transformation Consulting")
    print("=" * 60)

    # Check API availability
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        return

    # Initialize LLM
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    else:
        llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)

    print("✅ LLM initialized, creating enterprise consulting crew...")

    try:
        # Create specialized consulting tools
        analysis_tool = EnterpriseAnalysisTool()
        planning_tool = StrategicPlanningTool()
        financial_tool = FinancialModelingTool()
        presentation_tool = PresentationTool()

        # Define the consulting team agents
        senior_analyst = Agent(
            role="Senior Business Analyst",
            goal="Conduct comprehensive enterprise analysis and identify strategic opportunities and challenges",
            backstory="""You are a senior business analyst with 15+ years of experience in enterprise
            consulting. You specialize in business process analysis, financial assessment, market evaluation,
            and operational efficiency. Your analyses are thorough, data-driven, and provide actionable insights
            that drive strategic decision-making.""",
            tools=[analysis_tool],
            llm=llm,
            verbose=True
        )

        strategy_consultant = Agent(
            role="Senior Strategy Consultant",
            goal="Develop comprehensive strategic plans and transformation roadmaps based on business analysis",
            backstory="""You are a senior strategy consultant with expertise in business transformation,
            digital strategy, and organizational change. You have successfully led multiple enterprise
            transformation initiatives and excel at creating implementable strategic roadmaps that balance
            ambition with pragmatism.""",
            tools=[planning_tool],
            llm=llm,
            verbose=True
        )

        financial_advisor = Agent(
            role="Senior Financial Advisor",
            goal="Create detailed financial models, projections, and investment analysis to support strategic decisions",
            backstory="""You are a senior financial advisor and former investment banker with deep expertise
            in financial modeling, valuation, and corporate finance. You create comprehensive financial models
            that help executives make informed investment decisions and understand the financial implications
            of strategic initiatives.""",
            tools=[financial_tool],
            llm=llm,
            verbose=True
        )

        presentation_specialist = Agent(
            role="Executive Presentation Specialist",
            goal="Create compelling executive presentations that communicate strategic recommendations effectively",
            backstory="""You are an executive communication specialist with extensive experience in creating
            high-impact presentations for C-level executives and boards. You excel at distilling complex
            analysis into clear, compelling narratives that drive decision-making and secure buy-in for
            strategic initiatives.""",
            tools=[presentation_tool],
            llm=llm,
            verbose=True
        )

        # Define the consulting engagement scenario
        client_brief = """
        Mid-market manufacturing company ($12.5M annual revenue) seeking strategic transformation
        to compete with digital-first competitors. The company has strong market position but faces
        challenges from technology disruption, changing customer expectations, and operational
        inefficiencies. Leadership wants a comprehensive transformation plan with clear ROI and
        implementation roadmap.

        Key challenges identified:
        - Legacy systems and manual processes
        - Limited digital customer experience
        - Geographic market concentration
        - Aging workforce with skill gaps
        - Competitive pressure on margins
        """

        print(f"\n📋 CLIENT ENGAGEMENT BRIEF:")
        print("=" * 40)
        print(client_brief)

        # Define the comprehensive consulting tasks
        business_analysis_task = Task(
            description=f"""
            Conduct a comprehensive enterprise analysis for the client based on this brief: {client_brief}

            Your analysis should include:
            1. Current state assessment (financial, operational, market, technology)
            2. Strengths, weaknesses, opportunities, and threats analysis
            3. Key performance indicators and benchmarking
            4. Risk assessment and challenge identification
            5. Opportunity mapping and prioritization

            Provide detailed insights that will inform strategic planning and investment decisions.
            """,
            agent=senior_analyst,
            expected_output="Comprehensive enterprise analysis report with current state assessment, SWOT analysis, and opportunity identification"
        )

        strategic_planning_task = Task(
            description="""
            Based on the enterprise analysis, develop a comprehensive 3-year strategic transformation plan.

            The strategic plan should include:
            1. Vision, mission, and strategic objectives
            2. Key strategic initiatives and priorities
            3. Implementation roadmap with phases and milestones
            4. Resource requirements and capability development
            5. Success metrics and performance indicators
            6. Risk mitigation strategies

            Create a detailed, implementable strategy that addresses identified challenges and capitalizes on opportunities.
            """,
            agent=strategy_consultant,
            expected_output="Detailed 3-year strategic transformation plan with implementation roadmap and success metrics",
            context=[business_analysis_task]
        )

        financial_modeling_task = Task(
            description="""
            Create a comprehensive financial model and investment analysis based on the strategic plan.

            The financial model should include:
            1. 5-year revenue and expense projections
            2. Cash flow analysis and funding requirements
            3. Return on investment calculations and payback analysis
            4. Valuation impact and shareholder value creation
            5. Sensitivity analysis and scenario planning
            6. Financial risk assessment and mitigation

            Provide detailed financial justification for the strategic transformation initiative.
            """,
            agent=financial_advisor,
            expected_output="Comprehensive financial model with projections, ROI analysis, and investment recommendations",
            context=[business_analysis_task, strategic_planning_task]
        )

        executive_presentation_task = Task(
            description="""
            Create a compelling executive presentation that synthesizes the analysis, strategy, and financial model.

            The presentation should include:
            1. Executive summary with key recommendations
            2. Current state analysis and strategic rationale
            3. Strategic plan overview and implementation roadmap
            4. Financial projections and investment case
            5. Risk management and mitigation strategies
            6. Next steps and decision points

            Design the presentation for a board-level audience with clear calls to action.
            """,
            agent=presentation_specialist,
            expected_output="Executive presentation with comprehensive strategic recommendations and clear next steps",
            context=[business_analysis_task, strategic_planning_task, financial_modeling_task]
        )

        # Create and execute the consulting crew
        consulting_crew = Crew(
            agents=[senior_analyst, strategy_consultant, financial_advisor, presentation_specialist],
            tasks=[business_analysis_task, strategic_planning_task, financial_modeling_task, executive_presentation_task],
            process=Process.sequential,
            verbose=True
        )

        print(f"\n🚀 EXECUTING ENTERPRISE CONSULTING ENGAGEMENT")
        print("=" * 55)

        print("Phase 1: Business Analysis and Current State Assessment...")
        print("Phase 2: Strategic Planning and Roadmap Development...")
        print("Phase 3: Financial Modeling and Investment Analysis...")
        print("Phase 4: Executive Presentation Preparation...")

        result = consulting_crew.kickoff()

        print(f"\n📊 CONSULTING ENGAGEMENT RESULTS")
        print("=" * 40)
        print("✅ Enterprise analysis completed")
        print("✅ Strategic transformation plan developed")
        print("✅ Financial model and ROI analysis created")
        print("✅ Executive presentation prepared")

        # Display engagement summary
        print(f"\n📈 ENGAGEMENT SUMMARY")
        print("-" * 25)
        print("Consulting Team: 4 senior specialists")
        print("Analysis Scope: Comprehensive enterprise assessment")
        print("Strategic Horizon: 3-year transformation plan")
        print("Financial Model: 5-year projections and ROI analysis")
        print("Deliverable: Board-ready executive presentation")

        if hasattr(result, 'raw'):
            print(f"\nFinal deliverable preview:")
            print("-" * 30)
            print(f"{str(result)[:1000]}...")
            print(f"\nTotal content length: {len(str(result))} characters")

        print(f"\n🎯 EXERCISE OBJECTIVES ACHIEVED")
        print("=" * 35)
        print("✅ Multi-agent consulting team collaboration")
        print("✅ Comprehensive enterprise business analysis")
        print("✅ Strategic planning and roadmap development")
        print("✅ Detailed financial modeling and ROI analysis")
        print("✅ Executive-level presentation creation")
        print("✅ End-to-end consulting engagement simulation")

        print(f"\n🎉 Enterprise Consulting Crew Exercise Completed Successfully!")

    except Exception as e:
        print(f"❌ Consulting engagement error: {e}")


if __name__ == "__main__":
    run_enterprise_consulting_exercise()


# EXERCISE EXTENSIONS:
#
# 1. Add industry-specific analysis modules
# 2. Implement competitive intelligence gathering
# 3. Create change management and communication plans
# 4. Add regulatory compliance and legal considerations
# 5. Develop talent strategy and organizational design
# 6. Create technology vendor evaluation frameworks
# 7. Add customer journey mapping and experience design
# 8. Implement sustainability and ESG assessment
# 9. Create merger & acquisition analysis capabilities
# 10. Add post-implementation monitoring and optimization