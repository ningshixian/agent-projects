"""
Google Gemini Multimodal Business Assistant Exercise
Build a Complete Multimodal Business Assistant using Google Gemini

This exercise demonstrates building a sophisticated business assistant that can:
1. Analyze business documents (invoices, reports, contracts)
2. Process visual content (charts, images, presentations)
3. Extract and structure data from various formats
4. Generate insights and recommendations
5. Create business-ready outputs
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import tempfile
import base64
from dataclasses import dataclass

# Google AI imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()


@dataclass
class BusinessDocument:
    """Data class for business document information"""
    file_path: str
    document_type: str
    analysis_results: Dict[str, Any]
    extracted_data: Dict[str, Any]
    timestamp: str
    confidence_score: float


@dataclass
class BusinessInsight:
    """Data class for business insights"""
    insight_type: str
    title: str
    description: str
    impact_level: str  # low, medium, high, critical
    recommended_actions: List[str]
    supporting_data: Dict[str, Any]
    timestamp: str


class MultimodalBusinessAssistant:
    """Complete multimodal business assistant using Google Gemini"""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        # Initialize Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Configuration for different use cases
        self.precise_config = genai.types.GenerationConfig(
            temperature=0.1, top_p=0.8, max_output_tokens=4096
        )
        self.creative_config = genai.types.GenerationConfig(
            temperature=0.7, top_p=0.9, max_output_tokens=4096
        )

        # Storage for processed documents and insights
        self.processed_documents: List[BusinessDocument] = []
        self.business_insights: List[BusinessInsight] = []
        self.extracted_data_warehouse: Dict[str, Any] = {}

    def analyze_financial_document(self, document_path: str) -> BusinessDocument:
        """Analyze financial documents (invoices, receipts, financial reports)"""

        financial_analysis_prompt = """
        Analyze this financial document and extract comprehensive business intelligence:

        1. DOCUMENT CLASSIFICATION:
        - Document type (invoice, receipt, financial statement, report, etc.)
        - Document ID/number
        - Date and period covered
        - Currency used

        2. FINANCIAL DATA EXTRACTION:
        - All monetary amounts with labels
        - Line items with quantities and unit prices
        - Subtotals, taxes, discounts, and final totals
        - Payment terms and due dates
        - Account numbers or reference codes

        3. BUSINESS ENTITY INFORMATION:
        - Company names (payer and payee)
        - Contact information and addresses
        - Tax identification numbers
        - Business registration details

        4. TRANSACTION ANALYSIS:
        - Nature of the transaction
        - Recurring vs one-time payment
        - Category classification (operations, marketing, equipment, etc.)
        - Approval status or payment method

        5. COMPLIANCE AND AUDIT TRAIL:
        - Required fields completeness
        - Regulatory compliance indicators
        - Signature or authorization evidence
        - Supporting documentation references

        6. BUSINESS INSIGHTS:
        - Spending patterns indicated
        - Vendor/customer relationship insights
        - Cash flow implications
        - Budget category alignment

        7. ACTION ITEMS:
        - Required follow-up actions
        - Missing information to obtain
        - Approval workflows needed
        - Filing and record-keeping requirements

        Format response as detailed structured analysis with clear sections and actionable insights.
        """

        try:
            with open(document_path, 'rb') as file:
                file_data = file.read()

            file_part = {
                'mime_type': self._get_document_mime_type(document_path),
                'data': file_data
            }

            response = self.model.generate_content(
                [financial_analysis_prompt, file_part],
                generation_config=self.precise_config
            )

            # Extract structured data from response
            analysis_text = response.text if response.candidates else "Analysis failed"

            # Create business document record
            doc = BusinessDocument(
                file_path=document_path,
                document_type="financial",
                analysis_results={"full_analysis": analysis_text},
                extracted_data=self._extract_financial_data_from_analysis(analysis_text),
                timestamp=datetime.now().isoformat(),
                confidence_score=self._calculate_confidence_score(analysis_text)
            )

            self.processed_documents.append(doc)
            self._update_data_warehouse("financial", doc.extracted_data)

            return doc

        except Exception as e:
            return BusinessDocument(
                file_path=document_path,
                document_type="financial",
                analysis_results={"error": str(e)},
                extracted_data={},
                timestamp=datetime.now().isoformat(),
                confidence_score=0.0
            )

    def analyze_business_presentation(self, document_path: str) -> BusinessDocument:
        """Analyze business presentations and slide decks"""

        presentation_analysis_prompt = """
        Analyze this business presentation/slide deck comprehensively:

        1. PRESENTATION OVERVIEW:
        - Presentation title and purpose
        - Target audience identification
        - Total number of slides/pages
        - Presentation date and author

        2. CONTENT STRUCTURE ANALYSIS:
        - Main sections and topics covered
        - Logical flow and narrative arc
        - Key messages and value propositions
        - Supporting evidence and data

        3. BUSINESS METRICS AND DATA:
        - KPIs and performance indicators
        - Financial figures and projections
        - Market data and statistics
        - Growth rates and trends

        4. STRATEGIC INSIGHTS:
        - Business objectives and goals
        - Market opportunities identified
        - Competitive advantages highlighted
        - Risk factors mentioned

        5. VISUAL CONTENT ANALYSIS:
        - Charts and graphs effectiveness
        - Data visualization quality
        - Image and graphic usage
        - Design and branding consistency

        6. ACTIONABLE INTELLIGENCE:
        - Strategic recommendations
        - Investment opportunities
        - Operational improvements suggested
        - Market expansion possibilities

        7. STAKEHOLDER IMPLICATIONS:
        - Investor relations impact
        - Customer value propositions
        - Partner collaboration opportunities
        - Employee engagement factors

        8. COMPETITIVE ANALYSIS:
        - Competitive positioning
        - Market differentiation
        - Benchmark comparisons
        - Industry trends alignment

        Provide comprehensive business intelligence that would be valuable for strategic decision-making.
        """

        try:
            with open(document_path, 'rb') as file:
                file_data = file.read()

            file_part = {
                'mime_type': self._get_document_mime_type(document_path),
                'data': file_data
            }

            response = self.model.generate_content(
                [presentation_analysis_prompt, file_part],
                generation_config=self.precise_config
            )

            analysis_text = response.text if response.candidates else "Analysis failed"

            doc = BusinessDocument(
                file_path=document_path,
                document_type="presentation",
                analysis_results={"full_analysis": analysis_text},
                extracted_data=self._extract_presentation_data_from_analysis(analysis_text),
                timestamp=datetime.now().isoformat(),
                confidence_score=self._calculate_confidence_score(analysis_text)
            )

            self.processed_documents.append(doc)
            self._update_data_warehouse("presentations", doc.extracted_data)

            return doc

        except Exception as e:
            return BusinessDocument(
                file_path=document_path,
                document_type="presentation",
                analysis_results={"error": str(e)},
                extracted_data={},
                timestamp=datetime.now().isoformat(),
                confidence_score=0.0
            )

    def analyze_business_chart_or_graph(self, image_path: str, context: str = "") -> BusinessDocument:
        """Analyze business charts, graphs, and data visualizations"""

        chart_analysis_prompt = f"""
        Analyze this business chart/graph in detail:

        Context: {context if context else "No additional context provided"}

        1. CHART IDENTIFICATION:
        - Chart type (bar, line, pie, scatter, etc.)
        - Title and axis labels
        - Data source and time period
        - Units of measurement

        2. DATA EXTRACTION:
        - All data points and values
        - Trends and patterns
        - Highest and lowest values
        - Significant changes or anomalies

        3. BUSINESS INSIGHTS:
        - Performance interpretation
        - Growth or decline patterns
        - Seasonal variations
        - Comparative analysis between data series

        4. STRATEGIC IMPLICATIONS:
        - Business performance indicators
        - Areas of concern or opportunity
        - Resource allocation implications
        - Investment or operational decisions suggested

        5. FORECASTING AND TRENDS:
        - Projected future performance
        - Trend extrapolation
        - Cyclical patterns identified
        - External factor influences

        6. ACTIONABLE RECOMMENDATIONS:
        - Immediate actions required
        - Strategic pivots needed
        - Resource reallocation suggestions
        - Performance improvement opportunities

        7. QUALITY ASSESSMENT:
        - Data visualization effectiveness
        - Chart readability and clarity
        - Missing information or context
        - Recommendations for improvement

        Provide detailed analysis that enables data-driven business decisions.
        """

        try:
            with open(image_path, 'rb') as file:
                file_data = file.read()

            image_part = {
                'mime_type': self._get_image_mime_type(image_path),
                'data': file_data
            }

            response = self.model.generate_content(
                [chart_analysis_prompt, image_part],
                generation_config=self.precise_config
            )

            analysis_text = response.text if response.candidates else "Analysis failed"

            doc = BusinessDocument(
                file_path=image_path,
                document_type="chart_visualization",
                analysis_results={"full_analysis": analysis_text, "context": context},
                extracted_data=self._extract_chart_data_from_analysis(analysis_text),
                timestamp=datetime.now().isoformat(),
                confidence_score=self._calculate_confidence_score(analysis_text)
            )

            self.processed_documents.append(doc)
            self._update_data_warehouse("visualizations", doc.extracted_data)

            return doc

        except Exception as e:
            return BusinessDocument(
                file_path=image_path,
                document_type="chart_visualization",
                analysis_results={"error": str(e)},
                extracted_data={},
                timestamp=datetime.now().isoformat(),
                confidence_score=0.0
            )

    def generate_business_insights(self) -> List[BusinessInsight]:
        """Generate strategic business insights from all processed documents"""

        if not self.processed_documents:
            return []

        # Prepare data summary for analysis
        documents_summary = []
        for doc in self.processed_documents:
            documents_summary.append({
                "type": doc.document_type,
                "analysis": doc.analysis_results.get("full_analysis", "")[:500] + "...",
                "data": doc.extracted_data,
                "timestamp": doc.timestamp
            })

        insights_generation_prompt = f"""
        Based on the analysis of {len(self.processed_documents)} business documents, generate strategic business insights:

        Document Summary:
        {json.dumps(documents_summary, indent=2)}

        Generate insights in these categories:

        1. FINANCIAL PERFORMANCE INSIGHTS:
        - Revenue and expense trends
        - Profitability analysis
        - Cash flow patterns
        - Cost optimization opportunities

        2. OPERATIONAL EFFICIENCY INSIGHTS:
        - Process improvement opportunities
        - Resource utilization patterns
        - Bottlenecks and inefficiencies
        - Automation opportunities

        3. MARKET AND COMPETITIVE INSIGHTS:
        - Market position analysis
        - Competitive advantages/disadvantages
        - Growth opportunities
        - Market risks and challenges

        4. STRATEGIC RECOMMENDATIONS:
        - Short-term tactical actions (next 3 months)
        - Medium-term strategic moves (next 12 months)
        - Long-term vision alignment
        - Investment priorities

        5. RISK ASSESSMENT:
        - Financial risks identified
        - Operational risks
        - Market risks
        - Mitigation strategies

        For each insight, provide:
        - Clear title and description
        - Impact level (low, medium, high, critical)
        - Specific recommended actions
        - Supporting evidence from the documents

        Focus on actionable intelligence that enables better business decisions.
        """

        try:
            response = self.model.generate_content(
                insights_generation_prompt,
                generation_config=self.creative_config
            )

            insights_text = response.text if response.candidates else "Insights generation failed"

            # Parse insights from response (simplified extraction)
            insights = self._extract_insights_from_response(insights_text)

            self.business_insights.extend(insights)
            return insights

        except Exception as e:
            error_insight = BusinessInsight(
                insight_type="system_error",
                title="Insight Generation Error",
                description=f"Error generating insights: {str(e)}",
                impact_level="low",
                recommended_actions=["Review system configuration", "Check document quality"],
                supporting_data={},
                timestamp=datetime.now().isoformat()
            )
            return [error_insight]

    def create_executive_summary(self) -> str:
        """Create an executive summary based on all processed documents and insights"""

        if not self.processed_documents and not self.business_insights:
            return "No data available for executive summary."

        summary_prompt = f"""
        Create a comprehensive executive summary based on the business analysis performed:

        DOCUMENTS PROCESSED: {len(self.processed_documents)}
        INSIGHTS GENERATED: {len(self.business_insights)}

        Data Warehouse Summary:
        {json.dumps(self.extracted_data_warehouse, indent=2)}

        Create an executive summary that includes:

        1. EXECUTIVE OVERVIEW (2-3 sentences)
        - Key findings and overall business health
        - Most critical insights discovered

        2. KEY PERFORMANCE INDICATORS
        - Financial performance summary
        - Operational efficiency metrics
        - Market position assessment

        3. CRITICAL INSIGHTS
        - Top 3 most important insights
        - Impact on business operations
        - Strategic implications

        4. IMMEDIATE ACTION ITEMS
        - Urgent actions required (next 30 days)
        - Resource requirements
        - Expected outcomes

        5. STRATEGIC RECOMMENDATIONS
        - Medium-term initiatives (3-6 months)
        - Long-term strategic direction
        - Investment priorities

        6. RISK MITIGATION
        - Key risks identified
        - Mitigation strategies
        - Monitoring requirements

        Format as a professional executive summary suitable for C-level presentation.
        Keep it concise but comprehensive, focused on actionable intelligence.
        """

        try:
            response = self.model.generate_content(
                summary_prompt,
                generation_config=self.creative_config
            )

            return response.text if response.candidates else "Executive summary generation failed"

        except Exception as e:
            return f"Error creating executive summary: {str(e)}"

    def export_analysis_report(self, output_path: str = None) -> str:
        """Export complete analysis report to file"""

        if not output_path:
            output_path = f"business_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "documents_analyzed": len(self.processed_documents),
                "insights_generated": len(self.business_insights),
                "assistant_version": "1.0"
            },
            "processed_documents": [
                {
                    "file_path": doc.file_path,
                    "document_type": doc.document_type,
                    "analysis_results": doc.analysis_results,
                    "extracted_data": doc.extracted_data,
                    "timestamp": doc.timestamp,
                    "confidence_score": doc.confidence_score
                }
                for doc in self.processed_documents
            ],
            "business_insights": [
                {
                    "insight_type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "impact_level": insight.impact_level,
                    "recommended_actions": insight.recommended_actions,
                    "supporting_data": insight.supporting_data,
                    "timestamp": insight.timestamp
                }
                for insight in self.business_insights
            ],
            "data_warehouse": self.extracted_data_warehouse,
            "executive_summary": self.create_executive_summary()
        }

        try:
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            return output_path

        except Exception as e:
            return f"Error exporting report: {str(e)}"

    # Helper methods
    def _get_document_mime_type(self, file_path: str) -> str:
        """Get MIME type for document files"""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        return mime_types.get(extension, 'application/octet-stream')

    def _get_image_mime_type(self, file_path: str) -> str:
        """Get MIME type for image files"""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')

    def _calculate_confidence_score(self, analysis_text: str) -> float:
        """Calculate confidence score based on analysis completeness"""
        # Simple heuristic based on analysis length and completeness indicators
        confidence_indicators = [
            "specific" in analysis_text.lower(),
            "identified" in analysis_text.lower(),
            "analysis" in analysis_text.lower(),
            "recommendation" in analysis_text.lower(),
            len(analysis_text) > 500
        ]

        base_score = 0.5
        bonus_per_indicator = 0.1
        confidence = base_score + (sum(confidence_indicators) * bonus_per_indicator)

        return min(confidence, 1.0)

    def _extract_financial_data_from_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Extract structured financial data from analysis text"""
        # Simplified extraction - in production, use more sophisticated parsing
        return {
            "has_amounts": "$" in analysis_text,
            "has_dates": any(date_word in analysis_text.lower() for date_word in ["date", "due", "period"]),
            "document_length": len(analysis_text),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _extract_presentation_data_from_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Extract structured presentation data from analysis text"""
        return {
            "has_metrics": any(metric in analysis_text.lower() for metric in ["kpi", "metric", "performance"]),
            "has_strategy": any(strategy in analysis_text.lower() for strategy in ["strategy", "goal", "objective"]),
            "document_length": len(analysis_text),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _extract_chart_data_from_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Extract structured chart data from analysis text"""
        return {
            "has_data_points": any(data_word in analysis_text.lower() for data_word in ["data", "value", "trend"]),
            "has_insights": "insight" in analysis_text.lower(),
            "document_length": len(analysis_text),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _update_data_warehouse(self, category: str, data: Dict[str, Any]):
        """Update the data warehouse with new information"""
        if category not in self.extracted_data_warehouse:
            self.extracted_data_warehouse[category] = []

        self.extracted_data_warehouse[category].append({
            "data": data,
            "timestamp": datetime.now().isoformat()
        })

    def _extract_insights_from_response(self, insights_text: str) -> List[BusinessInsight]:
        """Extract structured insights from response text"""
        # Simplified extraction - create mock insights based on content
        insights = []

        insight_categories = [
            ("financial_performance", "Financial Performance Analysis"),
            ("operational_efficiency", "Operational Efficiency Review"),
            ("market_competitive", "Market & Competitive Position"),
            ("strategic_recommendations", "Strategic Recommendations")
        ]

        for category, title in insight_categories:
            if category.replace("_", " ") in insights_text.lower():
                insight = BusinessInsight(
                    insight_type=category,
                    title=title,
                    description=f"Analysis indicates important findings in {title.lower()}",
                    impact_level="medium",
                    recommended_actions=[
                        "Review detailed analysis",
                        "Discuss with stakeholders",
                        "Develop action plan"
                    ],
                    supporting_data={"source": "multimodal_analysis"},
                    timestamp=datetime.now().isoformat()
                )
                insights.append(insight)

        return insights


def create_sample_business_documents() -> List[str]:
    """Create sample business documents for testing"""

    documents = []

    # Sample Invoice
    invoice_content = """
    INVOICE #INV-2024-0156
    Date: March 15, 2024
    Due Date: April 14, 2024

    Bill To:
    Acme Corporation
    123 Business Ave
    Business City, BC 12345
    Tax ID: 123-45-6789

    From:
    Professional Services LLC
    456 Service Street
    Service City, SC 54321
    Tax ID: 987-65-4321

    Description                    Qty    Rate      Amount
    Strategic Consulting           40    $150.00   $6,000.00
    Market Research Analysis       1     $2,500.00 $2,500.00
    Implementation Support         20    $100.00   $2,000.00

    Subtotal:                                      $10,500.00
    Tax (8.5%):                                    $   892.50
    Total Amount Due:                              $11,392.50

    Payment Terms: Net 30
    Payment Methods: Check, ACH, Wire Transfer
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(invoice_content)
        documents.append(f.name)

    # Sample Business Report
    report_content = """
    QUARTERLY BUSINESS PERFORMANCE REPORT
    Q1 2024 Executive Summary

    FINANCIAL HIGHLIGHTS:
    - Revenue: $2.8M (+15% YoY growth)
    - Net Profit: $420K (15% profit margin)
    - Operating Expenses: $2.1M (-5% efficiency improvement)
    - Cash Flow: $380K positive

    KEY PERFORMANCE INDICATORS:
    - Customer Acquisition: 1,247 new customers (+22%)
    - Customer Retention Rate: 94% (+2% improvement)
    - Average Revenue Per User: $2,245
    - Monthly Recurring Revenue: $890K

    OPERATIONAL ACHIEVEMENTS:
    - Launched 3 new product features
    - Expanded team by 15 employees
    - Achieved 99.7% system uptime
    - Completed ISO certification

    MARKET POSITION:
    - Market share increased to 12% (+1.5%)
    - Customer satisfaction: 4.7/5.0
    - Net Promoter Score: 68 (Industry average: 45)

    STRATEGIC INITIATIVES:
    1. Product Development: AI-powered analytics platform
    2. Market Expansion: European market entry planned
    3. Partnership: Strategic alliance with TechCorp
    4. Investment: $5M Series B funding secured

    CHALLENGES AND RISKS:
    - Supply chain disruptions affecting delivery times
    - Increased competition in core market
    - Rising customer acquisition costs
    - Regulatory changes in key markets

    OUTLOOK Q2 2024:
    - Revenue target: $3.2M
    - New product launch scheduled
    - Team expansion to 125 employees
    - European market entry
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(report_content)
        documents.append(f.name)

    return documents


def run_multimodal_business_assistant_exercise():
    """Run the comprehensive multimodal business assistant exercise"""

    print("🏢 MULTIMODAL BUSINESS ASSISTANT EXERCISE")
    print("Comprehensive Business Intelligence Analysis")
    print("=" * 60)

    # Check API availability
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found. Please set it in your .env file")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return

    print("✅ Google API key found, initializing business assistant...")

    try:
        # Initialize the business assistant
        business_assistant = MultimodalBusinessAssistant()

        # Create sample business documents
        print("\n📄 Creating sample business documents for analysis...")
        sample_documents = create_sample_business_documents()

        print(f"Created {len(sample_documents)} sample documents")

        # Process each document
        print(f"\n🔍 ANALYZING BUSINESS DOCUMENTS")
        print("=" * 50)

        for i, doc_path in enumerate(sample_documents, 1):
            print(f"\n--- Document {i}: {Path(doc_path).name} ---")

            try:
                # Determine document type and process accordingly
                if "invoice" in Path(doc_path).name.lower() or "INV-" in open(doc_path).read():
                    print("📊 Processing financial document...")
                    result = business_assistant.analyze_financial_document(doc_path)
                    doc_type = "Financial Document"
                else:
                    print("📋 Processing business presentation/report...")
                    result = business_assistant.analyze_business_presentation(doc_path)
                    doc_type = "Business Report"

                print(f"✅ {doc_type} analysis completed")
                print(f"   Confidence Score: {result.confidence_score:.2f}")
                print(f"   Extracted Data Keys: {list(result.extracted_data.keys())}")

                # Show preview of analysis
                if result.analysis_results.get("full_analysis"):
                    preview = result.analysis_results["full_analysis"][:300] + "..."
                    print(f"   Analysis Preview: {preview}")

            except Exception as e:
                print(f"❌ Error processing document {i}: {e}")

        # Generate business insights
        print(f"\n🧠 GENERATING BUSINESS INSIGHTS")
        print("=" * 40)

        insights = business_assistant.generate_business_insights()

        print(f"✅ Generated {len(insights)} business insights")

        for i, insight in enumerate(insights, 1):
            print(f"\n--- Insight {i}: {insight.title} ---")
            print(f"Type: {insight.insight_type}")
            print(f"Impact Level: {insight.impact_level}")
            print(f"Description: {insight.description[:200]}...")
            print(f"Recommended Actions: {len(insight.recommended_actions)} items")

        # Create executive summary
        print(f"\n📋 CREATING EXECUTIVE SUMMARY")
        print("=" * 35)

        executive_summary = business_assistant.create_executive_summary()
        print("✅ Executive summary generated")
        print(f"\nExecutive Summary Preview:")
        print("-" * 30)
        print(executive_summary[:500] + "...")

        # Export comprehensive report
        print(f"\n💾 EXPORTING ANALYSIS REPORT")
        print("=" * 30)

        report_path = business_assistant.export_analysis_report()
        print(f"✅ Report exported to: {report_path}")

        # Display final statistics
        print(f"\n📈 ANALYSIS SUMMARY")
        print("=" * 25)
        print(f"Documents processed: {len(business_assistant.processed_documents)}")
        print(f"Insights generated: {len(business_assistant.business_insights)}")
        print(f"Data warehouse categories: {len(business_assistant.extracted_data_warehouse)}")

        avg_confidence = sum(doc.confidence_score for doc in business_assistant.processed_documents) / len(business_assistant.processed_documents) if business_assistant.processed_documents else 0
        print(f"Average confidence score: {avg_confidence:.2f}")

        print(f"\n🎯 EXERCISE OBJECTIVES ACHIEVED:")
        print("✅ Multimodal document analysis (text + images)")
        print("✅ Financial document processing and data extraction")
        print("✅ Business presentation analysis and insights")
        print("✅ Strategic insight generation from multiple sources")
        print("✅ Executive summary creation")
        print("✅ Comprehensive business intelligence reporting")

        # Clean up sample documents
        print(f"\n🧹 Cleaning up sample files...")
        for doc_path in sample_documents:
            try:
                os.unlink(doc_path)
                print(f"   Cleaned up: {Path(doc_path).name}")
            except Exception as e:
                print(f"   Warning: Could not clean up {Path(doc_path).name}: {e}")

        print(f"\n🎉 Multimodal Business Assistant Exercise Completed Successfully!")

    except Exception as e:
        print(f"❌ Exercise setup error: {e}")


if __name__ == "__main__":
    run_multimodal_business_assistant_exercise()


# EXERCISE EXTENSIONS:
#
# 1. Add real-time OCR for scanned documents
# 2. Implement voice-to-text analysis for meeting recordings
# 3. Create interactive dashboards for business insights
# 4. Add automated report generation and scheduling
# 5. Integrate with business systems (CRM, ERP, accounting)
# 6. Implement document classification and routing
# 7. Add multi-language document support
# 8. Create API endpoints for enterprise integration
# 9. Add advanced data visualization generation
# 10. Implement compliance checking for regulatory documents