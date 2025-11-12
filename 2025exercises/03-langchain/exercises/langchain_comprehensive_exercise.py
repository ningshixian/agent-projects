"""
LangChain Comprehensive Exercise
Build a complete business intelligence agent using LangChain

OBJECTIVE: Create a multi-functional business agent that can analyze data, generate reports, and provide insights
DIFFICULTY: Advanced
"""

import os
import json
import csv
import io
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationSummaryBufferMemory
from langchain.tools.base import BaseTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()


class BusinessDataManager:
    """Simulate business data storage and retrieval"""

    def __init__(self):
        # Sample business data
        self.sales_data = [
            {"month": "2024-01", "revenue": 45000, "customers": 150, "product": "Software"},
            {"month": "2024-02", "revenue": 52000, "customers": 180, "product": "Software"},
            {"month": "2024-03", "revenue": 48000, "customers": 165, "product": "Software"},
            {"month": "2024-04", "revenue": 61000, "customers": 210, "product": "Software"},
            {"month": "2024-05", "revenue": 58000, "customers": 195, "product": "Software"},
            {"month": "2024-06", "revenue": 67000, "customers": 230, "product": "Software"},
        ]

        self.employee_data = [
            {"name": "Alice Johnson", "department": "Engineering", "salary": 95000, "performance": 4.5},
            {"name": "Bob Smith", "department": "Sales", "salary": 75000, "performance": 4.2},
            {"name": "Carol Davis", "department": "Marketing", "salary": 70000, "performance": 4.7},
            {"name": "David Wilson", "department": "Engineering", "salary": 88000, "performance": 4.1},
            {"name": "Eve Brown", "department": "Sales", "salary": 72000, "performance": 4.6},
            {"name": "Frank Miller", "department": "HR", "salary": 65000, "performance": 4.3},
        ]

        self.expense_data = [
            {"category": "Office Rent", "amount": 8000, "month": "2024-06"},
            {"category": "Salaries", "amount": 45000, "month": "2024-06"},
            {"category": "Marketing", "amount": 12000, "month": "2024-06"},
            {"category": "Technology", "amount": 5000, "month": "2024-06"},
            {"category": "Utilities", "amount": 2000, "month": "2024-06"},
        ]

        self.customer_feedback = [
            {"rating": 5, "comment": "Excellent software, very user-friendly", "date": "2024-06-15"},
            {"rating": 4, "comment": "Good features but could use better documentation", "date": "2024-06-14"},
            {"rating": 5, "comment": "Outstanding customer support", "date": "2024-06-13"},
            {"rating": 3, "comment": "Software works but interface is confusing", "date": "2024-06-12"},
            {"rating": 4, "comment": "Great value for money", "date": "2024-06-11"},
            {"rating": 5, "comment": "Solved all our workflow problems", "date": "2024-06-10"},
        ]


class SalesAnalysisTool(BaseTool):
    """Tool for analyzing sales data"""

    name: str = "sales_analyzer"
    description: str = "Analyze sales data and generate insights. Input: 'summary', 'trends', or 'forecast'"

    def __init__(self):
        super().__init__()
        self.data_manager = BusinessDataManager()

    def _run(self, analysis_type: str) -> str:
        """Perform sales analysis"""
        try:
            sales_data = self.data_manager.sales_data
            analysis_type = analysis_type.lower().strip()

            if analysis_type == "summary":
                total_revenue = sum(item["revenue"] for item in sales_data)
                total_customers = sum(item["customers"] for item in sales_data)
                avg_revenue = total_revenue / len(sales_data)
                avg_customers = total_customers / len(sales_data)

                return f"""📊 SALES SUMMARY (6 months):

💰 Financial Metrics:
   • Total Revenue: ${total_revenue:,}
   • Average Monthly Revenue: ${avg_revenue:,.0f}
   • Revenue Growth: {((sales_data[-1]["revenue"] - sales_data[0]["revenue"]) / sales_data[0]["revenue"] * 100):+.1f}%

👥 Customer Metrics:
   • Total Customer Interactions: {total_customers:,}
   • Average Monthly Customers: {avg_customers:.0f}
   • Customer Growth: {((sales_data[-1]["customers"] - sales_data[0]["customers"]) / sales_data[0]["customers"] * 100):+.1f}%

📈 Performance:
   • Best Month: {max(sales_data, key=lambda x: x["revenue"])["month"]} (${max(sales_data, key=lambda x: x["revenue"])["revenue"]:,})
   • Lowest Month: {min(sales_data, key=lambda x: x["revenue"])["month"]} (${min(sales_data, key=lambda x: x["revenue"])["revenue"]:,})
                """

            elif analysis_type == "trends":
                # Calculate month-over-month growth
                trends = []
                for i in range(1, len(sales_data)):
                    prev_revenue = sales_data[i-1]["revenue"]
                    curr_revenue = sales_data[i]["revenue"]
                    growth = ((curr_revenue - prev_revenue) / prev_revenue) * 100
                    trends.append({
                        "month": sales_data[i]["month"],
                        "growth": growth,
                        "revenue": curr_revenue
                    })

                avg_growth = sum(t["growth"] for t in trends) / len(trends)

                trend_analysis = f"""📈 SALES TRENDS ANALYSIS:

🔄 Month-over-Month Growth:
"""
                for trend in trends:
                    trend_analysis += f"   • {trend['month']}: {trend['growth']:+.1f}% (${trend['revenue']:,})\n"

                trend_analysis += f"""
📊 Trend Summary:
   • Average Monthly Growth: {avg_growth:+.1f}%
   • Growth Consistency: {'Stable' if max(t['growth'] for t in trends) - min(t['growth'] for t in trends) < 20 else 'Volatile'}
   • Trend Direction: {'Positive' if avg_growth > 0 else 'Negative' if avg_growth < 0 else 'Flat'}

💡 Key Insights:
   • {'Strong growth momentum' if avg_growth > 5 else 'Moderate growth' if avg_growth > 0 else 'Declining performance'}
   • Revenue volatility: {max(t['growth'] for t in trends) - min(t['growth'] for t in trends):.1f}% range
                """
                return trend_analysis

            elif analysis_type == "forecast":
                # Simple linear forecast based on recent trend
                recent_months = sales_data[-3:]  # Last 3 months
                avg_recent_revenue = sum(item["revenue"] for item in recent_months) / len(recent_months)

                # Calculate growth rate
                if len(sales_data) >= 2:
                    latest_growth = ((sales_data[-1]["revenue"] - sales_data[-2]["revenue"]) / sales_data[-2]["revenue"]) * 100
                else:
                    latest_growth = 0

                # Project next 3 months
                forecast = []
                base_revenue = sales_data[-1]["revenue"]

                for i in range(1, 4):
                    projected_revenue = base_revenue * ((1 + latest_growth/100) ** i)
                    next_month = datetime.strptime(sales_data[-1]["month"], "%Y-%m") + timedelta(days=30*i)
                    forecast.append({
                        "month": next_month.strftime("%Y-%m"),
                        "projected_revenue": projected_revenue
                    })

                forecast_text = f"""🔮 SALES FORECAST (Next 3 Months):

📊 Projection Based On:
   • Recent Average Revenue: ${avg_recent_revenue:,.0f}
   • Latest Growth Rate: {latest_growth:+.1f}%
   • Historical Performance: 6 months of data

📈 Projected Revenue:
"""
                for proj in forecast:
                    forecast_text += f"   • {proj['month']}: ${proj['projected_revenue']:,.0f}\n"

                total_projected = sum(p["projected_revenue"] for p in forecast)
                forecast_text += f"""
💰 3-Month Projection Total: ${total_projected:,.0f}

⚠️  Forecast Assumptions:
   • Current growth trend continues
   • No major market disruptions
   • Seasonal patterns remain consistent
   • Product/service offerings unchanged

📊 Confidence Level: {'High' if abs(latest_growth) < 10 else 'Moderate' if abs(latest_growth) < 20 else 'Low'}
                """
                return forecast_text

            else:
                return "Please specify analysis type: 'summary', 'trends', or 'forecast'"

        except Exception as e:
            return f"Error in sales analysis: {str(e)}"

    async def _arun(self, analysis_type: str) -> str:
        return self._run(analysis_type)


class EmployeeAnalysisTool(BaseTool):
    """Tool for analyzing employee data"""

    name: str = "employee_analyzer"
    description: str = "Analyze employee data and HR metrics. Input: 'overview', 'performance', or 'compensation'"

    def __init__(self):
        super().__init__()
        self.data_manager = BusinessDataManager()

    def _run(self, analysis_type: str) -> str:
        """Perform employee analysis"""
        try:
            employee_data = self.data_manager.employee_data
            analysis_type = analysis_type.lower().strip()

            if analysis_type == "overview":
                total_employees = len(employee_data)
                departments = {}
                total_salary = sum(emp["salary"] for emp in employee_data)
                avg_salary = total_salary / total_employees
                avg_performance = sum(emp["performance"] for emp in employee_data) / total_employees

                for emp in employee_data:
                    dept = emp["department"]
                    if dept not in departments:
                        departments[dept] = 0
                    departments[dept] += 1

                overview = f"""👥 EMPLOYEE OVERVIEW:

📊 Workforce Summary:
   • Total Employees: {total_employees}
   • Average Salary: ${avg_salary:,.0f}
   • Average Performance Rating: {avg_performance:.2f}/5.0
   • Total Payroll: ${total_salary:,}/year

🏢 Department Breakdown:
"""
                for dept, count in departments.items():
                    percentage = (count / total_employees) * 100
                    overview += f"   • {dept}: {count} employees ({percentage:.1f}%)\n"

                return overview

            elif analysis_type == "performance":
                # Performance analysis
                high_performers = [emp for emp in employee_data if emp["performance"] >= 4.5]
                low_performers = [emp for emp in employee_data if emp["performance"] < 4.0]

                performance_by_dept = {}
                for emp in employee_data:
                    dept = emp["department"]
                    if dept not in performance_by_dept:
                        performance_by_dept[dept] = []
                    performance_by_dept[dept].append(emp["performance"])

                dept_avg_performance = {dept: sum(scores)/len(scores)
                                      for dept, scores in performance_by_dept.items()}

                performance_analysis = f"""⭐ PERFORMANCE ANALYSIS:

🏆 High Performers (4.5+):
"""
                for emp in high_performers:
                    performance_analysis += f"   • {emp['name']} ({emp['department']}): {emp['performance']}/5.0\n"

                performance_analysis += f"""
⚠️  Needs Improvement (<4.0):
"""
                if low_performers:
                    for emp in low_performers:
                        performance_analysis += f"   • {emp['name']} ({emp['department']}): {emp['performance']}/5.0\n"
                else:
                    performance_analysis += "   • No employees below 4.0 threshold\n"

                performance_analysis += f"""
🏢 Department Performance:
"""
                for dept, avg_perf in dept_avg_performance.items():
                    performance_analysis += f"   • {dept}: {avg_perf:.2f}/5.0\n"

                best_dept = max(dept_avg_performance.keys(), key=lambda x: dept_avg_performance[x])
                performance_analysis += f"""
💡 Insights:
   • Best Performing Department: {best_dept} ({dept_avg_performance[best_dept]:.2f}/5.0)
   • {len(high_performers)} high performers ({len(high_performers)/len(employee_data)*100:.1f}% of workforce)
   • Performance distribution: {'Well balanced' if len(low_performers) == 0 else 'Needs attention'}
                """
                return performance_analysis

            elif analysis_type == "compensation":
                # Compensation analysis
                salaries = [emp["salary"] for emp in employee_data]
                min_salary = min(salaries)
                max_salary = max(salaries)
                median_salary = sorted(salaries)[len(salaries)//2]

                comp_by_dept = {}
                for emp in employee_data:
                    dept = emp["department"]
                    if dept not in comp_by_dept:
                        comp_by_dept[dept] = []
                    comp_by_dept[dept].append(emp["salary"])

                dept_avg_salary = {dept: sum(sals)/len(sals)
                                 for dept, sals in comp_by_dept.items()}

                compensation_analysis = f"""💰 COMPENSATION ANALYSIS:

📊 Salary Statistics:
   • Minimum Salary: ${min_salary:,}
   • Maximum Salary: ${max_salary:,}
   • Median Salary: ${median_salary:,}
   • Salary Range: ${max_salary - min_salary:,}

🏢 Average Salary by Department:
"""
                for dept, avg_sal in dept_avg_salary.items():
                    compensation_analysis += f"   • {dept}: ${avg_sal:,.0f}\n"

                # Performance vs compensation analysis
                perf_comp_correlation = []
                for emp in employee_data:
                    perf_comp_correlation.append({
                        "name": emp["name"],
                        "performance": emp["performance"],
                        "salary": emp["salary"],
                        "value_ratio": emp["performance"] / (emp["salary"] / 1000)  # Performance per $1K salary
                    })

                best_value = max(perf_comp_correlation, key=lambda x: x["value_ratio"])

                compensation_analysis += f"""
⚡ Performance vs Compensation:
   • Best Value Employee: {best_value['name']} (Performance: {best_value['performance']}, Salary: ${best_value['salary']:,})
   • Pay Equity: {'Good distribution' if max_salary / min_salary < 1.5 else 'High variance' if max_salary / min_salary < 2 else 'Significant disparity'}

💡 Recommendations:
   • Salary range appears {'reasonable' if max_salary / min_salary < 1.5 else 'wide - review for equity'}
   • Consider performance-based adjustments for high performers
                """
                return compensation_analysis

            else:
                return "Please specify analysis type: 'overview', 'performance', or 'compensation'"

        except Exception as e:
            return f"Error in employee analysis: {str(e)}"

    async def _arun(self, analysis_type: str) -> str:
        return self._run(analysis_type)


class CustomerFeedbackTool(BaseTool):
    """Tool for analyzing customer feedback"""

    name: str = "feedback_analyzer"
    description: str = "Analyze customer feedback and satisfaction. Input: 'summary' or 'insights'"

    def __init__(self):
        super().__init__()
        self.data_manager = BusinessDataManager()

    def _run(self, analysis_type: str) -> str:
        """Perform customer feedback analysis"""
        try:
            feedback_data = self.data_manager.customer_feedback
            analysis_type = analysis_type.lower().strip()

            if analysis_type == "summary":
                total_feedback = len(feedback_data)
                ratings = [item["rating"] for item in feedback_data]
                avg_rating = sum(ratings) / len(ratings)

                rating_distribution = {}
                for rating in ratings:
                    if rating not in rating_distribution:
                        rating_distribution[rating] = 0
                    rating_distribution[rating] += 1

                summary = f"""⭐ CUSTOMER FEEDBACK SUMMARY:

📊 Overall Satisfaction:
   • Total Reviews: {total_feedback}
   • Average Rating: {avg_rating:.2f}/5.0
   • Customer Satisfaction: {avg_rating/5*100:.1f}%

📈 Rating Distribution:
"""
                for rating in sorted(rating_distribution.keys(), reverse=True):
                    count = rating_distribution[rating]
                    percentage = (count / total_feedback) * 100
                    stars = "⭐" * rating
                    summary += f"   • {stars} ({rating}): {count} reviews ({percentage:.1f}%)\n"

                # Sentiment analysis
                positive = len([r for r in ratings if r >= 4])
                negative = len([r for r in ratings if r <= 2])
                neutral = total_feedback - positive - negative

                summary += f"""
😊 Sentiment Breakdown:
   • Positive (4-5 stars): {positive} ({positive/total_feedback*100:.1f}%)
   • Neutral (3 stars): {neutral} ({neutral/total_feedback*100:.1f}%)
   • Negative (1-2 stars): {negative} ({negative/total_feedback*100:.1f}%)

💡 Overall Health: {'Excellent' if avg_rating >= 4.5 else 'Good' if avg_rating >= 4.0 else 'Fair' if avg_rating >= 3.5 else 'Needs Improvement'}
                """
                return summary

            elif analysis_type == "insights":
                # Text analysis of comments
                comments = [item["comment"].lower() for item in feedback_data]
                all_text = " ".join(comments)

                # Simple keyword analysis
                positive_keywords = ["excellent", "great", "good", "outstanding", "user-friendly", "solved"]
                negative_keywords = ["confusing", "poor", "bad", "difficult", "problem", "issue"]
                improvement_keywords = ["better", "improve", "documentation", "interface", "support"]

                positive_mentions = sum(1 for keyword in positive_keywords if keyword in all_text)
                negative_mentions = sum(1 for keyword in negative_keywords if keyword in all_text)
                improvement_mentions = sum(1 for keyword in improvement_keywords if keyword in all_text)

                # Recent trends
                recent_feedback = sorted(feedback_data, key=lambda x: x["date"], reverse=True)[:3]
                recent_avg = sum(item["rating"] for item in recent_feedback) / len(recent_feedback)

                insights = f"""🔍 CUSTOMER INSIGHTS:

💬 Comment Analysis:
   • Positive Mentions: {positive_mentions} occurrences
   • Negative Mentions: {negative_mentions} occurrences
   • Improvement Requests: {improvement_mentions} occurrences

📈 Recent Trend (Last 3 Reviews):
   • Recent Average Rating: {recent_avg:.2f}/5.0
   • Trend: {'Improving' if recent_avg > sum(item['rating'] for item in feedback_data)/len(feedback_data) else 'Stable' if abs(recent_avg - sum(item['rating'] for item in feedback_data)/len(feedback_data)) < 0.1 else 'Declining'}

🎯 Key Themes from Comments:
"""
                # Extract key themes
                themes = []
                for item in feedback_data:
                    comment = item["comment"].lower()
                    if "user-friendly" in comment or "interface" in comment:
                        themes.append("User Interface/Experience")
                    if "support" in comment or "customer" in comment:
                        themes.append("Customer Support")
                    if "features" in comment or "functionality" in comment:
                        themes.append("Product Features")
                    if "documentation" in comment:
                        themes.append("Documentation")

                theme_counts = {}
                for theme in themes:
                    if theme not in theme_counts:
                        theme_counts[theme] = 0
                    theme_counts[theme] += 1

                for theme, count in theme_counts.items():
                    insights += f"   • {theme}: {count} mentions\n"

                insights += f"""
🚨 Areas for Improvement:
   • {'Interface/UX' if 'confusing' in all_text else 'Documentation' if 'documentation' in all_text else 'General improvements based on feedback'}
   • Priority: {'High' if negative_mentions > positive_mentions else 'Medium' if improvement_mentions > 2 else 'Low'}

💡 Recommendations:
   • Focus on {'user experience improvements' if 'confusing' in all_text else 'documentation' if improvement_mentions > 2 else 'maintaining current quality'}
   • Customer satisfaction trend: {'Positive momentum' if recent_avg >= 4.0 else 'Monitor closely'}
                """
                return insights

            else:
                return "Please specify analysis type: 'summary' or 'insights'"

        except Exception as e:
            return f"Error in feedback analysis: {str(e)}"

    async def _arun(self, analysis_type: str) -> str:
        return self._run(analysis_type)


class BusinessReportGenerator(BaseTool):
    """Tool for generating comprehensive business reports"""

    name: str = "report_generator"
    description: str = "Generate comprehensive business reports. Input: 'executive' for executive summary, 'detailed' for full analysis"

    def __init__(self):
        super().__init__()
        self.data_manager = BusinessDataManager()
        # Initialize other tools to gather data
        self.sales_tool = SalesAnalysisTool()
        self.employee_tool = EmployeeAnalysisTool()
        self.feedback_tool = CustomerFeedbackTool()

    def _run(self, report_type: str) -> str:
        """Generate business reports"""
        try:
            report_type = report_type.lower().strip()

            if report_type == "executive":
                # Generate executive summary
                sales_summary = self.sales_tool._run("summary")
                employee_overview = self.employee_tool._run("overview")
                feedback_summary = self.feedback_tool._run("summary")

                # Extract key metrics
                total_revenue = sum(item["revenue"] for item in self.data_manager.sales_data)
                total_employees = len(self.data_manager.employee_data)
                avg_rating = sum(item["rating"] for item in self.data_manager.customer_feedback) / len(self.data_manager.customer_feedback)
                total_expenses = sum(item["amount"] for item in self.data_manager.expense_data)

                exec_summary = f"""
📊 EXECUTIVE BUSINESS SUMMARY
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

🎯 KEY PERFORMANCE INDICATORS:
   • Revenue (6 months): ${total_revenue:,}
   • Workforce Size: {total_employees} employees
   • Customer Satisfaction: {avg_rating:.1f}/5.0 ({avg_rating/5*100:.1f}%)
   • Monthly Expenses: ${total_expenses:,}
   • Profit Margin: {((total_revenue/6 - total_expenses) / (total_revenue/6) * 100):.1f}%

📈 BUSINESS HEALTH SCORE: {((avg_rating/5 * 0.3 + min(1, (total_revenue/6)/(total_expenses) * 0.4) + min(1, total_employees/10) * 0.3) * 100):.0f}/100

🔍 CRITICAL INSIGHTS:
   • Financial Performance: {'Strong' if total_revenue > 300000 else 'Good' if total_revenue > 200000 else 'Needs improvement'}
   • Customer Loyalty: {'Excellent' if avg_rating >= 4.5 else 'Good' if avg_rating >= 4.0 else 'Concerning'}
   • Operational Efficiency: {'Optimal' if total_revenue/6 > total_expenses*1.3 else 'Adequate' if total_revenue/6 > total_expenses else 'Critical'}

🎯 STRATEGIC RECOMMENDATIONS:
   • Revenue Growth: {'Maintain momentum' if total_revenue > 300000 else 'Accelerate acquisition'}
   • Customer Experience: {'Sustain excellence' if avg_rating >= 4.5 else 'Focus on improvements'}
   • Cost Management: {'Review efficiency' if total_revenue/6 < total_expenses*1.5 else 'Well controlled'}

⚠️  IMMEDIATE ACTION ITEMS:
   • {'Monitor customer satisfaction trends' if avg_rating < 4.0 else 'Scale successful practices'}
   • {'Review cost structure' if total_revenue/6 < total_expenses*1.3 else 'Invest in growth'}
   • {'Strengthen team performance' if len([e for e in self.data_manager.employee_data if e['performance'] < 4.0]) > 2 else 'Reward high performers'}

                """
                return exec_summary

            elif report_type == "detailed":
                # Generate detailed report
                detailed_report = f"""
📋 COMPREHENSIVE BUSINESS ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{'='*80}
1. SALES & REVENUE ANALYSIS
{'='*80}
{self.sales_tool._run("summary")}

{self.sales_tool._run("trends")}

{'='*80}
2. HUMAN RESOURCES ANALYSIS
{'='*80}
{self.employee_tool._run("overview")}

{self.employee_tool._run("performance")}

{self.employee_tool._run("compensation")}

{'='*80}
3. CUSTOMER SATISFACTION ANALYSIS
{'='*80}
{self.feedback_tool._run("summary")}

{self.feedback_tool._run("insights")}

{'='*80}
4. FINANCIAL OVERVIEW
{'='*80}
"""
                # Financial analysis
                expenses = self.data_manager.expense_data
                total_expenses = sum(item["amount"] for item in expenses)
                total_revenue = sum(item["revenue"] for item in self.data_manager.sales_data[-1:])  # Latest month
                profit_margin = ((total_revenue - total_expenses) / total_revenue * 100) if total_revenue > 0 else 0

                detailed_report += f"""
💰 Monthly Financial Summary:
   • Revenue (Latest Month): ${self.data_manager.sales_data[-1]['revenue']:,}
   • Total Expenses: ${total_expenses:,}
   • Net Profit: ${total_revenue - total_expenses:,}
   • Profit Margin: {profit_margin:.1f}%

📊 Expense Breakdown:
"""
                for expense in expenses:
                    percentage = (expense["amount"] / total_expenses) * 100
                    detailed_report += f"   • {expense['category']}: ${expense['amount']:,} ({percentage:.1f}%)\n"

                detailed_report += f"""
{'='*80}
5. STRATEGIC RECOMMENDATIONS & ACTION PLAN
{'='*80}

🎯 SHORT-TERM (Next 3 Months):
   • Maintain revenue growth trajectory
   • Address any performance issues identified
   • Continue monitoring customer satisfaction

📈 MEDIUM-TERM (Next 6-12 Months):
   • Scale successful customer acquisition strategies
   • Invest in high-performing team members
   • Optimize expense categories with low ROI

🚀 LONG-TERM (1+ Years):
   • Expand product/service offerings based on customer feedback
   • Build scalable operational processes
   • Develop succession planning for key roles

📊 RISK ASSESSMENT:
   • Revenue Risk: {'Low' if len(self.data_manager.sales_data) > 0 and self.data_manager.sales_data[-1]['revenue'] > self.data_manager.sales_data[0]['revenue'] else 'Medium'}
   • Customer Risk: {'Low' if sum(item['rating'] for item in self.data_manager.customer_feedback)/len(self.data_manager.customer_feedback) >= 4.0 else 'Medium'}
   • Operational Risk: {'Low' if len([e for e in self.data_manager.employee_data if e['performance'] >= 4.0]) > len(self.data_manager.employee_data)/2 else 'Medium'}

REPORT PREPARED BY: Business Intelligence Agent
NEXT REVIEW: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
                """
                return detailed_report

            else:
                return "Please specify report type: 'executive' or 'detailed'"

        except Exception as e:
            return f"Error generating report: {str(e)}"

    async def _arun(self, report_type: str) -> str:
        return self._run(report_type)


def create_business_intelligence_agent():
    """Create the main business intelligence agent"""

    print("🏢 Business Intelligence Agent Setup")
    print("="*40)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key required for this exercise")
        return None

    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.1,  # Low temperature for factual analysis
        model="gpt-3.5-turbo",
        max_tokens=1500
    )

    # Create business analysis tools
    tools = [
        SalesAnalysisTool(),
        EmployeeAnalysisTool(),
        CustomerFeedbackTool(),
        BusinessReportGenerator()
    ]

    # Advanced memory for maintaining context across analyses
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=1000
    )

    # Create the business intelligence agent
    bi_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3
    )

    print("✅ Business Intelligence Agent created with tools:")
    for tool in tools:
        print(f"   • {tool.name}: {tool.description}")

    return bi_agent


def run_business_scenarios(agent):
    """Run various business analysis scenarios"""

    print("\n📊 Business Analysis Scenarios")
    print("="*35)

    scenarios = [
        {
            "title": "Monthly Executive Brief",
            "query": "I need an executive summary of our business performance. Please provide a comprehensive overview of sales, employee performance, and customer satisfaction.",
            "expected_tools": ["sales_analyzer", "employee_analyzer", "feedback_analyzer"]
        },
        {
            "title": "Sales Performance Deep Dive",
            "query": "Our board wants a detailed analysis of sales trends and a forecast for the next quarter. What insights can you provide?",
            "expected_tools": ["sales_analyzer"]
        },
        {
            "title": "HR Performance Review",
            "query": "We're preparing for annual performance reviews. Can you analyze our employee performance and compensation data?",
            "expected_tools": ["employee_analyzer"]
        },
        {
            "title": "Customer Satisfaction Analysis",
            "query": "There are concerns about customer satisfaction. Please analyze our feedback data and provide insights on areas for improvement.",
            "expected_tools": ["feedback_analyzer"]
        },
        {
            "title": "Comprehensive Business Report",
            "query": "Generate a detailed business report for our quarterly board meeting. Include all key metrics and strategic recommendations.",
            "expected_tools": ["report_generator"]
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {scenario['title'].upper()}")
        print(f"{'='*60}")
        print(f"Business Request: {scenario['query']}")

        try:
            start_time = datetime.now()
            response = agent.run(scenario['query'])
            end_time = datetime.now()

            print(f"\n🤖 Business Intelligence Agent Response:")
            print(response)
            print(f"\n⏱️ Analysis completed in: {(end_time - start_time).total_seconds():.2f} seconds")

        except Exception as e:
            print(f"❌ Error in scenario {i}: {e}")

        if i < len(scenarios):
            input(f"\nPress Enter to continue to scenario {i+1}...")


def demonstrate_advanced_queries(agent):
    """Demonstrate complex multi-step business queries"""

    print("\n🧠 Advanced Business Queries")
    print("="*30)

    advanced_queries = [
        "Compare our sales performance with employee performance. Are there any correlations we should be aware of?",

        "Based on customer feedback and sales trends, what are the top 3 strategic priorities for the next quarter?",

        "If we maintain current growth trends, what will our revenue look like in 6 months? What staffing changes might we need?",

        "Create an action plan to improve our customer satisfaction score by 0.5 points. What specific areas should we focus on?"
    ]

    print("\nThese queries require the agent to:")
    print("• Use multiple tools in sequence")
    print("• Synthesize information across domains")
    print("• Provide strategic recommendations")
    print("• Consider business context and implications")

    for i, query in enumerate(advanced_queries, 1):
        print(f"\n--- Advanced Query {i} ---")
        print(f"Question: {query}")

        try:
            response = agent.run(query)
            print(f"\nBI Agent Analysis:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")

        if i < len(advanced_queries):
            input("\\nPress Enter for next query...")


def main():
    """Run the comprehensive LangChain business intelligence exercise"""

    print("🏢 LANGCHAIN BUSINESS INTELLIGENCE EXERCISE")
    print("="*50)

    print("""
This exercise demonstrates building a sophisticated business intelligence agent using LangChain.

📋 What you'll build:
   • Sales analysis and forecasting tools
   • Employee performance analytics
   • Customer satisfaction monitoring
   • Automated report generation
   • Multi-domain business insights

🎯 Skills demonstrated:
   • Custom tool development
   • Complex agent orchestration
   • Business domain expertise
   • Multi-step reasoning
   • Report generation
    """)

    # Create the business intelligence agent
    bi_agent = create_business_intelligence_agent()

    if not bi_agent:
        print("❌ Could not create business intelligence agent")
        return

    try:
        # Run business scenarios
        run_business_scenarios(bi_agent)

        # Advanced queries
        demonstrate_advanced_queries(bi_agent)

        print("\n🎉 Business Intelligence Exercise Completed!")

        print("""
💡 Key Learning Outcomes:
   1. Built sophisticated business analysis tools
   2. Created multi-domain agent workflows
   3. Implemented complex data analysis logic
   4. Generated executive-level insights
   5. Demonstrated real-world business applications

🚀 Next Steps:
   • Extend tools with real database connections
   • Add more sophisticated analytics
   • Implement automated report scheduling
   • Add data visualization capabilities
   • Scale for enterprise deployment
        """)

    except KeyboardInterrupt:
        print("\\n⏸️ Exercise interrupted by user")
    except Exception as e:
        print(f"❌ Exercise error: {e}")


if __name__ == "__main__":
    main()