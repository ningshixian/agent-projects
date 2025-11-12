"""
Parlant Basics - Official SDK Implementation
Conversational AI Framework for Building Customer-Facing Agents

Parlant 是一个旨在构建对话式 AI 代理的框架，其功能包括：
- 对话管理
- 工具集成到实际行动中
- 客户session会话管理
- Journey-based conversation flows
- 行为准则

Official Parlant SDK: https://github.com/emcie-co/parlant
"""

import os
import asyncio
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Official Parlant SDK imports
try:
    import parlant.sdk as p
    from parlant.sdk import ToolContext, ToolResult, NLPServices
    PARLANT_AVAILABLE = True
except ImportError:
    PARLANT_AVAILABLE = False
    print("⚠️  Parlant SDK not installed. Install with: pip install parlant")

load_dotenv()


# Define Parlant Tools using @p.tool decorator
if PARLANT_AVAILABLE:
    @p.tool
    async def get_account_balance(context: ToolContext) -> ToolResult:
        """Get the customer's current account balance."""
        # In production, this would query a real database
        # using context.customer_id for secure access

        # Simulate fetching balance
        mock_balances = {
            "demo_customer": 1250.75,
            "alice": 3420.50,
            "bob": 890.25
        }

        # Secure access using customer_id from context
        balance = mock_balances.get(context.customer_id, 0.00)

        return ToolResult(
            data=f"Your current account balance is ${balance:.2f}"
        )


    @p.tool
    async def schedule_appointment(
        context: ToolContext,
        date: str,
        time: str,
        service_type: str
    ) -> ToolResult:
        """Schedule an appointment for a customer."""
        # In production, this would integrate with a scheduling system

        appointment_id = f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return ToolResult(
            data={
                "appointment_id": appointment_id,
                "date": date,
                "time": time,
                "service_type": service_type,
                "status": "confirmed",
                "message": f"Appointment {appointment_id} scheduled for {date} at {time} for {service_type}"
            }
        )


    @p.tool
    async def get_order_status(
        context: ToolContext,
        order_id: str
    ) -> ToolResult:
        """Check the status of a customer order."""
        # Simulate order lookup
        mock_orders = {
            "ORD-001": {
                "status": "shipped",
                "tracking": "TRK123456",
                "estimated_delivery": "2025-10-05"
            },
            "ORD-002": {
                "status": "processing",
                "tracking": None,
                "estimated_delivery": "2025-10-08"
            }
        }

        order = mock_orders.get(order_id)

        if not order:
            return ToolResult(
                data=f"Order {order_id} not found. Please check the order number."
            )

        message = f"Order {order_id} is currently {order['status']}."
        if order['tracking']:
            message += f" Tracking number: {order['tracking']}."
        if order['estimated_delivery']:
            message += f" Estimated delivery: {order['estimated_delivery']}."

        return ToolResult(data=message)


    @p.tool
    async def transfer_to_human(context: ToolContext, reason: str) -> ToolResult:
        """Transfer the conversation to a human agent."""
        # Set agent to manual mode to stop automatic responses
        return ToolResult(
            data=f"Transferring you to a human agent. Reason: {reason}",
            control={"mode": "manual"}  # Stops automatic agent responses
        )


    @p.tool
    async def get_business_hours(context: ToolContext) -> ToolResult:
        """Get business hours information."""
        hours = {
            "Monday-Friday": "9:00 AM - 6:00 PM EST",
            "Saturday": "10:00 AM - 4:00 PM EST",
            "Sunday": "Closed"
        }

        return ToolResult(data=hours)


    @p.tool
    async def submit_feedback(
        context: ToolContext,
        rating: int,
        comments: Optional[str] = None
    ) -> ToolResult:
        """Submit customer feedback."""
        # Validate rating
        if rating < 1 or rating > 5:
            return ToolResult(
                data="Please provide a rating between 1 and 5 stars."
            )

        feedback_id = f"FB-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        message = f"Thank you for your {rating}-star rating!"
        if comments:
            message += f" Your feedback has been recorded (ID: {feedback_id})."

        return ToolResult(data=message)


async def demonstrate_basic_agent():
    """Demonstrate basic Parlant agent setup"""

    print("\n🤖 Basic Parlant Agent Demo")
    print("=" * 50)

    if not PARLANT_AVAILABLE:
        print("❌ Parlant SDK not available")
        return

    print("\n💡 Parlant Agent Concepts:")
    print("1. Server: Central management for agents and customers")
    print("2. Agent: AI entity with specific behavior and capabilities")
    print("3. Customer: User interacting with the agent")
    print("4. Session: Conversation instance between agent and customer")
    print("5. Tools: Functions the agent can call to perform actions")
    print("6. Guidelines: Rules governing agent behavior")

    print("\n📝 Agent Creation Example:")
    print("""
async with p.Server(nlp_service=NLPServices.openai) as server:
    # Create an agent
    agent = await server.create_agent(
        name="Customer Service Agent",
        description="Helpful, professional agent for customer support"
    )

    # Attach tools to agent
    await agent.attach_tool(
        condition="when customer asks about account balance",
        tool=get_account_balance
    )

    # Create a customer
    customer = await server.create_customer(
        name="Alice",
        metadata={"account_type": "premium"}
    )

    # Create a session
    session = await client.sessions.create(
        agent_id=agent.id,
        customer_id=customer.id
    )

    # Send a message
    await client.sessions.create_event(
        session_id=session.id,
        kind="message",
        source="customer",
        message="What's my account balance?"
    )

    # Agent automatically responds using tools!
    """)


async def demonstrate_tool_integration():
    """Demonstrate Parlant tool integration patterns"""

    print("\n🔧 Parlant Tool Integration Demo")
    print("=" * 50)

    print("\n💡 Tool Definition Patterns:")
    print("""
# 1. Basic tool
@p.tool
async def my_tool(context: ToolContext) -> ToolResult:
    return ToolResult(data="result")

# 2. Tool with parameters
@p.tool
async def search_products(
    context: ToolContext,
    query: str,
    category: Optional[str] = None
) -> ToolResult:
    results = await DB.search(query, category)
    return ToolResult(data=results)

# 3. Tool with parameter source control
from typing import Annotated

@p.tool
async def transfer_money(
    context: ToolContext,
    amount: Annotated[float, p.ToolParameterOptions(
        source="customer"  # Must come from customer, not agent inference
    )],
    recipient: Annotated[str, p.ToolParameterOptions(
        source="customer"
    )]
) -> ToolResult:
    # Process transfer...
    return ToolResult(data="Transfer completed")

# 4. Tool with access to server objects
@p.tool
async def get_customer_info(context: ToolContext) -> ToolResult:
    server = p.ToolContextAccessor(context).server

    # Access current agent
    agent = await server.get_agent(id=context.agent_id)

    # Access current customer
    customer = await server.get_customer(id=context.customer_id)

    return ToolResult(data=customer.metadata)
    """)


async def demonstrate_session_management():
    """Demonstrate session and event management"""

    print("\n💬 Session Management Demo")
    print("=" * 50)

    print("\n💡 Session Lifecycle:")
    print("""
# 1. Create session
session = await client.sessions.create(
    agent_id=agent_id,
    customer_id=customer_id,
    title="Support Session"
)

# 2. Send customer message
await client.sessions.create_event(
    session_id=session.id,
    kind="message",
    source="customer",
    message="I need help with my order"
)

# 3. Poll for agent responses
new_events = await client.sessions.list_events(
    session_id=session.id,
    min_offset=last_offset,
    wait_for_data=60  # Wait up to 60 seconds
)

# 4. Process events
for event in new_events:
    if event.kind == "message" and event.source == "ai_agent":
        print(f"Agent: {event.data['message']}")
    elif event.kind == "tool":
        # Tool was called
        tool_result = event.data['tool_calls'][0]['result']
        print(f"Tool result: {tool_result}")
    """)


async def demonstrate_guidelines():
    """Demonstrate agent guidelines and behavior control"""

    print("\n📋 Agent Guidelines Demo")
    print("=" * 50)

    print("\n💡 Guideline Patterns:")
    print("""
# 1. Condition-based guidelines
await agent.create_guideline(
    condition="when customer is frustrated or angry",
    action="be extra empathetic and apologize for inconvenience"
)

# 2. Tool-based guidelines
await agent.create_guideline(
    condition="when customer asks about account balance",
    action="use the get_account_balance tool",
    tools=[get_account_balance]
)

# 3. Journey guidelines (conversation flow)
await agent.create_guideline(
    condition="when customer wants to schedule appointment",
    action="ask for preferred date, time, and service type, then schedule"
)

# 4. Escalation guidelines
await agent.create_guideline(
    condition="when issue cannot be resolved or customer explicitly requests human",
    action="transfer to human agent",
    tools=[transfer_to_human]
)

# 5. Simpler tool attachment
await agent.attach_tool(
    condition="when customer needs order status",
    tool=get_order_status
)
    """)


async def demonstrate_customer_management():
    """Demonstrate customer entity management"""

    print("\n👤 Customer Management Demo")
    print("=" * 50)

    print("\n💡 Customer Patterns:")
    print("""
# 1. Create customer with metadata
customer = await server.create_customer(
    name="Alice Smith",
    metadata={
        "external_id": "USER-12345",
        "account_type": "premium",
        "location": "USA",
        "preferences": {"language": "en", "notifications": True}
    }
)

# 2. Access customer in tool
@p.tool
async def get_customer_location(context: ToolContext) -> ToolResult:
    server = p.ToolContextAccessor(context).server
    customer = await server.find_customer(id=context.customer_id)

    location = customer.metadata.get("location", "Unknown")
    return ToolResult(data=f"Customer location: {location}")

# 3. Secure data access
@p.tool
async def get_transactions(context: ToolContext) -> ToolResult:
    # SECURE: Uses customer_id from context (authenticated)
    transactions = await DB.get_transactions(context.customer_id)
    return ToolResult(data=transactions)

    # INSECURE: Do NOT do this
    # name = arguments.get("customer_name")
    # transactions = await DB.get_by_name(name)  # Security risk!
    """)


async def demonstrate_production_setup():
    """Demonstrate production-ready setup"""

    print("\n🚀 Production Setup Demo")
    print("=" * 50)

    print("\n💡 Complete Production Example:")
    print("""
import parlant.sdk as p
from parlant.sdk import NLPServices

async def setup_production_agent():
    # 1. Initialize server with NLP service
    async with p.Server(
        nlp_service=NLPServices.openai  # or vertex, anthropic
    ) as server:

        # 2. Create agent
        agent = await server.create_agent(
            name="Customer Support Agent",
            description="Professional, empathetic support agent"
        )

        # 3. Add guidelines
        await agent.create_guideline(
            condition="always be polite and professional",
            action="maintain friendly tone and show empathy"
        )

        await agent.create_guideline(
            condition="when customer provides account info",
            action="verify identity before accessing sensitive data"
        )

        # 4. Attach tools
        await agent.attach_tool(
            condition="when customer asks about balance",
            tool=get_account_balance
        )

        await agent.attach_tool(
            condition="when customer wants to schedule",
            tool=schedule_appointment
        )

        await agent.attach_tool(
            condition="when issue requires human intervention",
            tool=transfer_to_human
        )

        # 5. Create customers as needed
        customer = await server.create_customer(
            name="Customer Name",
            metadata={"source": "web", "account_id": "ACC123"}
        )

        # 6. Create session for interaction
        # (This would typically happen in your API endpoint)

        return agent, customer

# In your web application:
# - Customer visits chat interface
# - Create session with agent
# - Stream messages back and forth
# - Agent uses tools automatically based on guidelines
    """)


def main():
    """Main demonstration function"""
    print("🗣️ Parlant Basics - Official SDK Implementation")
    print("=" * 60)

    if not PARLANT_AVAILABLE:
        print("\n❌ Parlant SDK not installed")
        print("\nInstall Parlant:")
        print("  pip install parlant")
        print("\nOr check official docs:")
        print("  https://parlant.io")
        print("  https://github.com/emcie-co/parlant")
        return

    print("\n✅ Parlant SDK available")
    print("\n📚 Key Concepts:")
    print("• Server: Central management for agents and customers")
    print("• Agents: AI entities with specific behaviors")
    print("• Tools: Functions agents can call (@p.tool)")
    print("• Guidelines: Rules governing agent behavior")
    print("• Sessions: Conversation instances")
    print("• Customers: Users with metadata")

    # Run async demonstrations
    asyncio.run(demonstrate_basic_agent())
    asyncio.run(demonstrate_tool_integration())
    asyncio.run(demonstrate_session_management())
    asyncio.run(demonstrate_guidelines())
    asyncio.run(demonstrate_customer_management())
    asyncio.run(demonstrate_production_setup())

    print("\n" + "=" * 60)
    print("🎉 Parlant Basics Demo Complete!")

    print("\n🔑 Key Takeaways:")
    print("1. Use @p.tool decorator for all agent tools")
    print("2. Tools receive ToolContext with customer/agent IDs")
    print("3. Guidelines connect conditions to actions/tools")
    print("4. Sessions manage conversation state automatically")
    print("5. Customer metadata enables personalization")
    print("6. Secure tool access using context.customer_id")

    print("\n📖 Learn More:")
    print("• Official Docs: https://parlant.io")
    print("• GitHub: https://github.com/emcie-co/parlant")
    print("• Examples: docs/quickstart/examples.md")

    print("\n➡️  Next: Run dialogue_management.py for advanced patterns")


if __name__ == "__main__":
    main()