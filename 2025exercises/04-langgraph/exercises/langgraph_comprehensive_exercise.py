"""
LangGraph Comprehensive Exercise
Build a Complete E-commerce Order Processing System

This exercise challenges you to build a sophisticated multi-stage workflow
using LangGraph that processes e-commerce orders from validation to fulfillment.
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class OrderProcessingState(TypedDict):
    """State for the complete order processing workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Order Information
    order_id: str
    customer_info: Dict[str, Any]
    items: List[Dict[str, Any]]
    payment_info: Dict[str, Any]
    shipping_address: Dict[str, str]

    # Processing Status
    current_stage: str
    validation_results: Dict[str, bool]
    inventory_check: Dict[str, Any]
    payment_status: str
    shipping_details: Dict[str, Any]

    # Workflow Control
    processing_errors: List[str]
    requires_manual_review: bool
    estimated_delivery: str
    total_amount: float
    order_status: str


class EcommerceWorkflow:
    """Complete e-commerce order processing workflow with LangGraph"""

    def __init__(self, model_provider: str = "openai"):
        # Initialize LLM
        if model_provider == "openai" and os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        elif model_provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.1)
        else:
            raise ValueError(f"No API key found for {model_provider}")

        # Mock databases
        self.inventory_db = self._initialize_inventory()
        self.customer_db = self._initialize_customers()
        self.shipping_db = self._initialize_shipping()

        # Create the workflow
        self.workflow = self._build_workflow()

    def _initialize_inventory(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock inventory database"""
        return {
            "laptop_001": {"name": "Gaming Laptop Pro", "price": 1299.99, "stock": 5, "category": "electronics"},
            "phone_002": {"name": "Smartphone Elite", "price": 899.99, "stock": 12, "category": "electronics"},
            "headphones_003": {"name": "Wireless Headphones", "price": 199.99, "stock": 0, "category": "audio"},
            "book_004": {"name": "Python Programming Guide", "price": 49.99, "stock": 25, "category": "books"},
            "chair_005": {"name": "Ergonomic Office Chair", "price": 299.99, "stock": 3, "category": "furniture"},
            "monitor_006": {"name": "4K Gaming Monitor", "price": 549.99, "stock": 8, "category": "electronics"},
            "keyboard_007": {"name": "Mechanical Keyboard", "price": 129.99, "stock": 15, "category": "accessories"},
            "mouse_008": {"name": "Gaming Mouse", "price": 79.99, "stock": 20, "category": "accessories"}
        }

    def _initialize_customers(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock customer database"""
        return {
            "cust_001": {
                "name": "Alice Johnson", "email": "alice@email.com", "tier": "premium",
                "address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}
            },
            "cust_002": {
                "name": "Bob Smith", "email": "bob@email.com", "tier": "standard",
                "address": {"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90001"}
            },
            "cust_003": {
                "name": "Carol Davis", "email": "carol@email.com", "tier": "premium",
                "address": {"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}
            }
        }

    def _initialize_shipping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mock shipping options"""
        return {
            "standard": {"name": "Standard Shipping", "cost": 9.99, "days": 7},
            "express": {"name": "Express Shipping", "cost": 19.99, "days": 3},
            "overnight": {"name": "Overnight Shipping", "cost": 39.99, "days": 1}
        }

    def _build_workflow(self) -> StateGraph:
        """Build the complete order processing workflow"""
        workflow = StateGraph(OrderProcessingState)

        # Add workflow nodes
        workflow.add_node("validate_order", self._validate_order)
        workflow.add_node("check_inventory", self._check_inventory)
        workflow.add_node("process_payment", self._process_payment)
        workflow.add_node("calculate_shipping", self._calculate_shipping)
        workflow.add_node("generate_confirmation", self._generate_confirmation)
        workflow.add_node("handle_errors", self._handle_errors)

        # Set entry point
        workflow.set_entry_point("validate_order")

        # Add conditional routing
        workflow.add_conditional_edges(
            "validate_order",
            self._route_after_validation,
            {
                "check_inventory": "check_inventory",
                "handle_errors": "handle_errors",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "check_inventory",
            self._route_after_inventory,
            {
                "process_payment": "process_payment",
                "handle_errors": "handle_errors",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "process_payment",
            self._route_after_payment,
            {
                "calculate_shipping": "calculate_shipping",
                "handle_errors": "handle_errors",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "calculate_shipping",
            self._route_after_shipping,
            {
                "generate_confirmation": "generate_confirmation",
                "handle_errors": "handle_errors",
                END: END
            }
        )

        workflow.add_edge("generate_confirmation", END)
        workflow.add_edge("handle_errors", END)

        return workflow

    def _validate_order(self, state: OrderProcessingState) -> OrderProcessingState:
        """Validate order details and customer information"""
        print("🔍 Stage 1: Validating Order")

        customer_id = state.get("customer_info", {}).get("id", "")
        items = state.get("items", [])
        payment_info = state.get("payment_info", {})

        validation_results = {
            "customer_valid": customer_id in self.customer_db,
            "items_valid": len(items) > 0 and all("product_id" in item for item in items),
            "payment_valid": "card_number" in payment_info and "cvv" in payment_info,
            "address_valid": bool(state.get("shipping_address", {}))
        }

        # Use LLM to generate validation insights
        validation_prompt = f"""
        Analyze this order validation for potential issues:

        Customer ID: {customer_id}
        Items count: {len(items)}
        Payment method: {'Card' if payment_info else 'None'}
        Shipping address: {'Provided' if state.get('shipping_address') else 'Missing'}

        Validation results: {json.dumps(validation_results, indent=2)}

        Identify any concerns and provide recommendations for order processing.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=validation_prompt)])
            analysis_message = response
        except Exception as e:
            analysis_message = AIMessage(content=f"Validation analysis error: {e}")

        # Calculate total amount
        total_amount = sum(
            self.inventory_db.get(item["product_id"], {}).get("price", 0) * item.get("quantity", 1)
            for item in items
        )

        errors = []
        if not validation_results["customer_valid"]:
            errors.append("Invalid customer ID")
        if not validation_results["items_valid"]:
            errors.append("Invalid or missing items")
        if not validation_results["payment_valid"]:
            errors.append("Invalid payment information")
        if not validation_results["address_valid"]:
            errors.append("Missing shipping address")

        return {
            "messages": [analysis_message],
            "current_stage": "inventory_check" if not errors else "error_handling",
            "validation_results": validation_results,
            "processing_errors": errors,
            "total_amount": total_amount,
            "requires_manual_review": len(errors) > 2
        }

    def _check_inventory(self, state: OrderProcessingState) -> OrderProcessingState:
        """Check inventory availability for all items"""
        print("📦 Stage 2: Checking Inventory")

        items = state.get("items", [])
        inventory_results = {}
        availability_issues = []

        for item in items:
            product_id = item["product_id"]
            requested_qty = item.get("quantity", 1)

            if product_id in self.inventory_db:
                available_stock = self.inventory_db[product_id]["stock"]
                is_available = available_stock >= requested_qty

                inventory_results[product_id] = {
                    "product_name": self.inventory_db[product_id]["name"],
                    "requested": requested_qty,
                    "available": available_stock,
                    "in_stock": is_available
                }

                if not is_available:
                    availability_issues.append(
                        f"{self.inventory_db[product_id]['name']}: Need {requested_qty}, have {available_stock}"
                    )
            else:
                inventory_results[product_id] = {
                    "product_name": "Unknown Product",
                    "requested": requested_qty,
                    "available": 0,
                    "in_stock": False
                }
                availability_issues.append(f"Product {product_id} not found")

        # Use LLM to analyze inventory situation
        inventory_prompt = f"""
        Analyze this inventory check for an e-commerce order:

        Items requested: {len(items)}
        Products in stock: {sum(1 for r in inventory_results.values() if r['in_stock'])}
        Availability issues: {len(availability_issues)}

        Issues found: {availability_issues}

        Provide recommendations for handling this inventory situation, including:
        1. Whether to proceed with partial fulfillment
        2. Alternative products to suggest
        3. Customer communication strategy
        """

        try:
            response = self.llm.invoke([HumanMessage(content=inventory_prompt)])
            analysis_message = response
        except Exception as e:
            analysis_message = AIMessage(content=f"Inventory analysis error: {e}")

        errors = state.get("processing_errors", [])
        if availability_issues:
            errors.extend(availability_issues)

        return {
            "messages": [analysis_message],
            "current_stage": "payment_processing" if not availability_issues else "error_handling",
            "inventory_check": inventory_results,
            "processing_errors": errors,
            "requires_manual_review": state.get("requires_manual_review", False) or len(availability_issues) > 0
        }

    def _process_payment(self, state: OrderProcessingState) -> OrderProcessingState:
        """Process payment for the order"""
        print("💳 Stage 3: Processing Payment")

        payment_info = state.get("payment_info", {})
        total_amount = state.get("total_amount", 0)
        customer_info = state.get("customer_info", {})

        # Simulate payment processing
        payment_success = True
        payment_issues = []

        # Check for common payment issues
        if total_amount > 5000:  # High-value transaction
            payment_issues.append("High-value transaction requires additional verification")
            payment_success = False

        if payment_info.get("card_number", "").startswith("4000"):  # Mock declined card
            payment_issues.append("Payment declined by issuing bank")
            payment_success = False

        # Simulate fraud detection
        fraud_score = random.uniform(0, 100)
        if fraud_score > 85:
            payment_issues.append(f"High fraud risk score: {fraud_score:.1f}")
            payment_success = False

        payment_status = "approved" if payment_success else "declined"

        # Use LLM to analyze payment situation
        payment_prompt = f"""
        Analyze this payment processing result for an e-commerce order:

        Order total: ${total_amount:.2f}
        Customer tier: {customer_info.get('tier', 'standard')}
        Payment status: {payment_status}
        Issues found: {payment_issues}
        Fraud score: {fraud_score:.1f}

        Provide recommendations for:
        1. How to proceed with this payment status
        2. Customer communication approach
        3. Alternative payment options if needed
        4. Risk mitigation strategies
        """

        try:
            response = self.llm.invoke([HumanMessage(content=payment_prompt)])
            analysis_message = response
        except Exception as e:
            analysis_message = AIMessage(content=f"Payment analysis error: {e}")

        errors = state.get("processing_errors", [])
        if not payment_success:
            errors.extend(payment_issues)

        return {
            "messages": [analysis_message],
            "current_stage": "shipping_calculation" if payment_success else "error_handling",
            "payment_status": payment_status,
            "processing_errors": errors
        }

    def _calculate_shipping(self, state: OrderProcessingState) -> OrderProcessingState:
        """Calculate shipping options and costs"""
        print("🚚 Stage 4: Calculating Shipping")

        customer_info = state.get("customer_info", {})
        shipping_address = state.get("shipping_address", {})
        items = state.get("items", [])
        total_weight = sum(item.get("weight", 1.0) for item in items)  # Mock weight calculation

        # Get customer shipping preferences based on tier
        customer_tier = self.customer_db.get(customer_info.get("id", ""), {}).get("tier", "standard")

        # Calculate shipping options
        shipping_options = {}
        for option_id, option_data in self.shipping_db.items():
            base_cost = option_data["cost"]

            # Adjust cost based on weight and distance (simplified)
            weight_factor = 1 + (total_weight - 1) * 0.1
            adjusted_cost = base_cost * weight_factor

            # Premium customers get discounts
            if customer_tier == "premium":
                adjusted_cost *= 0.9

            # Calculate delivery date
            delivery_date = (datetime.now() + timedelta(days=option_data["days"])).strftime("%Y-%m-%d")

            shipping_options[option_id] = {
                "name": option_data["name"],
                "cost": round(adjusted_cost, 2),
                "delivery_date": delivery_date,
                "days": option_data["days"]
            }

        # Select recommended shipping option
        if customer_tier == "premium":
            recommended_option = "express"
        elif total_weight > 5:
            recommended_option = "standard"
        else:
            recommended_option = "express"

        selected_shipping = shipping_options[recommended_option]

        # Use LLM to analyze shipping options
        shipping_prompt = f"""
        Analyze shipping calculation for this order:

        Customer tier: {customer_tier}
        Shipping address: {shipping_address.get('city', 'Unknown')}, {shipping_address.get('state', 'Unknown')}
        Total weight: {total_weight:.1f} lbs
        Items count: {len(items)}

        Available shipping options:
        {json.dumps(shipping_options, indent=2)}

        Recommended option: {recommended_option}

        Provide analysis on:
        1. Appropriateness of recommended shipping option
        2. Cost optimization opportunities
        3. Delivery timeline considerations
        4. Customer satisfaction factors
        """

        try:
            response = self.llm.invoke([HumanMessage(content=shipping_prompt)])
            analysis_message = response
        except Exception as e:
            analysis_message = AIMessage(content=f"Shipping analysis error: {e}")

        return {
            "messages": [analysis_message],
            "current_stage": "confirmation_generation",
            "shipping_details": {
                "options": shipping_options,
                "selected": selected_shipping,
                "recommended_option": recommended_option
            },
            "estimated_delivery": selected_shipping["delivery_date"]
        }

    def _generate_confirmation(self, state: OrderProcessingState) -> OrderProcessingState:
        """Generate order confirmation and final summary"""
        print("✅ Stage 5: Generating Order Confirmation")

        order_id = state.get("order_id", "")
        customer_info = state.get("customer_info", {})
        items = state.get("items", [])
        total_amount = state.get("total_amount", 0)
        shipping_details = state.get("shipping_details", {})
        estimated_delivery = state.get("estimated_delivery", "")

        # Generate comprehensive order summary
        order_summary = {
            "order_id": order_id,
            "customer": customer_info,
            "items_count": len(items),
            "subtotal": total_amount,
            "shipping_cost": shipping_details.get("selected", {}).get("cost", 0),
            "total": total_amount + shipping_details.get("selected", {}).get("cost", 0),
            "estimated_delivery": estimated_delivery,
            "processing_timestamp": datetime.now().isoformat()
        }

        # Use LLM to generate customer-friendly confirmation message
        confirmation_prompt = f"""
        Generate a professional order confirmation message for this e-commerce order:

        Order Details:
        {json.dumps(order_summary, indent=2)}

        Include:
        1. Warm acknowledgment of the order
        2. Clear summary of items and costs
        3. Shipping and delivery information
        4. Next steps for the customer
        5. Contact information for support
        6. Professional, friendly tone

        Make it customer-focused and reassuring.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=confirmation_prompt)])
            confirmation_message = response
        except Exception as e:
            confirmation_message = AIMessage(content=f"Order confirmation generated with errors: {e}")

        return {
            "messages": [confirmation_message],
            "current_stage": "completed",
            "order_status": "confirmed",
            "final_total": order_summary["total"]
        }

    def _handle_errors(self, state: OrderProcessingState) -> OrderProcessingState:
        """Handle errors and determine resolution strategy"""
        print("⚠️  Error Handling Stage")

        errors = state.get("processing_errors", [])
        requires_manual_review = state.get("requires_manual_review", False)
        current_stage = state.get("current_stage", "unknown")

        # Use LLM to analyze errors and suggest resolution
        error_prompt = f"""
        Analyze these order processing errors and recommend solutions:

        Current stage: {current_stage}
        Errors encountered: {errors}
        Requires manual review: {requires_manual_review}

        For each error, provide:
        1. Severity level (low, medium, high, critical)
        2. Possible causes
        3. Recommended resolution steps
        4. Whether customer notification is needed
        5. Estimated resolution time

        Provide a clear action plan for resolving these issues.
        """

        try:
            response = self.llm.invoke([HumanMessage(content=error_prompt)])
            error_analysis = response
        except Exception as e:
            error_analysis = AIMessage(content=f"Error analysis failed: {e}")

        # Determine final status
        if len(errors) > 3 or requires_manual_review:
            final_status = "requires_manual_intervention"
        elif len(errors) > 1:
            final_status = "partially_processed"
        else:
            final_status = "processing_error"

        return {
            "messages": [error_analysis],
            "current_stage": "error_resolution",
            "order_status": final_status
        }

    # Routing methods
    def _route_after_validation(self, state: OrderProcessingState) -> str:
        """Route after order validation"""
        errors = state.get("processing_errors", [])
        if errors:
            return "handle_errors"
        return "check_inventory"

    def _route_after_inventory(self, state: OrderProcessingState) -> str:
        """Route after inventory check"""
        inventory_check = state.get("inventory_check", {})
        has_stock_issues = any(not result["in_stock"] for result in inventory_check.values())

        if has_stock_issues:
            return "handle_errors"
        return "process_payment"

    def _route_after_payment(self, state: OrderProcessingState) -> str:
        """Route after payment processing"""
        payment_status = state.get("payment_status", "")
        if payment_status != "approved":
            return "handle_errors"
        return "calculate_shipping"

    def _route_after_shipping(self, state: OrderProcessingState) -> str:
        """Route after shipping calculation"""
        shipping_details = state.get("shipping_details", {})
        if not shipping_details:
            return "handle_errors"
        return "generate_confirmation"

    def process_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete order through the workflow"""

        # Compile the workflow
        app = self.workflow.compile()

        # Create initial state from order data
        initial_state = {
            "messages": [HumanMessage(content=f"Processing order: {order_data.get('order_id', 'unknown')}")],
            "order_id": order_data.get("order_id", f"ord_{random.randint(1000, 9999)}"),
            "customer_info": order_data.get("customer_info", {}),
            "items": order_data.get("items", []),
            "payment_info": order_data.get("payment_info", {}),
            "shipping_address": order_data.get("shipping_address", {}),
            "current_stage": "validation",
            "validation_results": {},
            "inventory_check": {},
            "payment_status": "",
            "shipping_details": {},
            "processing_errors": [],
            "requires_manual_review": False,
            "estimated_delivery": "",
            "total_amount": 0.0,
            "order_status": "processing"
        }

        try:
            # Execute the workflow
            result = app.invoke(initial_state)
            return result

        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Workflow execution error: {e}")],
                "order_status": "system_error",
                "processing_errors": [str(e)]
            }


def create_sample_orders() -> List[Dict[str, Any]]:
    """Create sample orders for testing"""
    return [
        {
            "order_id": "ord_001",
            "customer_info": {"id": "cust_001", "name": "Alice Johnson"},
            "items": [
                {"product_id": "laptop_001", "quantity": 1},
                {"product_id": "mouse_008", "quantity": 1}
            ],
            "payment_info": {"card_number": "1234567890123456", "cvv": "123", "expiry": "12/25"},
            "shipping_address": {"street": "123 Main St", "city": "New York", "state": "NY", "zip": "10001"}
        },
        {
            "order_id": "ord_002",
            "customer_info": {"id": "cust_002", "name": "Bob Smith"},
            "items": [
                {"product_id": "headphones_003", "quantity": 2},  # Out of stock item
                {"product_id": "book_004", "quantity": 1}
            ],
            "payment_info": {"card_number": "4000000000000000", "cvv": "456", "expiry": "06/26"},  # Declined card
            "shipping_address": {"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90001"}
        },
        {
            "order_id": "ord_003",
            "customer_info": {"id": "cust_003", "name": "Carol Davis"},
            "items": [
                {"product_id": "chair_005", "quantity": 1},
                {"product_id": "monitor_006", "quantity": 2},
                {"product_id": "keyboard_007", "quantity": 1}
            ],
            "payment_info": {"card_number": "5555555555554444", "cvv": "789", "expiry": "09/27"},
            "shipping_address": {"street": "789 Pine Rd", "city": "Chicago", "state": "IL", "zip": "60601"}
        },
        {
            "order_id": "ord_004",
            "customer_info": {"id": "invalid_customer"},  # Invalid customer
            "items": [{"product_id": "laptop_001", "quantity": 1}],
            "payment_info": {},  # Missing payment info
            "shipping_address": {}  # Missing address
        }
    ]


def run_comprehensive_exercise():
    """Run the comprehensive LangGraph e-commerce exercise"""

    print("🛒 LANGGRAPH COMPREHENSIVE EXERCISE")
    print("E-commerce Order Processing System")
    print("=" * 50)

    # Check API availability
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ No API key found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        return

    print("✅ API key found, initializing e-commerce workflow system...")

    try:
        # Initialize the workflow system
        ecommerce_system = EcommerceWorkflow()

        # Get sample orders
        sample_orders = create_sample_orders()

        print(f"\n📦 Processing {len(sample_orders)} sample orders...\n")

        results_summary = []

        for i, order in enumerate(sample_orders, 1):
            print(f"{'='*60}")
            print(f"ORDER {i}: {order['order_id']}")
            print(f"Customer: {order['customer_info']['name']}")
            print(f"Items: {len(order['items'])} item(s)")
            print(f"{'='*60}")

            try:
                # Process the order through the complete workflow
                result = ecommerce_system.process_order(order)

                # Extract key results
                order_status = result.get("order_status", "unknown")
                current_stage = result.get("current_stage", "unknown")
                errors = result.get("processing_errors", [])
                total_amount = result.get("total_amount", 0)
                final_total = result.get("final_total", total_amount)

                print(f"\\n📊 ORDER PROCESSING RESULTS:")
                print(f"Status: {order_status}")
                print(f"Final Stage: {current_stage}")
                print(f"Order Total: ${total_amount:.2f}")
                if final_total != total_amount:
                    print(f"Final Total (with shipping): ${final_total:.2f}")

                if errors:
                    print(f"Errors: {len(errors)}")
                    for error in errors[:3]:  # Show first 3 errors
                        print(f"  • {error}")
                else:
                    print("✅ No processing errors")

                # Show final stage message
                if result.get("messages"):
                    final_message = result["messages"][-1].content
                    print(f"\\n💬 FINAL MESSAGE:")
                    print(f"{final_message[:200]}...")

                # Collect summary data
                results_summary.append({
                    "order_id": order["order_id"],
                    "status": order_status,
                    "errors": len(errors),
                    "total": final_total
                })

            except Exception as e:
                print(f"❌ Error processing order {order['order_id']}: {e}")
                results_summary.append({
                    "order_id": order["order_id"],
                    "status": "system_error",
                    "errors": 1,
                    "total": 0
                })

            print(f"\\n{'-'*60}\\n")

        # Display overall summary
        print("📈 PROCESSING SUMMARY")
        print("=" * 30)

        successful_orders = sum(1 for r in results_summary if r["status"] == "confirmed")
        error_orders = sum(1 for r in results_summary if r["status"] in ["system_error", "requires_manual_intervention"])
        partial_orders = len(results_summary) - successful_orders - error_orders

        total_revenue = sum(r["total"] for r in results_summary if r["status"] == "confirmed")

        print(f"Total orders processed: {len(results_summary)}")
        print(f"Successfully completed: {successful_orders}")
        print(f"Requiring manual intervention: {error_orders}")
        print(f"Partially processed: {partial_orders}")
        print(f"Total revenue from successful orders: ${total_revenue:.2f}")

        print(f"\\nSuccess rate: {(successful_orders/len(results_summary)*100):.1f}%")

        print("\\n🎯 EXERCISE OBJECTIVES ACHIEVED:")
        print("✅ Multi-stage workflow with conditional routing")
        print("✅ Complex state management across processing phases")
        print("✅ Error handling and manual review triggering")
        print("✅ Integration with mock external systems (inventory, payment)")
        print("✅ LLM-powered analysis and decision making")
        print("✅ Comprehensive order processing pipeline")

    except Exception as e:
        print(f"❌ Exercise setup error: {e}")


if __name__ == "__main__":
    run_comprehensive_exercise()


# EXERCISE EXTENSIONS:
#
# 1. Add real database integration instead of mock data
# 2. Implement actual payment processing with Stripe/PayPal APIs
# 3. Add email notifications at each stage
# 4. Create a web interface for order status tracking
# 5. Implement inventory reservation during processing
# 6. Add support for partial shipments and backorders
# 7. Create analytics dashboard for order processing metrics
# 8. Add A/B testing for different workflow variations
# 9. Implement order modification and cancellation workflows
# 10. Add integration with shipping carrier APIs for real tracking