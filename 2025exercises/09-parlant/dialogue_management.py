"""
Dialogue Management - Advanced Conversation Flow Control

This module demonstrates sophisticated dialogue management techniques:
- State-based conversation flows
- Dynamic conversation routing
- Context-aware responses
- Multi-path dialogue trees
- Conversation recovery and error handling
- Advanced memory management

Building on parlant_basics.py, this shows how to create complex,
goal-oriented conversation systems.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import openai
import anthropic
from dotenv import load_dotenv

# Import base classes from parlant_basics
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parlant_basics import ConversationContext, ConversationTurn, IntentType, ConversationalAgent

# Load environment variables
load_dotenv()


class DialogueState(Enum):
    """Dialogue flow states"""
    INITIAL = "initial"
    GREETING = "greeting"
    INFORMATION_GATHERING = "info_gathering"
    PROCESSING_REQUEST = "processing_request"
    CONFIRMING_ACTION = "confirming_action"
    COMPLETING_TASK = "completing_task"
    CLARIFICATION_NEEDED = "clarification_needed"
    ERROR_RECOVERY = "error_recovery"
    ENDING = "ending"
    ESCALATION = "escalation"


class DialogueAction(Enum):
    """Actions that can be taken in dialogue flow"""
    ASK_QUESTION = "ask_question"
    PROVIDE_INFO = "provide_info"
    REQUEST_CONFIRMATION = "request_confirmation"
    EXECUTE_ACTION = "execute_action"
    ESCALATE = "escalate"
    END_CONVERSATION = "end_conversation"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"
    RECOVER_FROM_ERROR = "recover_from_error"


@dataclass
class DialogueNode:
    """A node in the dialogue flow"""
    state: DialogueState
    prompt_template: str
    required_info: List[str] = field(default_factory=list)
    next_states: Dict[str, DialogueState] = field(default_factory=dict)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)
    max_retries: int = 3
    retry_count: int = 0


class AdvancedDialogueManager(ConversationalAgent):
    """Advanced dialogue manager with state-based conversation flows"""

    def __init__(self, agent_name: str = "Advanced Assistant", flow_config: Dict = None):
        super().__init__(agent_name)
        self.dialogue_flows = flow_config or self._create_default_flow()
        self.current_nodes: Dict[str, DialogueNode] = {}  # session_id -> current_node
        self.collected_info: Dict[str, Dict[str, Any]] = {}  # session_id -> collected_data

    def _create_default_flow(self) -> Dict[DialogueState, DialogueNode]:
        """Create a default dialogue flow"""
        return {
            DialogueState.INITIAL: DialogueNode(
                state=DialogueState.INITIAL,
                prompt_template="Hello! I'm {agent_name}. How can I help you today?",
                next_states={
                    "greeting": DialogueState.GREETING,
                    "question": DialogueState.INFORMATION_GATHERING,
                    "request": DialogueState.INFORMATION_GATHERING
                }
            ),
            DialogueState.GREETING: DialogueNode(
                state=DialogueState.GREETING,
                prompt_template="Nice to meet you! What would you like to accomplish today?",
                next_states={
                    "question": DialogueState.INFORMATION_GATHERING,
                    "request": DialogueState.INFORMATION_GATHERING,
                    "goodbye": DialogueState.ENDING
                }
            ),
            DialogueState.INFORMATION_GATHERING: DialogueNode(
                state=DialogueState.INFORMATION_GATHERING,
                prompt_template="I'd be happy to help. Could you provide more details about {topic}?",
                required_info=["user_request", "context_details"],
                next_states={
                    "complete_info": DialogueState.PROCESSING_REQUEST,
                    "need_clarification": DialogueState.CLARIFICATION_NEEDED,
                    "insufficient_info": DialogueState.INFORMATION_GATHERING
                }
            ),
            DialogueState.PROCESSING_REQUEST: DialogueNode(
                state=DialogueState.PROCESSING_REQUEST,
                prompt_template="Let me process that for you. Based on what you've told me about {context}, I understand you want to {action}.",
                next_states={
                    "needs_confirmation": DialogueState.CONFIRMING_ACTION,
                    "can_complete": DialogueState.COMPLETING_TASK,
                    "need_more_info": DialogueState.INFORMATION_GATHERING
                }
            ),
            DialogueState.CONFIRMING_ACTION: DialogueNode(
                state=DialogueState.CONFIRMING_ACTION,
                prompt_template="To confirm, you'd like me to {action} with the following details: {details}. Is this correct?",
                next_states={
                    "confirmed": DialogueState.COMPLETING_TASK,
                    "needs_changes": DialogueState.INFORMATION_GATHERING,
                    "cancel": DialogueState.ENDING
                }
            ),
            DialogueState.COMPLETING_TASK: DialogueNode(
                state=DialogueState.COMPLETING_TASK,
                prompt_template="Perfect! I've {completed_action}. Is there anything else I can help you with?",
                next_states={
                    "new_request": DialogueState.INFORMATION_GATHERING,
                    "satisfied": DialogueState.ENDING
                }
            ),
            DialogueState.CLARIFICATION_NEEDED: DialogueNode(
                state=DialogueState.CLARIFICATION_NEEDED,
                prompt_template="I want to make sure I understand correctly. When you mention {unclear_part}, do you mean {interpretation}?",
                next_states={
                    "clarified": DialogueState.PROCESSING_REQUEST,
                    "still_unclear": DialogueState.CLARIFICATION_NEEDED,
                    "start_over": DialogueState.INFORMATION_GATHERING
                },
                max_retries=2
            ),
            DialogueState.ERROR_RECOVERY: DialogueNode(
                state=DialogueState.ERROR_RECOVERY,
                prompt_template="I apologize for the confusion. Let me try a different approach. {recovery_message}",
                next_states={
                    "recovered": DialogueState.INFORMATION_GATHERING,
                    "escalate": DialogueState.ESCALATION
                }
            ),
            DialogueState.ESCALATION: DialogueNode(
                state=DialogueState.ESCALATION,
                prompt_template="I'm having trouble helping with this request. Let me connect you with a specialist who can better assist you.",
                next_states={
                    "escalated": DialogueState.ENDING
                }
            ),
            DialogueState.ENDING: DialogueNode(
                state=DialogueState.ENDING,
                prompt_template="Thank you for chatting with me today! Have a great day!",
                next_states={}
            )
        }

    def get_current_node(self, session_id: str) -> DialogueNode:
        """Get current dialogue node for session"""
        if session_id not in self.current_nodes:
            self.current_nodes[session_id] = self.dialogue_flows[DialogueState.INITIAL]
        return self.current_nodes[session_id]

    def get_collected_info(self, session_id: str) -> Dict[str, Any]:
        """Get collected information for session"""
        if session_id not in self.collected_info:
            self.collected_info[session_id] = {}
        return self.collected_info[session_id]

    def determine_next_state(self, current_node: DialogueNode, context: ConversationContext,
                           user_message: str, intent: IntentType) -> DialogueState:
        """Determine next dialogue state based on current context"""
        collected_info = self.get_collected_info(context.session_id)

        # State transition logic based on current state and user input
        current_state = current_node.state

        if current_state == DialogueState.INITIAL:
            if intent == IntentType.GREETING:
                return DialogueState.GREETING
            elif intent in [IntentType.QUESTION, IntentType.REQUEST]:
                return DialogueState.INFORMATION_GATHERING
            else:
                return DialogueState.INFORMATION_GATHERING

        elif current_state == DialogueState.GREETING:
            if intent == IntentType.GOODBYE:
                return DialogueState.ENDING
            else:
                return DialogueState.INFORMATION_GATHERING

        elif current_state == DialogueState.INFORMATION_GATHERING:
            # Check if we have enough information
            required_info = current_node.required_info
            has_required_info = all(key in collected_info for key in required_info)

            if has_required_info:
                return DialogueState.PROCESSING_REQUEST
            else:
                # Check if user provided new information
                if self._extract_new_information(user_message, collected_info):
                    # Recheck if we now have enough info
                    has_required_info = all(key in collected_info for key in required_info)
                    if has_required_info:
                        return DialogueState.PROCESSING_REQUEST
                return DialogueState.INFORMATION_GATHERING

        elif current_state == DialogueState.PROCESSING_REQUEST:
            # Determine if we need confirmation or can complete directly
            request_complexity = self._assess_request_complexity(collected_info)
            if request_complexity == "high":
                return DialogueState.CONFIRMING_ACTION
            else:
                return DialogueState.COMPLETING_TASK

        elif current_state == DialogueState.CONFIRMING_ACTION:
            if intent == IntentType.CONFIRMATION or "yes" in user_message.lower():
                return DialogueState.COMPLETING_TASK
            elif "no" in user_message.lower() or intent == IntentType.CLARIFICATION:
                return DialogueState.INFORMATION_GATHERING
            else:
                return DialogueState.CLARIFICATION_NEEDED

        elif current_state == DialogueState.COMPLETING_TASK:
            if intent in [IntentType.QUESTION, IntentType.REQUEST]:
                return DialogueState.INFORMATION_GATHERING
            elif intent == IntentType.GOODBYE:
                return DialogueState.ENDING
            else:
                return DialogueState.ENDING

        elif current_state == DialogueState.CLARIFICATION_NEEDED:
            current_node.retry_count += 1
            if current_node.retry_count >= current_node.max_retries:
                return DialogueState.ERROR_RECOVERY
            else:
                return DialogueState.PROCESSING_REQUEST

        elif current_state == DialogueState.ERROR_RECOVERY:
            if self._can_recover():
                return DialogueState.INFORMATION_GATHERING
            else:
                return DialogueState.ESCALATION

        # Default fallback
        return current_state

    def _extract_new_information(self, user_message: str, collected_info: Dict[str, Any]) -> bool:
        """Extract and store new information from user message"""
        # Simple information extraction (in production, use NLP/NER)
        message_lower = user_message.lower()
        extracted_any = False

        # Extract common request types
        if not collected_info.get("user_request"):
            if any(word in message_lower for word in ["book", "schedule", "reserve"]):
                collected_info["user_request"] = "booking"
                extracted_any = True
            elif any(word in message_lower for word in ["cancel", "cancellation"]):
                collected_info["user_request"] = "cancellation"
                extracted_any = True
            elif any(word in message_lower for word in ["change", "modify", "update"]):
                collected_info["user_request"] = "modification"
                extracted_any = True

        # Extract context details
        if not collected_info.get("context_details"):
            # Look for dates, times, names, etc.
            import re

            # Date patterns
            date_patterns = [r'\b\d{1,2}/\d{1,2}/\d{4}\b', r'\b\d{1,2}-\d{1,2}-\d{4}\b']
            for pattern in date_patterns:
                dates = re.findall(pattern, user_message)
                if dates:
                    collected_info["date"] = dates[0]
                    extracted_any = True

            # Time patterns
            time_pattern = r'\b\d{1,2}:\d{2}\s*(am|pm)?\b'
            times = re.findall(time_pattern, user_message, re.IGNORECASE)
            if times:
                collected_info["time"] = times[0][0] if isinstance(times[0], tuple) else times[0]
                extracted_any = True

            # If we found date or time, consider context details extracted
            if collected_info.get("date") or collected_info.get("time"):
                collected_info["context_details"] = "temporal_info_provided"
                extracted_any = True

        return extracted_any

    def _assess_request_complexity(self, collected_info: Dict[str, Any]) -> str:
        """Assess the complexity of the user's request"""
        request_type = collected_info.get("user_request", "unknown")

        # High complexity requests need confirmation
        high_complexity = ["cancellation", "modification", "refund"]
        if request_type in high_complexity:
            return "high"

        # Medium complexity
        medium_complexity = ["booking", "scheduling"]
        if request_type in medium_complexity:
            return "medium"

        return "low"

    def _can_recover(self) -> bool:
        """Determine if we can recover from current error state"""
        # Simple recovery logic - in production, this would be more sophisticated
        return True

    def generate_contextualized_response(self, node: DialogueNode, context: ConversationContext,
                                       collected_info: Dict[str, Any]) -> str:
        """Generate a response based on current dialogue node and context"""
        template = node.prompt_template

        # Fill in template variables
        replacements = {
            "agent_name": self.agent_name,
            "topic": collected_info.get("user_request", "your request"),
            "context": ", ".join([f"{k}: {v}" for k, v in collected_info.items()]),
            "action": collected_info.get("user_request", "help you"),
            "details": json.dumps(collected_info, indent=2),
            "completed_action": f"completed your {collected_info.get('user_request', 'request')}",
            "unclear_part": "that part of your request",
            "interpretation": "the most common meaning",
            "recovery_message": "Could you please rephrase your request?"
        }

        # Replace template variables
        response = template
        for key, value in replacements.items():
            response = response.replace(f"{{{key}}}", str(value))

        return response

    def process_conversation_turn(self, user_id: str, session_id: str, message: str,
                                use_anthropic: bool = False) -> Dict[str, Any]:
        """Process conversation turn with advanced dialogue management"""
        # Get context and current node
        context = self.get_or_create_context(user_id, session_id)
        current_node = self.get_current_node(session_id)
        collected_info = self.get_collected_info(session_id)

        # Update last interaction time
        context.last_interaction = time.time()

        # Classify intent and extract entities
        intent, confidence = self.classify_intent(message)
        entities = self.extract_entities(message)

        # Extract new information
        self._extract_new_information(message, collected_info)

        # Add to conversation history
        turn_data = {
            'timestamp': time.time(),
            'speaker': 'user',
            'message': message,
            'intent': intent.value if intent else None,
            'confidence': confidence,
            'entities': entities,
            'dialogue_state': current_node.state.value
        }
        context.conversation_history.append(turn_data)

        # Determine next state
        next_state = self.determine_next_state(current_node, context, message, intent)
        next_node = self.dialogue_flows[next_state]

        # Update current node
        self.current_nodes[session_id] = next_node

        # Generate contextualized response
        if next_state == DialogueState.ENDING and intent != IntentType.GOODBYE:
            # Use LLM for more natural ending
            if use_anthropic and self.anthropic_client:
                response_text = self.generate_response_with_anthropic(context, message)
            else:
                response_text = self.generate_response_with_openai(context, message)
        else:
            # Use template-based response
            response_text = self.generate_contextualized_response(next_node, context, collected_info)

        # Add agent response to history
        agent_turn = {
            'timestamp': time.time(),
            'speaker': 'agent',
            'message': response_text,
            'intent': None,
            'confidence': 1.0,
            'entities': {},
            'dialogue_state': next_state.value
        }
        context.conversation_history.append(agent_turn)

        return {
            'response': response_text,
            'intent': intent.value,
            'confidence': confidence,
            'entities': entities,
            'dialogue_state': next_state.value,
            'collected_info': collected_info,
            'session_id': session_id,
            'conversation_complete': next_state == DialogueState.ENDING
        }


class BookingDialogueManager(AdvancedDialogueManager):
    """Specialized dialogue manager for booking/reservation systems"""

    def __init__(self):
        booking_flow = self._create_booking_flow()
        super().__init__("Booking Assistant", booking_flow)

    def _create_booking_flow(self) -> Dict[DialogueState, DialogueNode]:
        """Create specialized booking dialogue flow"""
        return {
            DialogueState.INITIAL: DialogueNode(
                state=DialogueState.INITIAL,
                prompt_template="Welcome to our booking system! I can help you make, modify, or cancel reservations. What would you like to do?",
                next_states={"any": DialogueState.INFORMATION_GATHERING}
            ),
            DialogueState.INFORMATION_GATHERING: DialogueNode(
                state=DialogueState.INFORMATION_GATHERING,
                prompt_template="I'd be happy to help with your {user_request}. I'll need a few details: What date and time work best for you?",
                required_info=["user_request", "date", "time"],
                next_states={"complete": DialogueState.PROCESSING_REQUEST}
            ),
            DialogueState.PROCESSING_REQUEST: DialogueNode(
                state=DialogueState.PROCESSING_REQUEST,
                prompt_template="Let me check availability for {date} at {time} for your {user_request}.",
                next_states={"available": DialogueState.CONFIRMING_ACTION}
            ),
            DialogueState.CONFIRMING_ACTION: DialogueNode(
                state=DialogueState.CONFIRMING_ACTION,
                prompt_template="Great! I found availability on {date} at {time}. Shall I confirm this booking for you?",
                next_states={
                    "confirmed": DialogueState.COMPLETING_TASK,
                    "modify": DialogueState.INFORMATION_GATHERING
                }
            ),
            DialogueState.COMPLETING_TASK: DialogueNode(
                state=DialogueState.COMPLETING_TASK,
                prompt_template="Perfect! Your booking is confirmed for {date} at {time}. You'll receive a confirmation email shortly. Is there anything else I can help you with?",
                next_states={
                    "done": DialogueState.ENDING,
                    "new_booking": DialogueState.INFORMATION_GATHERING
                }
            ),
            DialogueState.ENDING: DialogueNode(
                state=DialogueState.ENDING,
                prompt_template="Thank you for using our booking system! Have a wonderful day!",
                next_states={}
            )
        }


# Demonstration Functions
def demonstrate_basic_dialogue_flow():
    """Demonstrate basic dialogue flow management"""
    print("\n" + "="*60)
    print("🔄 BASIC DIALOGUE FLOW DEMO")
    print("="*60)

    manager = AdvancedDialogueManager()
    user_id = "demo_user"
    session_id = "flow_demo"

    conversation = [
        "Hi there!",
        "I need to make a booking",
        "I'd like to book for tomorrow at 2pm",
        "Yes, that sounds perfect",
        "No, that's all for today"
    ]

    print("Demonstrating state-based dialogue flow...\n")

    for message in conversation:
        print(f"👤 User: {message}")

        try:
            result = manager.process_conversation_turn(user_id, session_id, message)
            print(f"🤖 Agent: {result['response']}")
            print(f"   State: {result['dialogue_state']}")
            print(f"   Collected: {list(result['collected_info'].keys())}")

            if result['conversation_complete']:
                print("   🏁 Conversation completed")
                break

        except Exception as e:
            print(f"🤖 Agent: Error in dialogue flow: {e}")

        print()


def demonstrate_booking_dialogue():
    """Demonstrate specialized booking dialogue"""
    print("\n" + "="*60)
    print("📅 BOOKING DIALOGUE DEMO")
    print("="*60)

    booking_manager = BookingDialogueManager()
    user_id = "customer_456"
    session_id = "booking_session"

    booking_conversation = [
        "Hello, I want to make a reservation",
        "I'd like to book for this Friday at 7:30pm",
        "Yes, please confirm that booking",
        "That's all, thank you!"
    ]

    print("Demonstrating specialized booking flow...\n")

    for message in booking_conversation:
        print(f"👤 Customer: {message}")

        try:
            result = booking_manager.process_conversation_turn(user_id, session_id, message)
            print(f"📅 Booking System: {result['response']}")
            print(f"   Current State: {result['dialogue_state']}")

            if result['collected_info']:
                print(f"   📋 Booking Details: {result['collected_info']}")

        except Exception as e:
            print(f"📅 Booking System: Error: {e}")

        print()


def demonstrate_error_recovery():
    """Demonstrate error recovery and clarification"""
    print("\n" + "="*60)
    print("🔧 ERROR RECOVERY DEMO")
    print("="*60)

    manager = AdvancedDialogueManager()
    user_id = "error_user"
    session_id = "error_session"

    error_conversation = [
        "I need to do something",
        "It's about the thing",
        "You know, the usual stuff",
        "Actually, I want to book a table for dinner",
        "Tomorrow at 8pm would be great"
    ]

    print("Demonstrating error recovery and clarification handling...\n")

    for message in error_conversation:
        print(f"👤 User: {message}")

        try:
            result = manager.process_conversation_turn(user_id, session_id, message)
            print(f"🤖 Agent: {result['response']}")
            print(f"   State: {result['dialogue_state']}")

            if result['dialogue_state'] == 'clarification_needed':
                print("   🔍 System detected need for clarification")
            elif result['dialogue_state'] == 'error_recovery':
                print("   🔧 System initiated error recovery")

        except Exception as e:
            print(f"🤖 Agent: System error: {e}")

        print()


def demonstrate_conversation_analytics():
    """Demonstrate conversation analytics and insights"""
    print("\n" + "="*60)
    print("📊 CONVERSATION ANALYTICS DEMO")
    print("="*60)

    manager = AdvancedDialogueManager()
    user_id = "analytics_user"
    session_id = "analytics_session"

    # Simulate a complete conversation
    full_conversation = [
        "Hello there",
        "I want to make a reservation",
        "For next Friday at 7pm",
        "Yes, confirm that please",
        "Perfect, thank you"
    ]

    for message in full_conversation:
        manager.process_conversation_turn(user_id, session_id, message)

    # Analyze conversation
    context = manager.get_or_create_context(user_id, session_id)
    collected_info = manager.get_collected_info(session_id)

    print("📈 Conversation Analysis:")
    print(f"   Total turns: {len(context.conversation_history)}")
    print(f"   Duration: {time.time() - context.conversation_history[0]['timestamp']:.1f} seconds")
    print(f"   Information collected: {len(collected_info)} items")
    print(f"   Final state: {manager.get_current_node(session_id).state.value}")

    # State progression
    states_visited = []
    for turn in context.conversation_history:
        if 'dialogue_state' in turn and turn['dialogue_state']:
            states_visited.append(turn['dialogue_state'])

    print(f"   State progression: {' → '.join(dict.fromkeys(states_visited))}")

    # Intent distribution
    intents = [turn.get('intent') for turn in context.conversation_history if turn.get('intent')]
    intent_counts = {}
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print(f"   Intent distribution: {intent_counts}")


def main():
    """Main demonstration function"""
    print("🎭 Advanced Dialogue Management")
    print("State-based Conversation Flow Control")
    print("=" * 60)

    print("\nAdvanced Features Demonstrated:")
    print("• State-based dialogue flows")
    print("• Dynamic conversation routing")
    print("• Information gathering and validation")
    print("• Error recovery and clarification handling")
    print("• Specialized dialogue managers (booking, support, etc.)")
    print("• Conversation analytics and insights")

    # Check API setup
    openai_available = bool(os.getenv("OPENAI_API_KEY"))
    anthropic_available = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"\nAPI Availability:")
    print(f"• OpenAI: {'✅' if openai_available else '❌'}")
    print(f"• Anthropic: {'✅' if anthropic_available else '❌'}")

    try:
        # Run demonstrations
        demonstrate_basic_dialogue_flow()
        demonstrate_booking_dialogue()
        demonstrate_error_recovery()
        demonstrate_conversation_analytics()

        print("\n" + "="*60)
        print("🎉 Advanced Dialogue Management Demo Complete!")

        print("\nKey Concepts Mastered:")
        print("• State-based conversation flow control")
        print("• Dynamic routing based on context and intent")
        print("• Information collection and validation")
        print("• Error recovery and graceful failure handling")
        print("• Specialized dialogue managers for specific domains")
        print("• Conversation analytics and performance tracking")

        print("\nProduction Implementation Tips:")
        print("• Define clear dialogue states for your use case")
        print("• Implement comprehensive validation rules")
        print("• Plan for error scenarios and recovery paths")
        print("• Use analytics to optimize conversation flows")
        print("• Test with real user scenarios")
        print("• Implement conversation handoff to humans when needed")

        print("\nNext Steps:")
        print("• Design dialogue flows for your specific domain")
        print("• Implement custom validation and business logic")
        print("• Add integration with backend systems")
        print("• Deploy with conversation monitoring and analytics")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Ensure dependencies are installed:")
        print("• pip install openai anthropic python-dotenv")


if __name__ == "__main__":
    main()