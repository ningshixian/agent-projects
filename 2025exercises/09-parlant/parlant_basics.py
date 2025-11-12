"""
Parlant Basics - Conversational AI Framework

Parlant is a framework designed specifically for building conversational AI agents
with natural dialogue capabilities, context management, and sophisticated conversation flows.

Note: This module provides a conceptual implementation since Parlant may not be
a standard package. The concepts demonstrated here are universal for conversation systems.

Parlant 是一个旨在构建对话式 AI 代理的框架，其功能包括：
- 对话状态管理
- Turn-taking and dialogue flow
- 客户session会话管理
- Journey-based conversation flows
- 意图识别和响应生成
- 多轮对话

https://github.com/TranslationalAICenterISU/AgenticAITutorial2025Sept/blob/544cbf16c8c4b59b26dabb0e556150dcb49c805a/09-parlant/parlant_basics.py
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import openai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class IntentType(Enum):
    """Common conversation intent types"""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"


@dataclass
class ConversationContext:
    """Maintains conversation state and context"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_info: Dict[str, Any] = field(default_factory=dict)
    current_intent: Optional[IntentType] = None
    waiting_for: Optional[str] = None  # What we're waiting for from user
    conversation_state: str = "active"
    last_interaction: float = field(default_factory=time.time)
    topic_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: float
    speaker: str  # 'user' or 'agent'
    message: str
    intent: Optional[IntentType] = None
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationalAgent:
    """Base conversational agent with context management"""

    def __init__(self, agent_name: str = "Assistant"):
        self.agent_name = agent_name
        self.contexts: Dict[str, ConversationContext] = {}
        self.intent_patterns = self._initialize_intent_patterns()
        self.openai_client = self._setup_openai()
        self.anthropic_client = self._setup_anthropic()

    def _setup_openai(self) -> Optional[openai.OpenAI]:
        """Setup OpenAI client for conversation"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return openai.OpenAI(api_key=api_key)
        return None

    def _setup_anthropic(self) -> Optional[anthropic.Anthropic]:
        """Setup Anthropic client for conversation"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
        return None

    def _initialize_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize simple intent recognition patterns"""
        return {
            IntentType.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"
            ],
            IntentType.QUESTION: [
                "what", "how", "when", "where", "why", "who", "can you", "could you", "?"
            ],
            IntentType.REQUEST: [
                "please", "can you help", "i need", "i want", "could you", "help me"
            ],
            IntentType.COMPLAINT: [
                "problem", "issue", "wrong", "error", "not working", "frustrated", "disappointed"
            ],
            IntentType.GOODBYE: [
                "bye", "goodbye", "see you", "thanks", "that's all", "end", "stop"
            ],
            IntentType.CONFIRMATION: [
                "yes", "yeah", "correct", "right", "exactly", "that's right", "confirm"
            ],
            IntentType.CLARIFICATION: [
                "what do you mean", "can you explain", "i don't understand", "clarify", "confused"
            ]
        }

    def get_or_create_context(self, user_id: str, session_id: str) -> ConversationContext:
        """Get or create conversation context"""
        context_key = f"{user_id}_{session_id}"
        if context_key not in self.contexts:
            self.contexts[context_key] = ConversationContext(
                user_id=user_id,
                session_id=session_id
            )
        return self.contexts[context_key]

    def classify_intent(self, message: str) -> Tuple[IntentType, float]:
        """Simple intent classification based on keywords"""
        message_lower = message.lower()

        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                intent_scores[intent] = score / len(patterns)

        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            return best_intent, confidence

        return IntentType.UNKNOWN, 0.0

    def extract_entities(self, message: str) -> Dict[str, Any]:
        """Simple entity extraction (in practice, use NER models)"""
        entities = {}

        # Simple patterns for common entities
        import re

        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, message)
        if emails:
            entities['email'] = emails[0]

        # Phone number extraction (simple pattern)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, message)
        if phones:
            entities['phone'] = phones[0]

        # Numbers
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, message)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]

        return entities

    def generate_response_with_openai(self, context: ConversationContext, message: str) -> str:
        """Generate response using OpenAI"""
        if not self.openai_client:
            return "I'm sorry, I'm having trouble connecting to my language model."

        # Build conversation history for context
        messages = [
            {
                "role": "system",
                "content": f"You are {self.agent_name}, a helpful conversational AI assistant. "
                          f"Maintain context from previous messages. Be conversational and helpful."
            }
        ]

        # Add recent conversation history
        for turn in context.conversation_history[-5:]:  # Last 5 turns
            role = "user" if turn['speaker'] == 'user' else "assistant"
            messages.append({"role": role, "content": turn['message']})

        # Add current message
        messages.append({"role": "user", "content": message})

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I apologize, I'm experiencing technical difficulties: {str(e)}"

    def generate_response_with_anthropic(self, context: ConversationContext, message: str) -> str:
        """Generate response using Anthropic Claude"""
        if not self.anthropic_client:
            return "I'm sorry, I'm having trouble connecting to my language model."

        # Build conversation context
        conversation_context = ""
        for turn in context.conversation_history[-5:]:
            speaker = "Human" if turn['speaker'] == 'user' else "Assistant"
            conversation_context += f"{speaker}: {turn['message']}\n"

        prompt = f"""You are {self.agent_name}, a helpful conversational AI assistant.

Previous conversation:
{conversation_context}

Current human message: {message}

Please respond naturally and helpfully, maintaining the conversation context."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"I apologize, I'm experiencing technical difficulties: {str(e)}"

    def process_conversation_turn(self, user_id: str, session_id: str, message: str,
                                use_anthropic: bool = False) -> Dict[str, Any]:
        """Process a complete conversation turn"""
        # Get or create context
        context = self.get_or_create_context(user_id, session_id)

        # Update last interaction time
        context.last_interaction = time.time()

        # Classify intent and extract entities
        intent, confidence = self.classify_intent(message)
        entities = self.extract_entities(message)

        # Create conversation turn
        turn = ConversationTurn(
            timestamp=time.time(),
            speaker='user',
            message=message,
            intent=intent,
            confidence=confidence,
            entities=entities
        )

        # Add to conversation history
        context.conversation_history.append({
            'timestamp': turn.timestamp,
            'speaker': turn.speaker,
            'message': turn.message,
            'intent': turn.intent.value if turn.intent else None,
            'confidence': turn.confidence,
            'entities': turn.entities
        })

        # Update context based on intent
        context.current_intent = intent
        self._update_context_from_intent(context, intent, entities)

        # Generate response
        if use_anthropic and self.anthropic_client:
            response_text = self.generate_response_with_anthropic(context, message)
        else:
            response_text = self.generate_response_with_openai(context, message)

        # Add agent response to history
        agent_turn = {
            'timestamp': time.time(),
            'speaker': 'agent',
            'message': response_text,
            'intent': None,
            'confidence': 1.0,
            'entities': {}
        }
        context.conversation_history.append(agent_turn)

        return {
            'response': response_text,
            'intent': intent.value,
            'confidence': confidence,
            'entities': entities,
            'context_state': context.conversation_state,
            'session_id': session_id
        }

    def _update_context_from_intent(self, context: ConversationContext,
                                   intent: IntentType, entities: Dict[str, Any]):
        """Update conversation context based on detected intent"""
        if intent == IntentType.GREETING:
            context.conversation_state = "greeting"
        elif intent == IntentType.GOODBYE:
            context.conversation_state = "ending"
        elif intent == IntentType.REQUEST:
            context.waiting_for = "fulfillment"
        elif intent == IntentType.QUESTION:
            context.waiting_for = "answer"
        elif intent == IntentType.COMPLAINT:
            context.conversation_state = "problem_solving"

        # Update user info with extracted entities
        if entities:
            context.user_info.update(entities)

    def get_conversation_summary(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        context = self.get_or_create_context(user_id, session_id)

        return {
            'user_id': user_id,
            'session_id': session_id,
            'total_turns': len(context.conversation_history),
            'conversation_state': context.conversation_state,
            'current_intent': context.current_intent.value if context.current_intent else None,
            'user_info': context.user_info,
            'last_interaction': context.last_interaction,
            'session_duration': time.time() - (context.conversation_history[0]['timestamp'] if context.conversation_history else time.time())
        }


class CustomerServiceAgent(ConversationalAgent):
    """Specialized customer service conversational agent"""

    def __init__(self):
        super().__init__("Customer Service Agent")
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict[str, str]:
        """Initialize simple knowledge base for customer service"""
        return {
            'hours': "Our customer service hours are Monday-Friday 9AM-6PM EST.",
            'return_policy': "You can return items within 30 days of purchase with receipt.",
            'shipping': "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days.",
            'payment': "We accept all major credit cards, PayPal, and bank transfers.",
            'contact': "You can reach us at support@company.com or call 1-800-SUPPORT.",
            'warranty': "All products come with a 1-year manufacturer warranty.",
            'tracking': "You can track your order using the tracking number sent to your email."
        }

    def search_knowledge_base(self, query: str) -> Optional[str]:
        """Search knowledge base for relevant information"""
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        return None

    def generate_response_with_openai(self, context: ConversationContext, message: str) -> str:
        """Enhanced response generation with knowledge base integration"""
        # First check knowledge base
        kb_response = self.search_knowledge_base(message)

        if kb_response:
            return f"{kb_response} Is there anything else I can help you with?"

        # Otherwise use standard response generation
        return super().generate_response_with_openai(context, message)


class PersonalAssistantAgent(ConversationalAgent):
    """Personal assistant specialized conversational agent"""

    def __init__(self):
        super().__init__("Personal Assistant")
        self.capabilities = [
            "scheduling", "reminders", "weather", "calculations",
            "general questions", "recommendations"
        ]

    def process_conversation_turn(self, user_id: str, session_id: str, message: str,
                                use_anthropic: bool = False) -> Dict[str, Any]:
        """Enhanced processing with assistant-specific features"""
        result = super().process_conversation_turn(user_id, session_id, message, use_anthropic)

        # Add capability suggestions if user seems confused
        if result['intent'] == 'unknown':
            result['response'] += f"\n\nI can help with: {', '.join(self.capabilities[:3])}. What would you like to do?"

        return result


# Demonstration Functions
def demonstrate_basic_conversation():
    """Demonstrate basic conversation flow"""
    print("\n" + "="*50)
    print("💬 BASIC CONVERSATION DEMO")
    print("="*50)

    agent = ConversationalAgent("Demo Assistant")
    user_id = "demo_user"
    session_id = "session_001"

    conversation_flow = [
        "Hello there!",
        "What can you help me with?",
        "I have a question about your services",
        "What are your hours?",
        "Thank you for your help",
        "Goodbye!"
    ]

    print("Starting conversation simulation...\n")

    for i, message in enumerate(conversation_flow):
        print(f"👤 User: {message}")

        try:
            result = agent.process_conversation_turn(user_id, session_id, message)
            print(f"🤖 Agent: {result['response']}")
            print(f"   Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            if result['entities']:
                print(f"   Entities: {result['entities']}")
        except Exception as e:
            print(f"🤖 Agent: I'm sorry, I encountered an error: {e}")

        print()
        time.sleep(1)  # Simulate conversation pacing

    # Show conversation summary
    summary = agent.get_conversation_summary(user_id, session_id)
    print("📊 Conversation Summary:")
    print(f"   Total turns: {summary['total_turns']}")
    print(f"   Final state: {summary['conversation_state']}")
    print(f"   Session duration: {summary['session_duration']:.1f} seconds")


def demonstrate_customer_service():
    """Demonstrate customer service agent"""
    print("\n" + "="*50)
    print("🛎️ CUSTOMER SERVICE DEMO")
    print("="*50)

    agent = CustomerServiceAgent()
    user_id = "customer_123"
    session_id = "support_001"

    service_queries = [
        "Hi, I need help with a return",
        "What are your return policies?",
        "How long does shipping usually take?",
        "What payment methods do you accept?"
    ]

    for query in service_queries:
        print(f"👤 Customer: {query}")

        try:
            result = agent.process_conversation_turn(user_id, session_id, query)
            print(f"🛎️ Support: {result['response']}")
            print(f"   Intent: {result['intent']}")
        except Exception as e:
            print(f"🛎️ Support: I apologize for the technical difficulty: {e}")

        print()


def demonstrate_context_management():
    """Demonstrate conversation context and memory"""
    print("\n" + "="*50)
    print("🧠 CONTEXT MANAGEMENT DEMO")
    print("="*50)

    agent = PersonalAssistantAgent()
    user_id = "context_user"
    session_id = "context_session"

    # Multi-turn conversation that builds context
    context_conversation = [
        "My name is Alice and I work at TechCorp",
        "I need to schedule a meeting for next week",
        "The meeting is about the quarterly review",
        "Can you remind me what we discussed about my name earlier?",
        "What company did I mention I work for?"
    ]

    print("Demonstrating context preservation across turns...\n")

    for message in context_conversation:
        print(f"👤 Alice: {message}")

        try:
            result = agent.process_conversation_turn(user_id, session_id, message)
            print(f"🤖 Assistant: {result['response']}")

            # Show extracted entities if any
            if result['entities']:
                print(f"   📝 Extracted: {result['entities']}")
        except Exception as e:
            print(f"🤖 Assistant: I'm having trouble processing that: {e}")

        print()

    # Show final context state
    context = agent.get_or_create_context(user_id, session_id)
    print("📋 Final Context State:")
    print(f"   User info collected: {context.user_info}")
    print(f"   Current intent: {context.current_intent}")
    print(f"   Conversation state: {context.conversation_state}")


def demonstrate_intent_classification():
    """Demonstrate intent recognition capabilities"""
    print("\n" + "="*50)
    print("🎯 INTENT CLASSIFICATION DEMO")
    print("="*50)

    agent = ConversationalAgent()

    test_messages = [
        "Hello! How are you doing today?",
        "What time does the store close?",
        "Can you please help me find my order?",
        "I'm having trouble with my account login",
        "Thanks for your help, goodbye!",
        "Yes, that's exactly what I meant",
        "I don't understand what you're asking"
    ]

    print("Testing intent classification on various messages:\n")

    for message in test_messages:
        intent, confidence = agent.classify_intent(message)
        print(f"Message: '{message}'")
        print(f"Intent: {intent.value} (confidence: {confidence:.2f})")
        print()


def main():
    """Main demonstration function"""
    print("🗣️ Parlant Basics - Conversational AI Framework")
    print("=" * 60)

    print("\nKey Features Demonstrated:")
    print("• Conversation context management")
    print("• Intent classification and entity extraction")
    print("• Multi-turn dialogue handling")
    print("• Specialized agent types (customer service, personal assistant)")
    print("• Memory and state preservation")

    # Check API setup
    openai_available = bool(os.getenv("OPENAI_API_KEY"))
    anthropic_available = bool(os.getenv("ANTHROPIC_API_KEY"))

    print(f"\nAPI Availability:")
    print(f"• OpenAI: {'✅' if openai_available else '❌'}")
    print(f"• Anthropic: {'✅' if anthropic_available else '❌'}")

    if not (openai_available or anthropic_available):
        print("\n⚠️ No API keys configured - responses will be limited")

    try:
        # Run demonstrations
        demonstrate_intent_classification()
        demonstrate_basic_conversation()
        demonstrate_customer_service()
        demonstrate_context_management()

        print("\n" + "="*60)
        print("🎉 Parlant Basics Demo Complete!")

        print("\nKey Concepts Learned:")
        print("• Conversation state management across multiple turns")
        print("• Intent recognition for understanding user goals")
        print("• Entity extraction for capturing important information")
        print("• Context preservation for maintaining conversation memory")
        print("• Specialized agents for domain-specific conversations")

        print("\nProduction Considerations:")
        print("• Use proper NLP models for intent classification")
        print("• Implement robust entity recognition (NER)")
        print("• Add conversation analytics and monitoring")
        print("• Handle edge cases and fallback scenarios")
        print("• Scale context storage for multiple concurrent users")

        print("\nNext: Run dialogue_management.py for advanced conversation flows")

    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Make sure you have the required packages installed:")
        print("• pip install openai anthropic python-dotenv")


if __name__ == "__main__":
    main()