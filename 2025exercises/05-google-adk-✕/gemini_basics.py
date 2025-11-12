"""
Google Gemini Basics
Introduction to using Google's Gemini models for agentic AI applications
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Google AI imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()


class GeminiAgent:
    """Agent using Google's Gemini model"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Configure safety settings (more permissive for development)
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        # Generation configuration
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )

        self.conversation_history = []

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Have a conversation with Gemini"""
        try:
            # Prepare the prompt
            full_prompt = message
            if system_prompt:
                full_prompt = f"System: {system_prompt}\\n\\nUser: {message}"

            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            # Check if response was blocked
            if response.candidates:
                response_text = response.text

                # Store in conversation history
                self.conversation_history.append({"user": message, "assistant": response_text})

                return response_text
            else:
                return "Response was blocked by safety filters or other constraints."

        except Exception as e:
            return f"Error generating response: {e}"

    def analyze_image(self, image_path: str, prompt: str = "Describe this image") -> str:
        """Analyze an image with Gemini"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return f"Image file not found: {image_path}"

            # Read and prepare image
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            # Create image part
            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            # Generate response with image
            response = self.model.generate_content(
                [prompt, image_part],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            return response.text if response.candidates else "Image analysis blocked."

        except Exception as e:
            return f"Error analyzing image: {e}"

    def function_calling_demo(self, user_request: str) -> str:
        """Demonstrate function calling with Gemini"""

        # Define available functions
        function_declarations = [
            {
                "name": "calculate",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or location name"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]

        try:
            # Create model with function calling
            model_with_functions = genai.GenerativeModel(
                self.model_name,
                tools=function_declarations
            )

            # Generate response
            response = model_with_functions.generate_content(
                user_request,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call'):
                        # Execute the function
                        function_name = part.function_call.name
                        function_args = dict(part.function_call.args)

                        result = self._execute_function(function_name, function_args)

                        return f"Function called: {function_name}\\nArguments: {function_args}\\nResult: {result}"

            return response.text if response.candidates else "No response generated."

        except Exception as e:
            return f"Error with function calling: {e}"

    def _execute_function(self, function_name: str, args: Dict[str, Any]) -> str:
        """Execute a function call (mock implementation)"""

        if function_name == "calculate":
            expression = args.get("expression", "")
            try:
                # Simple eval for demo - use safe math library in production
                result = eval(expression)
                return f"Calculation result: {result}"
            except Exception as e:
                return f"Calculation error: {e}"

        elif function_name == "get_weather":
            location = args.get("location", "")
            # Mock weather data
            return f"Weather in {location}: 22°C, partly cloudy, 60% humidity"

        else:
            return f"Unknown function: {function_name}"

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type based on file extension"""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.conversation_history:
            return "No conversation history"

        try:
            # Create summary prompt
            history_text = "\\n".join([
                f"User: {turn['user']}\\nAssistant: {turn['assistant']}"
                for turn in self.conversation_history
            ])

            summary_prompt = f"Summarize this conversation in 2-3 sentences:\\n\\n{history_text}"

            response = self.model.generate_content(
                summary_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=200
                ),
                safety_settings=self.safety_settings
            )

            return response.text if response.candidates else "Could not generate summary."

        except Exception as e:
            return f"Error generating summary: {e}"


class MultimodalAgent:
    """Agent that can process multiple types of input"""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def analyze_document(self, file_path: str, question: str = "Analyze this document") -> str:
        """Analyze various document types"""
        try:
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"

            with open(file_path, 'rb') as file:
                file_data = file.read()

            # Determine file type and create appropriate part
            file_extension = Path(file_path).suffix.lower()

            if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                mime_type = f"image/{file_extension[1:]}" if file_extension != '.jpg' else 'image/jpeg'
            elif file_extension == '.pdf':
                mime_type = 'application/pdf'
            elif file_extension in ['.txt', '.md']:
                mime_type = 'text/plain'
            else:
                mime_type = 'application/octet-stream'

            file_part = {
                'mime_type': mime_type,
                'data': file_data
            }

            response = self.model.generate_content([question, file_part])
            return response.text if response.candidates else "Analysis blocked."

        except Exception as e:
            return f"Error analyzing document: {e}"

    def compare_images(self, image1_path: str, image2_path: str,
                      comparison_prompt: str = "Compare these two images") -> str:
        """Compare two images"""
        try:
            # Read both images
            with open(image1_path, 'rb') as f1:
                image1_data = f1.read()

            with open(image2_path, 'rb') as f2:
                image2_data = f2.read()

            # Create image parts
            image1_part = {
                'mime_type': self._get_mime_type(image1_path),
                'data': image1_data
            }

            image2_part = {
                'mime_type': self._get_mime_type(image2_path),
                'data': image2_data
            }

            # Generate comparison
            response = self.model.generate_content([
                comparison_prompt,
                image1_part,
                image2_part
            ])

            return response.text if response.candidates else "Comparison blocked."

        except Exception as e:
            return f"Error comparing images: {e}"

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for file"""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')


def list_available_models():
    """List available Gemini models"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not found")
            return

        genai.configure(api_key=api_key)

        print("📋 Available Gemini Models:")
        print("-" * 40)

        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✅ {model.name}")
                print(f"   Display Name: {model.display_name}")
                print(f"   Description: {model.description}")
                print(f"   Input Token Limit: {model.input_token_limit}")
                print(f"   Output Token Limit: {model.output_token_limit}")
                print()

    except Exception as e:
        print(f"❌ Error listing models: {e}")


def demonstrate_gemini():
    """Demonstrate Gemini capabilities"""

    print("🔍 Google Gemini Demonstration")
    print("=" * 40)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found. Please set it in your .env file")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return

    print("✅ Google API key found")

    # List available models
    print("\\n📋 Available Models:")
    list_available_models()

    # Test basic chat
    print("\\n💬 Testing Basic Chat")
    print("-" * 30)

    try:
        agent = GeminiAgent("gemini-2.5-flash")

        system_prompt = "You are a helpful AI assistant specializing in technology and programming."

        test_questions = [
            "What are the main advantages of using Google's Gemini models?",
            "Explain the concept of multimodal AI in simple terms.",
            "How does Gemini compare to other large language models?"
        ]

        for question in test_questions:
            print(f"\\n👤 User: {question}")
            response = agent.chat(question, system_prompt)
            print(f"🤖 Gemini: {response}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Chat test error: {e}")

    # Test function calling
    print("\\n🔧 Testing Function Calling")
    print("-" * 30)

    try:
        function_requests = [
            "Calculate the result of 25 * 4 + 100",
            "What's the weather like in San Francisco?",
            "Can you help me with both a calculation (15 + 27) and weather for New York?"
        ]

        for request in function_requests:
            print(f"\\n👤 User: {request}")
            response = agent.function_calling_demo(request)
            print(f"🤖 Gemini: {response}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Function calling test error: {e}")

    # Show conversation summary
    print("\\n📋 Conversation Summary")
    print("-" * 30)
    try:
        summary = agent.get_conversation_summary()
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"❌ Summary error: {e}")

    print("\\n✅ Gemini demonstration completed!")
    print("\\nKey takeaways:")
    print("- Gemini offers strong reasoning capabilities")
    print("- Function calling enables tool integration")
    print("- Multimodal capabilities support images and documents")
    print("- Safety settings provide content filtering")


if __name__ == "__main__":
    demonstrate_gemini()