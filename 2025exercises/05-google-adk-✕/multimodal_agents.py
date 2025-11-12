"""
Google Gemini Multimodal Agents
Advanced examples of building multimodal AI agents with Gemini
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import base64

# Google AI imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()


class VisualAnalysisAgent:
    """Agent specialized in visual content analysis"""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Optimized settings for visual analysis
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent analysis
            top_p=0.8,
            max_output_tokens=2048,
        )

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

    def analyze_ui_screenshot(self, image_path: str) -> Dict[str, Any]:
        """Analyze UI screenshots for usability insights"""

        analysis_prompt = """
        Analyze this user interface screenshot and provide detailed feedback:

        1. VISUAL HIERARCHY:
        - Is there a clear visual hierarchy?
        - Which elements draw attention first?
        - Are important actions prominently displayed?

        2. USABILITY ASSESSMENT:
        - How intuitive is the navigation?
        - Are buttons and links clearly identifiable?
        - Is the layout clean and organized?

        3. ACCESSIBILITY CONSIDERATIONS:
        - Text readability and contrast
        - Button sizes and spacing
        - Color usage for information

        4. IMPROVEMENT RECOMMENDATIONS:
        - Specific suggestions for better UX
        - Layout optimizations
        - Color and typography improvements

        5. OVERALL RATING:
        - Rate the UI from 1-10 for usability
        - Provide reasoning for the rating

        Format your response as structured analysis with clear sections.
        """

        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            response = self.model.generate_content(
                [analysis_prompt, image_part],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            return {
                "analysis": response.text if response.candidates else "Analysis blocked",
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path
            }

        except Exception as e:
            return {"error": str(e), "analysis": None}

    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract and structure text from images (OCR functionality)"""

        ocr_prompt = """
        Extract all text from this image and structure it logically:

        1. MAIN HEADINGS/TITLES (if any)
        2. BODY TEXT (paragraphs, descriptions)
        3. LABELS AND CAPTIONS
        4. NUMBERS AND QUANTITIES
        5. URLS AND EMAIL ADDRESSES
        6. ANY OTHER TEXT ELEMENTS

        For each category, list the text found and indicate its approximate position (top, middle, bottom, left, right, center).

        If the image contains forms, tables, or structured data, preserve that structure in your output.

        Also provide:
        - Overall text quality assessment
        - Any text that appears unclear or might be misread
        - Language(s) detected
        """

        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            response = self.model.generate_content(
                [ocr_prompt, image_part],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            return {
                "extracted_text": response.text if response.candidates else "Extraction blocked",
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path
            }

        except Exception as e:
            return {"error": str(e), "extracted_text": None}

    def analyze_product_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze product images for e-commerce insights"""

        product_analysis_prompt = """
        Analyze this product image for e-commerce purposes:

        1. PRODUCT IDENTIFICATION:
        - What type of product is this?
        - Brand (if visible)
        - Model or specific variant (if identifiable)
        - Key features visible in the image

        2. IMAGE QUALITY ASSESSMENT:
        - Resolution and clarity
        - Lighting and shadows
        - Background and composition
        - Multiple angles shown (if applicable)

        3. MARKETING EFFECTIVENESS:
        - How appealing is the presentation?
        - Does it highlight key selling points?
        - Professional photography quality
        - Staging and context

        4. MISSING ELEMENTS:
        - What additional images would be helpful?
        - Scale/size references needed?
        - Detail shots required?
        - Lifestyle context missing?

        5. COMPETITIVE ANALYSIS:
        - How does this compare to typical product photos?
        - Industry standard compliance
        - Unique selling proposition visibility

        6. RECOMMENDATIONS:
        - Specific improvements for better conversion
        - Additional photos to include
        - Staging suggestions

        Provide actionable insights for improving the product listing.
        """

        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            response = self.model.generate_content(
                [product_analysis_prompt, image_part],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            return {
                "product_analysis": response.text if response.candidates else "Analysis blocked",
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path
            }

        except Exception as e:
            return {"error": str(e), "product_analysis": None}

    def compare_visual_versions(self, image1_path: str, image2_path: str,
                               comparison_context: str = "general") -> Dict[str, Any]:
        """Compare two versions of visual content"""

        comparison_prompts = {
            "ui_design": """
                Compare these two UI designs and analyze:
                1. Layout differences and improvements
                2. Visual hierarchy changes
                3. Color scheme modifications
                4. Typography updates
                5. User experience implications
                6. Which version is more effective and why
                """,
            "product": """
                Compare these two product images:
                1. Photography quality differences
                2. Product presentation improvements
                3. Background and staging changes
                4. Lighting and composition differences
                5. Marketing effectiveness comparison
                6. Recommendation for best version
                """,
            "general": """
                Compare these two images and provide:
                1. Key visual differences
                2. Quality assessment of each
                3. Context and purpose evaluation
                4. Strengths and weaknesses of each
                5. Overall recommendation
                """
        }

        prompt = comparison_prompts.get(comparison_context, comparison_prompts["general"])

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

            response = self.model.generate_content(
                [prompt, "First image:", image1_part, "Second image:", image2_part],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )

            return {
                "comparison": response.text if response.candidates else "Comparison blocked",
                "images": [image1_path, image2_path],
                "context": comparison_context,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e), "comparison": None}

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


class DocumentProcessingAgent:
    """Agent for processing various document types with multimodal capabilities"""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.generation_config = genai.types.GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            max_output_tokens=4096,
        )

    def analyze_invoice_or_receipt(self, document_path: str) -> Dict[str, Any]:
        """Extract structured data from invoices or receipts"""

        invoice_prompt = """
        Analyze this invoice/receipt and extract structured information:

        1. VENDOR INFORMATION:
        - Company name
        - Address
        - Contact information
        - Tax ID or registration number

        2. CUSTOMER/BILLING INFORMATION:
        - Customer name
        - Billing address
        - Customer ID (if present)

        3. DOCUMENT DETAILS:
        - Invoice/receipt number
        - Date of issue
        - Due date (if applicable)
        - Payment terms

        4. LINE ITEMS:
        For each item/service:
        - Description
        - Quantity
        - Unit price
        - Line total

        5. FINANCIAL SUMMARY:
        - Subtotal
        - Tax amount and rate
        - Discount (if any)
        - Total amount
        - Currency

        6. PAYMENT INFORMATION:
        - Payment method accepted/used
        - Account details (if visible)

        Format the response as structured JSON-like data for easy parsing.
        If any information is unclear or missing, indicate that explicitly.
        """

        return self._process_document(document_path, invoice_prompt, "invoice_data")

    def analyze_contract_or_agreement(self, document_path: str) -> Dict[str, Any]:
        """Analyze contracts and legal agreements"""

        contract_prompt = """
        Analyze this contract/agreement document and extract key information:

        1. DOCUMENT TYPE AND TITLE:
        - Type of contract/agreement
        - Official title
        - Document ID or reference number

        2. PARTIES INVOLVED:
        - All parties to the agreement
        - Their roles (client, vendor, contractor, etc.)
        - Contact information

        3. TERM AND DURATION:
        - Start date
        - End date or duration
        - Renewal terms (if any)

        4. KEY OBLIGATIONS:
        - Main obligations of each party
        - Deliverables or services
        - Performance standards

        5. FINANCIAL TERMS:
        - Payment amounts
        - Payment schedule
        - Currency
        - Late fees or penalties

        6. IMPORTANT CLAUSES:
        - Termination conditions
        - Confidentiality requirements
        - Liability limitations
        - Dispute resolution

        7. CRITICAL DATES:
        - All important deadlines
        - Milestone dates
        - Review periods

        8. RISK FACTORS:
        - Potential areas of concern
        - Ambiguous terms
        - Missing standard clauses

        Provide a comprehensive analysis that would be useful for legal review.
        """

        return self._process_document(document_path, contract_prompt, "contract_analysis")

    def analyze_report_or_presentation(self, document_path: str) -> Dict[str, Any]:
        """Analyze business reports and presentations"""

        report_prompt = """
        Analyze this business report/presentation and provide:

        1. DOCUMENT OVERVIEW:
        - Document type and purpose
        - Intended audience
        - Date and author (if visible)
        - Executive summary

        2. STRUCTURE ANALYSIS:
        - Main sections and topics
        - Logical flow and organization
        - Visual elements (charts, graphs, images)

        3. KEY FINDINGS:
        - Main conclusions or recommendations
        - Important data points
        - Trends or patterns identified
        - Performance metrics

        4. DATA ANALYSIS:
        - Types of data presented
        - Data quality and completeness
        - Visual representation effectiveness
        - Statistical significance

        5. ACTIONABLE INSIGHTS:
        - Strategic recommendations
        - Areas requiring attention
        - Opportunities identified
        - Risk factors mentioned

        6. PRESENTATION QUALITY:
        - Clarity and readability
        - Visual design effectiveness
        - Professional appearance
        - Completeness of information

        7. FOLLOW-UP ACTIONS:
        - Recommended next steps
        - Information gaps to address
        - Stakeholders to involve
        - Timeline considerations

        Provide insights that would be valuable for business decision-making.
        """

        return self._process_document(document_path, report_prompt, "report_analysis")

    def extract_form_data(self, document_path: str) -> Dict[str, Any]:
        """Extract data from forms and applications"""

        form_prompt = """
        Extract all form data from this document:

        1. FORM IDENTIFICATION:
        - Form title and type
        - Form number or ID
        - Version or revision date

        2. PERSONAL INFORMATION FIELDS:
        - Names (first, last, middle)
        - Addresses (current, previous)
        - Contact information (phone, email)
        - Identification numbers (SSN, ID, etc.)

        3. FORM-SPECIFIC DATA:
        - All filled-in fields and their values
        - Checkboxes and their status
        - Date fields
        - Signature areas and dates

        4. INCOMPLETE OR UNCLEAR AREAS:
        - Missing required information
        - Illegible handwriting
        - Partially filled sections

        5. VALIDATION CHECKS:
        - Data consistency
        - Required field completion
        - Format validation (dates, numbers, etc.)

        Present the data in a structured format that could be used for data entry or validation.
        Clearly mark any uncertain or incomplete information.
        """

        return self._process_document(document_path, form_prompt, "form_data")

    def _process_document(self, document_path: str, prompt: str, analysis_type: str) -> Dict[str, Any]:
        """Generic document processing method"""
        try:
            if not os.path.exists(document_path):
                return {"error": f"Document not found: {document_path}"}

            with open(document_path, 'rb') as file:
                file_data = file.read()

            # Determine MIME type
            file_extension = Path(document_path).suffix.lower()
            mime_type = self._get_document_mime_type(file_extension)

            file_part = {
                'mime_type': mime_type,
                'data': file_data
            }

            response = self.model.generate_content(
                [prompt, file_part],
                generation_config=self.generation_config
            )

            return {
                analysis_type: response.text if response.candidates else "Analysis blocked",
                "document_path": document_path,
                "document_type": file_extension,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e), analysis_type: None}

    def _get_document_mime_type(self, extension: str) -> str:
        """Get appropriate MIME type for document"""
        mime_types = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png'
        }
        return mime_types.get(extension, 'application/octet-stream')


class ContentCreationAgent:
    """Agent for creating multimodal content"""

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=4096,
        )

    def generate_alt_text(self, image_path: str) -> str:
        """Generate accessibility-friendly alt text for images"""

        alt_text_prompt = """
        Generate concise, descriptive alt text for this image that would be helpful for screen readers:

        Requirements:
        - Be specific and descriptive
        - Keep it concise (under 125 characters if possible)
        - Focus on relevant content and context
        - Avoid starting with "Image of" or "Picture of"
        - Include relevant details for understanding the image's purpose

        Return only the alt text, without additional formatting or explanation.
        """

        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            response = self.model.generate_content(
                [alt_text_prompt, image_part],
                generation_config=self.generation_config
            )

            return response.text.strip() if response.candidates else "Alt text generation blocked"

        except Exception as e:
            return f"Error generating alt text: {e}"

    def create_social_media_content(self, image_path: str, platform: str = "general") -> Dict[str, Any]:
        """Create social media content based on an image"""

        platform_specs = {
            "instagram": {
                "caption_length": "2200 characters max",
                "hashtag_count": "up to 30 hashtags",
                "tone": "casual, engaging, visual-focused"
            },
            "linkedin": {
                "caption_length": "3000 characters max",
                "hashtag_count": "3-5 professional hashtags",
                "tone": "professional, informative, business-focused"
            },
            "twitter": {
                "caption_length": "280 characters max",
                "hashtag_count": "1-2 relevant hashtags",
                "tone": "concise, engaging, conversation-starter"
            },
            "facebook": {
                "caption_length": "63206 characters max (but keep under 500 for engagement)",
                "hashtag_count": "1-5 hashtags",
                "tone": "friendly, community-focused, story-driven"
            },
            "general": {
                "caption_length": "flexible length",
                "hashtag_count": "relevant hashtags",
                "tone": "engaging and appropriate for the content"
            }
        }

        spec = platform_specs.get(platform, platform_specs["general"])

        content_prompt = f"""
        Create social media content for {platform.title()} based on this image:

        Platform specifications:
        - Caption length: {spec['caption_length']}
        - Hashtags: {spec['hashtag_count']}
        - Tone: {spec['tone']}

        Please provide:
        1. MAIN CAPTION: Engaging caption that describes the image and provides context
        2. HASHTAGS: Relevant hashtags for discoverability
        3. CALL-TO-ACTION: Appropriate engagement prompt
        4. ALTERNATIVE CAPTIONS: 2 shorter/longer variations

        Make the content engaging and authentic while being optimized for the platform.
        """

        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            response = self.model.generate_content(
                [content_prompt, image_part],
                generation_config=self.generation_config
            )

            return {
                "platform": platform,
                "content": response.text if response.candidates else "Content creation blocked",
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e), "content": None}

    def generate_image_description(self, image_path: str, detail_level: str = "detailed") -> str:
        """Generate detailed descriptions of images for various use cases"""

        detail_prompts = {
            "brief": "Provide a brief, one-sentence description of what's shown in this image.",
            "detailed": """Provide a comprehensive description of this image including:
                - Main subjects and objects
                - Setting and environment
                - Colors, lighting, and mood
                - Actions or activities taking place
                - Notable details or interesting elements""",
            "artistic": """Analyze this image from an artistic perspective:
                - Composition and visual elements
                - Color palette and lighting techniques
                - Style and artistic approach
                - Emotional impact and mood
                - Technical aspects (if apparent)"""
        }

        prompt = detail_prompts.get(detail_level, detail_prompts["detailed"])

        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

            image_part = {
                'mime_type': self._get_mime_type(image_path),
                'data': image_data
            }

            response = self.model.generate_content(
                [prompt, image_part],
                generation_config=self.generation_config
            )

            return response.text if response.candidates else "Description generation blocked"

        except Exception as e:
            return f"Error generating description: {e}"

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type based on file extension"""
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')


def create_sample_image(width: int = 400, height: int = 300, color: str = "blue") -> str:
    """Create a simple sample image for testing (requires PIL)"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a simple colored rectangle with some text
        img = Image.new('RGB', (width, height), color=color)
        draw = ImageDraw.Draw(img)

        # Add some text
        text = f"Sample Image\\n{width}x{height}\\nColor: {color}"

        # Try to use a default font, fall back to basic if needed
        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Calculate text position (centered)
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text) * 6  # Approximate
            text_height = 15  # Approximate

        x = (width - text_width) // 2
        y = (height - text_height) // 2

        draw.text((x, y), text, fill="white", font=font)

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        img.save(temp_file.name)
        temp_file.close()

        return temp_file.name

    except ImportError:
        print("PIL not available. Install with: pip install Pillow")
        return None


def demonstrate_multimodal_agents():
    """Demonstrate multimodal agent capabilities"""

    print("🎨 GOOGLE GEMINI MULTIMODAL AGENTS DEMO")
    print("=" * 50)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found. Please set it in your .env file")
        return

    print("✅ Google API key found, initializing multimodal agents...")

    # Test Visual Analysis Agent
    print("\n👀 Testing Visual Analysis Agent")
    print("-" * 40)

    try:
        visual_agent = VisualAnalysisAgent()

        # Create a sample image for testing
        sample_image = create_sample_image(600, 400, "lightblue")

        if sample_image:
            print(f"Created sample image: {sample_image}")

            # Test OCR functionality
            print("\n📝 Testing Text Extraction:")
            ocr_result = visual_agent.extract_text_from_image(sample_image)
            if ocr_result.get("extracted_text"):
                print("✅ Text extraction successful")
                print(f"Result preview: {ocr_result['extracted_text'][:200]}...")

            # Clean up
            os.unlink(sample_image)
        else:
            print("⚠️  Could not create sample image for testing")

    except Exception as e:
        print(f"❌ Visual analysis error: {e}")

    # Test Document Processing Agent
    print("\n📄 Testing Document Processing Agent")
    print("-" * 40)

    try:
        doc_agent = DocumentProcessingAgent()

        # Create a simple text file for testing
        sample_text = """
        INVOICE #12345
        Date: 2024-01-15

        Bill To:
        John Doe
        123 Main Street
        Anytown, ST 12345

        Description          Qty    Price    Total
        Consulting Services   10    $100.00  $1,000.00
        Software License       1    $250.00  $  250.00

        Subtotal:                            $1,250.00
        Tax (8%):                            $  100.00
        Total:                               $1,350.00
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(sample_text)
            temp_file_path = temp_file.name

        print(f"Created sample document: {temp_file_path}")

        # Test invoice processing
        invoice_result = doc_agent.analyze_invoice_or_receipt(temp_file_path)
        if invoice_result.get("invoice_data"):
            print("✅ Invoice analysis successful")
            print(f"Result preview: {invoice_result['invoice_data'][:300]}...")

        # Clean up
        os.unlink(temp_file_path)

    except Exception as e:
        print(f"❌ Document processing error: {e}")

    # Test Content Creation Agent
    print("\n✨ Testing Content Creation Agent")
    print("-" * 40)

    try:
        content_agent = ContentCreationAgent()

        # Create another sample image
        sample_image = create_sample_image(500, 300, "lightgreen")

        if sample_image:
            print(f"Created sample image: {sample_image}")

            # Test alt text generation
            print("\n🔤 Generating Alt Text:")
            alt_text = content_agent.generate_alt_text(sample_image)
            print(f"Generated alt text: {alt_text}")

            # Test social media content creation
            print("\n📱 Creating Social Media Content:")
            social_content = content_agent.create_social_media_content(sample_image, "instagram")
            if social_content.get("content"):
                print("✅ Social media content creation successful")
                print(f"Content preview: {social_content['content'][:200]}...")

            # Clean up
            os.unlink(sample_image)

    except Exception as e:
        print(f"❌ Content creation error: {e}")

    print("\n🎉 Multimodal agents demonstration completed!")
    print("\n💡 Key Capabilities Demonstrated:")
    print("1. Visual Analysis: UI screenshots, product images, OCR")
    print("2. Document Processing: Invoices, contracts, forms, reports")
    print("3. Content Creation: Alt text, social media content, descriptions")
    print("4. Comparative Analysis: Version comparisons and improvements")
    print("5. Structured Data Extraction: From various document types")

    print("\n🔧 Use Cases:")
    print("  • E-commerce product analysis and optimization")
    print("  • Document automation and data extraction")
    print("  • Accessibility content generation")
    print("  • Social media content creation at scale")
    print("  • UI/UX analysis and improvement recommendations")


if __name__ == "__main__":
    demonstrate_multimodal_agents()