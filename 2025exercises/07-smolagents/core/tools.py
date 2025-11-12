"""
Custom tools for SmolAgents project.
"""

import json
import requests
from smolagents import tool, Tool
from smolagents import WebSearchTool, GoogleSearchTool, WikipediaSearchTool, VisitWebpageTool
from smolagents import FinalAnswerTool, PythonInterpreterTool
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# default tools
search_tool = WebSearchTool()
# search_tool = GoogleSearchTool()
visitwebpage_tool = VisitWebpageTool()
final_answer_tool = FinalAnswerTool()
python_tool = PythonInterpreterTool()

# ## Importing a Tool from the Hub
# from smolagents import load_tool
# image_generation_tool = load_tool(
#     "m-ric/text-to-image",
#     trust_remote_code=True
# )

# # 将HF Space作为工具集成
# image_generation_tool = Tool.from_space(
#     "black-forest-labs/FLUX.1-schnell",
#     name="ai_image_generator",
#     description="通过AI生成图像"
# )

# ## Importing a LangChain Tool
# from langchain.agents import load_tools
# search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])


@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get the current weather at the given location using the WeatherStack API.

    Args:
        location: The location (city name).
        celsius: Whether to return the temperature in Celsius (default is False, which returns Fahrenheit).

    Returns:
        A string describing the current weather at the location.
    """
    api_key = "294b07e9ec23e262dc8d9f3f281ac1b5"  # Replace with your API key from https://weatherstack.com/
    units = "m" if celsius else "f"  # 'm' for Celsius, 'f' for Fahrenheit

    url = f"http://api.weatherstack.com/current"
    params = {
        "access_key": api_key,
        "query": location,
        "units": "m"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if "error" in data:
            return f"获取天气数据错误: {data['error'].get('info', 'Unable to fetch weather data.')}"

        weather = data["current"]["weather_descriptions"][0]
        temp = data["current"]["temperature"]
        temp_unit = "°C" if celsius else "°F"

        return f"The current weather in {location} is {weather} with a temperature of {temp} {temp_unit}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"


@tool
def get_time_in_timezone(location: str) -> str:
    """
    Fetches the current time for a given location using the World Time API.
    Args:
        location: The location for which to fetch the current time, formatted as 'Region/City'.
    Returns:
        str: A string indicating the current time in the specified location, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"http://worldtimeapi.org/api/timezone/{location}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        current_time = data["datetime"]

        return f"The current time in {location} is {current_time}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching time data: {str(e)}"


@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate.
        
    Returns:
        The result of the calculation.
    """
    try:
        # Note: In production, use a safer evaluation method
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def get_time() -> str:
    """
    Get the current time.
    
    Returns:
        A string with the current time.
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.user_input = ""

    async def _validate_question(self, question) -> tuple[bool, str]:
        # Helper method to validate the question
        pass

    async def forward(self, question: str) -> str:
        # Validate the question first
        success, response = await self._validate_question(question)
        if not success:
            self.logger.error(response)
            return f"Error: {response}"
        # Ask the validated question and ensure non-empty response
        self.logger.info(f"Asking user: {question}")
        while True:
            self.user_input = input(f"{question} => Type your answer here:").strip()
            if self.user_input:
                break
            print("Please provide a non-empty answer.")
        self.logger.info(f"Received user input: {self.user_input}")
        return self.user_input

user_input_tool = UserInputTool()   


def call_for_cc_api(query: str) -> str:
    url = "https://lpai-inference-guan.inner.chj.cloud/inference/enterpris-smartbusiness/ssc-faq/8092predict4cc"
    payload = json.dumps({
        "text": query,
        "user_id": "smolagents",
        "top_k": 5,
    })
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.text


class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves relevant sections from customer service documentation."
    inputs = {
        "query": {
            "type": "string",
            "description": "A search query optimized for semantic similarity with target documents."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        response = call_for_cc_api(query)
        return "\nRetrieved documents:\n" + response

retriever_tool = RetrieverTool()

