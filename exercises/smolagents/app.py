from smolagents import CodeAgent, HfApiModel, Tool
from smolagents import load_tool,tool,DuckDuckGoSearchTool
import datetime
import requests
import pytz
import yaml
from tools.web_search import DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# # Importing a Tool from the Hub
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Importing a Hugging Face Space as a Tool
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        FinalAnswerTool(), 
        DuckDuckGoSearchTool(), 
        VisitWebpageTool(),
        image_generation_tool,
        get_current_time_in_timezone
    ], ## add your tools here (don't remove final answer)
    
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

# agent.run("Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros.")

# # Sharing Our Custom Party Preparator Agent to the Hub
# agent.push_to_hub('ningshixian/SelfAgent')
# # download the agent again, use the code below:
# alfred_agent = agent.from_hub('ningshixian/SelfAgent', trust_remote_code=True)

GradioUI(agent).launch()