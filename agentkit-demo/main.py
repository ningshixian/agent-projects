import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List

from openai import OpenAI
from agents import Agent, RunConfig, Runner, function_tool
from agents.model_settings import ModelSettings
from chatkit.agents import AgentContext, stream_agent_response
from chatkit.agents import simple_to_agent_input
from chatkit.server import ChatKitServer, NonStreamingResult, StreamingResult
from chatkit.types import (
    Annotation,
    AssistantMessageContent,
    AssistantMessageItem,
    Attachment,
    ClientToolCallItem,
    FileSource,
    ThreadItem,
    ThreadItemDoneEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.responses import JSONResponse, Response, StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_store import MemoryStore
from assistant_agent import (
    assistant_agent,
)

app = FastAPI(title="ChatKit Knowledge Retrieval API")

# 加载环境变量
load_dotenv()

# model = os.getenv("LLM_MODEL_ID")
# apiKey = os.getenv("LLM_API_KEY")
# client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)


class MyChatKitServer(ChatKitServer):
    def __init__(self, agent: Agent[AgentContext]) -> None:
        self.store: Any = MemoryStore()
        super().__init__(self.store)
        self.assistant = agent

    async def respond(
        self,
        thread: ThreadMetadata,
        input_user_message: UserMessageItem | None,
        context: Any,
    ) -> AsyncIterator[ThreadStreamEvent]:
        context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )
        result = Runner.run_streamed(
            self.assistant,
            await simple_to_agent_input(input_user_message) if input_user_message else [],
            context=context,
            # run_config=RunConfig(model_settings=ModelSettings()),
        )
        async for event in stream_agent_response(context, result):
            yield event

    # ...

chatkit_server = MyChatKitServer(agent=assistant_agent)

def get_server() -> MyChatKitServer:
    return chatkit_server


@app.post("/chatkit")
async def chatkit_endpoint(
    request: Request, server: MyChatKitServer = Depends(get_server)
):

    payload = await request.body()
    result = await server.process(payload, {"request": request})

    if isinstance(result, StreamingResult):
        return StreamingResponse(result, media_type="text/event-stream")
    if hasattr(result, "json"):
        return Response(content=result.json, media_type="application/json")
    return JSONResponse(result)

# 运行：uvicorn main:app --reload

