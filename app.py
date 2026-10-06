import json
import uuid
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import TOOLS, run_tool

# --- Config ---

SYSTEM_PROMPT = """You are Ground Truth, an assistant that reads open map data the way a development \
economist would. People name any place on Earth (a market, a neighborhood, a town, their hometown) and \
you describe what its local economy looks like, how formal or informal it seems, and how far that read \
can be trusted. The approach mirrors IMF work that estimates informal economies from open geospatial \
data for places with no recent business survey.

How to investigate a place:
1. locate_place to get coordinates, country_code and suggested_radius_m.
2. scan_economic_footprint with those coordinates and radius.
3. estimate_formality with the same coordinates and radius, passing country_code.
4. check_map_coverage with the same coordinates and radius. Always do this before concluding: if coverage \
is low, lead with that caveat.
Call get_country_context when the user asks about the national picture, or when coverage is low and \
national statistics are the better guide.
For comparisons, investigate each place, then compare them side by side.
Reuse coordinates already found earlier in this conversation instead of locating the same place again.

How to answer:
- Write like an economist talking to a colleague: plain, short sentences.
- Start with one sentence that answers the question.
- Then give 3 to 4 simple bullets with the evidence. Each bullet is one plain sentence. Do not start \
bullets with bold labels such as "Density:" and do not use headings.
- Mention counts the natural way, for example "25 banks and 2 open markets within 500 m". Use densities \
per km2 only when comparing places of different sizes or when the user asks.
- Give the informality score and its band, for example "an informality score of 32 out of 100, leaning \
formal". Always call it the informality score (0 = fully formal, 100 = fully informal).
- Say what drives the score and how well the area is mapped.
- If estimate_formality returns an undercount_warning, say so plainly and lower your confidence.
- End with one plain sentence on how far to trust the result.
- No bold, no italics, no emojis. Keep it under about 150 words unless the user asks for more detail.
- Use only numbers that appear in tool results. Never invent figures.
- Remember that map counts measure what volunteers have mapped, not a census. Say so when it matters.
- If a tool returns an error, follow its advice (for example retry with a smaller radius) or tell the \
user exactly what to try next.
- If the user asks about something unrelated, say in one sentence what you do and suggest a place to try."""
MAX_TOOL_ROUNDS = 10

# --- The Harness ---


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
            tools=TOOLS,
        ).choices[0].message

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content or "I couldn't put an answer together. Try rephrasing the question.", tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return (
        "I ran out of investigation steps before finishing. Try asking about one place at a time.",
        tool_calls,
    )


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append user's message to the context
    sessions[session_id] += [{"role": "user", "content": request.message}]

    try:
        response, tool_calls = run_agent(sessions[session_id])
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
