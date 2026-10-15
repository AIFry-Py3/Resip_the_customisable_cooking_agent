import uuid
from typing import TypedDict, Annotated, Literal
import operator

from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

SYSTEM_PROMPT = """You are a dinner-planning and recipe customization assistant.
For every recipe request, account for urgency, available ingredients, energy levels,
likes/dislikes, and health conditions (e.g. acidity -> avoid excess tomato/chili/garlic,
suggest fennel/cumin/ginger instead). Give a clear ingredient list and simple numbered steps.
Keep tone warm and practical."""


# ---------- Topic 1: typed State ----------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # reducer: add_messages
    ingredients_seen: Annotated[list[str], operator.add]    # Topic 2: second reducer
    intent: str                                             # no reducer -> overwritten
    trim_info: dict                                         # no reducer -> overwritten


# ---------- Router node + conditional edge ----------
def route_node(state: ChatState) -> dict:
    last = state["messages"][-1]
    text = last.content.lower() if isinstance(last.content, str) else ""
    keywords = ["recipe", "cook", "dinner", "pasta", "eat", "meal", "lunch", "breakfast", "food"]
    intent = "recipe" if any(k in text for k in keywords) else "chat"
    return {"intent": intent}


def route_condition(state: ChatState) -> Literal["recipe_node", "chat_node"]:
    return "recipe_node" if state["intent"] == "recipe" else "chat_node"


# ---------- Topic 3: trimming/filtering ----------
def _trim(messages: list[BaseMessage]) -> tuple[list[BaseMessage], dict]:
    before = len(messages)
    trimmed = trim_messages(
        messages,
        max_tokens=800,
        strategy="last",
        token_counter=llm,
        include_system=False,
        allow_partial=False,
        start_on="human",
    )
    after = len(trimmed)
    return trimmed, {"before": before, "after": after}


# ---------- Topic 4: long-term memory (store) ----------
def _profile_text(config: RunnableConfig, store: BaseStore) -> str:
    user_id = config["configurable"].get("user_id", "default_user")
    item = store.get(("profiles", user_id), "profile")
    if item and item.value:
        p = item.value
        return f"\n\nKnown user profile — name: {p.get('name','unknown')}, preference: {p.get('preference','none')}."
    return ""


def recipe_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    trimmed, info = _trim(state["messages"])
    sys = SystemMessage(content=SYSTEM_PROMPT + _profile_text(config, store))
    response = llm.invoke([sys] + trimmed)
    return {"messages": [response], "trim_info": info, "ingredients_seen": []}


def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore) -> dict:
    trimmed, info = _trim(state["messages"])
    sys = SystemMessage(content=SYSTEM_PROMPT + _profile_text(config, store))
    response = llm.invoke([sys] + trimmed)
    return {"messages": [response], "trim_info": info}


# ---------- Short-term memory: checkpointer ----------
checkpointer = InMemorySaver()
store = InMemoryStore()

graph = StateGraph(ChatState)
graph.add_node("route_node", route_node)
graph.add_node("recipe_node", recipe_node)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "route_node")
graph.add_conditional_edges("route_node", route_condition, {
    "recipe_node": "recipe_node",
    "chat_node": "chat_node",
})
graph.add_edge("recipe_node", END)
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer, store=store)


def save_profile(user_id: str, name: str, preference: str):
    store.put(("profiles", user_id), "profile", {"name": name, "preference": preference})


def get_profile(user_id: str):
    item = store.get(("profiles", user_id), "profile")
    return item.value if item else None


# ---------- Image ingredient detection ----------
import base64
from langchain_core.messages import HumanMessage as _HumanMessage

def detect_ingredients(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    msg = _HumanMessage(content=[
        {"type": "text", "text": "List only the food ingredients visible in this image, as a plain comma-separated list. No extra commentary."},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
    ])
    response = llm.invoke([msg])
    return response.content