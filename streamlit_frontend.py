import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot, save_profile, get_profile, detect_ingredients

st.set_page_config(page_title="Customisable Recipe Maker Chatbot", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #1a1915;
    color: #e8e6df;
}
[data-testid="stSidebar"] {
    background-color: #211f1a;
}
.stChatMessage {
    border-radius: 16px;
    padding: 4px 8px;
    background-color: #26241f;
}
h1, h2, h3, p, span, label {
    color: #e8e6df !important;
}
.stButton button {
    border-radius: 20px;
    border: 1px solid #3d3a33;
    background-color: #26241f;
    color: #e8e6df;
}
.stButton button:hover {
    border-color: #cc785c;
    color: #cc785c;
}
[data-testid="stChatInput"] {
    background-color: #26241f;
}
</style>
""", unsafe_allow_html=True)

USER_ID = "u1"  # fixed demo user -> profile is shared across all threads

if "threads" not in st.session_state:
    st.session_state["threads"] = [str(uuid.uuid4())]
if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = st.session_state["threads"][0]
if "histories" not in st.session_state:
    st.session_state["histories"] = {st.session_state["current_thread"]: []}

with st.sidebar:
    st.header("Threads")
    if st.button("+ New thread"):
        new_id = str(uuid.uuid4())
        st.session_state["threads"].append(new_id)
        st.session_state["histories"][new_id] = []
        st.session_state["current_thread"] = new_id

    st.session_state["current_thread"] = st.selectbox(
        "Switch thread",
        st.session_state["threads"],
        index=st.session_state["threads"].index(st.session_state["current_thread"]),
    )

    st.divider()
    st.header("Long-term profile")
    name_in = st.text_input("Name")
    pref_in = st.text_input("Preference (e.g. 'has acidity, avoid spicy')")
    if st.button("Save profile"):
        save_profile(USER_ID, name_in, pref_in)
        st.success("Saved — switch threads and ask a recipe question to see it recalled!")

    st.write("Stored profile:", get_profile(USER_ID))

    st.divider()
    st.header("Graph structure")
    st.code(chatbot.get_graph().draw_mermaid(), language="mermaid")

thread_id = st.session_state["current_thread"]
CONFIG = {"configurable": {"thread_id": thread_id, "user_id": USER_ID}}

st.title("🍝 Customisable Recipe Maker Chatbot")
st.caption("Tell me what you're craving, your time, energy, and any health needs — I'll customize the recipe for you.")
st.caption(f"Current thread: `{thread_id}`")

st.write("")
cols = st.columns(4)
suggestions = [
    "I have pasta cravings but acidity issues",
    "Quick 15-min dinner with what's in my fridge",
    "Low-energy meal, minimal steps",
    "Healthy dinner under 400 calories",
]
clicked_suggestion = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked_suggestion = s

st.write("")
uploaded_image = st.file_uploader("Upload a photo of your fridge/cupboard to list ingredients", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    if st.button("Detect ingredients"):
        with st.spinner("Looking at your photo..."):
            ingredients = detect_ingredients(uploaded_image.getvalue())
        st.write("**Ingredients detected:**")
        st.write(ingredients)

history = st.session_state["histories"].setdefault(thread_id, [])
for message in history:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here") or clicked_suggestion
if user_input:
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)

    ai_message = response["messages"][-1].content
    history.append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)

    trim_info = response.get("trim_info", {})
    st.caption(
        f"Messages before trim: {trim_info.get('before','?')} → after trim: {trim_info.get('after','?')} "
        f"(token budget: 800)"
    )