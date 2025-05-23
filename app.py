import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=600)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query Arxiv for academic papers")
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=600)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki, description="Query Wikipedia for general knowledge")
tavily = TavilySearchResults()
tools = [arxiv, wiki, tavily]

# Initialize LLM and bind tools
llm = ChatGroq(model="qwen-qwq-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# Define tool_calling_llm function
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build LangGraph workflow
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", END)
graph = builder.compile()

# Streamlit app
st.title("Question-Answering Research Bot")
st.write("Ask any question, and I'll answer using Arxiv, Wikipedia, or web search!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle user input
if prompt := st.chat_input("Ask your question:"):
    # Add user message to chat history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run LangGraph workflow with chat history
    state = {"messages": st.session_state.messages}
    with st.chat_message("assistant"):
        with st.spinner("Processing your question..."):
            try:
                result = graph.invoke(state)
                last_message = result["messages"][-1]
                
                # Display response
                if isinstance(last_message, AIMessage):
                    if last_message.content:
                        st.markdown(last_message.content)
                    elif last_message.tool_calls:
                        st.markdown("Fetching information...")
                        # Display tool results (last message in state)
                        tool_result = result["messages"][-1]
                        if isinstance(tool_result, ToolMessage):
                            st.markdown(tool_result.content)
                        else:
                            st.markdown("Tool invoked, but no result returned.")
                    else:
                        st.markdown("No response generated.")
                elif isinstance(last_message, ToolMessage):
                    st.markdown(last_message.content)
                else:
                    st.markdown(last_message.content)
                
                # Update chat history with bot response
                st.session_state.messages.append(last_message)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")