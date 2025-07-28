import streamlit as st
from langchain_groq import ChatGroq
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Tools setup ---
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

# --- Streamlit UI ---
st.title("LangChain - Chat with Wikipedia, Arxiv and Search")
st.write("""
This chatbot can search **Wikipedia**, **Arxiv**, and perform **web search** using DuckDuckGo.
Responses are streamed using `StreamlitCallbackHandler`.
""")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I am a chatbot who can search the web. How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input(placeholder="Ask something (e.g., What is Machine Learning?)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM (Groq HuggingFace model)
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Tools list
    tools = [wiki, arxiv, search]

    # Initialize agent
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Show assistant response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.write(response)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
