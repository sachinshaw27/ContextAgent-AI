from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="ContextAgent AI",
    page_icon="📄",
    layout="wide"
)


# ---------------- SESSION STATE ---------------- #

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- PROCESS DOCUMENTS ---------------- #

def process_document(path):

    # Load PDFs
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    st.session_state.vector_store = vector_db

    # OpenRouter LLM
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )

    # Retrieval Tool
    @tool
    def retrieve_context(query: str):
        """
        Retrieve relevant information from uploaded PDFs.
        """

        retrieved_docs = vector_db.similarity_search(
            query=query,
            k=3
        )

        context = ""

        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"

        return context

    # Agent
    agent = initialize_agent(
        tools=[retrieve_context],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ---------------- STREAMLIT UI ---------------- #

st.title("📄 ContextAgent AI")

# Upload PDFs
if not st.session_state.document_uploaded:

    uploaded = st.file_uploader(
        label="Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing PDFs..."):

            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)

            # Save PDFs
            for file in uploaded:

                file_path = os.path.join(path, file.name)

                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # Process docs
            process_document(path)

            st.success("Documents processed successfully!")

            st.rerun()


# ---------------- CHAT UI ---------------- #

if st.session_state.document_uploaded and st.session_state.agent:

    # Chat History
    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        st.chat_message(role).markdown(content)

    # Chat Input
    query = st.chat_input(
        "Ask questions from uploaded PDFs..."
    )

    if query:

        # User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        st.chat_message("user").markdown(query)

        # AI Response
        with st.spinner("Thinking..."):

            response = st.session_state.agent.run(query)

        # Show Response
        st.chat_message("assistant").markdown(response)

        # Store Response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


# ---------------- STREAMLIT UI ---------------- #

# st.title("📄 AI PDF RAG Chatbot")
st.title("📄 ContextAgent AI")

# Upload PDFs
if not st.session_state.document_uploaded:

    uploaded = st.file_uploader(
        label="Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing PDFs..."):

            path = "./doc_files/"

            # Create folder
            os.makedirs(path, exist_ok=True)

            # Save uploaded PDFs
            for file in uploaded:

                file_path = os.path.join(path, file.name)

                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # Process docs
            process_document(path)

            st.success("Documents processed successfully!")

            st.rerun()


# ---------------- CHAT UI ---------------- #

if st.session_state.document_uploaded and st.session_state.agent:

    # Show chat history
    for message in st.session_state.messages:

        role = message["role"]
        content = message["content"]

        st.chat_message(role).markdown(content)

    # Chat input
    query = st.chat_input(
        "Ask questions from uploaded PDFs..."
    )

    if query:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        st.chat_message("user").markdown(query)

        # AI response
        with st.spinner("Thinking..."):

            response = st.session_state.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": "1"
                    }
                }
            )

            answer = response["messages"][-1].content

        # Show AI response
        st.chat_message("ai").markdown(answer)

        # Store response
        st.session_state.messages.append(
            {
                "role": "ai",
                "content": answer
            }
        )
