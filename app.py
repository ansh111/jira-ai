import streamlit as st
import requests
import json
from jira import JIRA
from dotenv import load_dotenv
import os
import asyncio
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

#  Constants
JIRA_MAX_DATA_LIMIT = 2500

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
HF_KEY = os.getenv("HF_TOKEN")

# Streamlit UI
st.title("Jira AI Assistant (Groq-Powered)")
st.sidebar.header("Jira Settings")
project_key = st.sidebar.text_input("Jira Project Key", "")
jira_server = st.sidebar.text_input("Jira Server Key", "")
jira_email = st.sidebar.text_input("Jira email address", "")
jira_token = st.sidebar.text_input("Jira API Token", "", type= "password")

# Function to fetch Jira issues
async def get_jira_issues(project_key):
    # Connect to Jira
    jira_options = {"server": jira_server}
    jira = JIRA(options=jira_options, basic_auth=(jira_email, jira_token))
    start_at = 0
    max_results = 100
    all_issues = []
    while True:
         query = f"project={project_key} ORDER BY created DESC"
         issues = jira.search_issues(query, startAt = start_at, maxResults=max_results)
         if not issues:
             break
         all_issues.extend(issues)
         start_at += max_results
         if len(all_issues) > JIRA_MAX_DATA_LIMIT:
             break

    documents = []
    for issue in all_issues:
        issue_text = f"""
        Issue Key: {issue.key}
        Summary: {issue.fields.summary}
        Priority: {issue.fields.priority.name}
        Labels: {', '.join(issue.fields.labels)}
        Assignee: {issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned"}
        Status: {issue.fields.status.name},
        Issue Type: {issue.fields.issuetype.name}
        Original Estimates: {getattr(issue.fields, "customfield_10047", "N/A")}
        Dev End Date: {getattr(issue.fields, "customfield_10040", "N/A")}
        QA End Date: {getattr(issue.fields, "customfield_10041", "N/A")}
        On-Call PM: {getattr(issue.fields, "customfield_10045.displayName", "N/A")}
        Region: {getattr(issue.fields, "customfield_10048", "N/A")}
        """
        documents.append(Document(page_content=issue_text.strip()))
    return documents    

prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assistant that provides detailed and well-explained answers based on the given context.
    
    <context>
    {context}
    </context>
    
    Please analyze the context thoroughly and provide a comprehensive, insightful, and well-structured response.

    Question: {input}

    Answer:
    """
)


llm=ChatGroq(groq_api_key= GROQ_API_KEY,model_name="llama3-70b-8192",max_tokens=1000)

async def generate_transcript_embeddings(project_key):
     st.session_state.docs = await get_jira_issues(project_key)
     st.write(st.session_state.docs) ## Document Loading
     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    

async def create_vector_embedding(project_key):
    if "vectors" not in st.session_state:
        st.session_state.vectors = []
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    await generate_transcript_embeddings(project_key)

if st.button("Fetch Jira Details"):
    if project_key and jira_email and jira_server and jira_token:
        asyncio.run(create_vector_embedding(project_key=project_key))
    else:
        st.write("Missing either project key or jira email or jira server or jira token please provide information")

question = st.text_input("Ask a question about Jira issues:")           

if st.button("Get Answer"):
    if project_key and question:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever(search_kwargs={"k": 100})
        retrieved_docs = retriever.get_relevant_documents(question)
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        response=retrieval_chain.invoke({'input':question})
        final_response =  response['answer']
        st.write(final_response)
    else:
        st.error("Please enter a project key and a question.")
