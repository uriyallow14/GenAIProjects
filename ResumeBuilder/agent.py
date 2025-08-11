import os
from dotenv import load_dotenv
# from typing import Annotated, Literal, List
# from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# import google.generativeai as genai
import streamlit as st
import PyPDF2 as pdf
import json


load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash-lite", model_provider="google_genai")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# llm = genai.GenerativeModel('gemini-2.0-flash-lite')

class ResumeState(MessagesState):
    resume: str
    job_desc: str
    features_extractor: str

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += str(page.extract_text())
    return text

def extract_relevant_features(state: ResumeState) -> ResumeState:
    """
    Extract structured features from a resume and job description.
    """

    prompt = f"""
    Given the following job description and resume, extract:
    - LinkedIn profile URL
    - GitHub profile URL
    - All skills from resume
    - Skills explicitly mentioned in the job description
    - Missing skills for this job

    Return the result as a valid JSON object.

    Job Description:
    {state['job_desc']}

    Resume:
    {state['resume']}
    """

    response = llm.invoke(prompt).content
    state["features_extractor"] = response
    try:
        return json.loads(response)
    except:
        return {"error": "Failed to parse LLM output", "raw_output": response}


def expert(state: ResumeState) -> ResumeState:
    system_message = f"""
        You are a resume expert. Improve the user's resume based on the job description.

        Job description: {state['job_desc']}

        Resume: {state['resume']}

        Rules:
        - Do not add skills or experiences not already in the resume.
        - Keep relevant skills.
        - Keep relevant courses.
        - Use linkedin url (if available inside the resume) in order to add relevant skills from candidate linkedin profile 
        (do not assume, check the url content).
        - Use github url (if available inside the resume) in order to add relevant projects from candidate github profile 
        (do not assume, check the url content).
        - The improved resume should be generalized in order to be used in similar jobs description or similar companies.
        - Be precise and realistic.
    """

    messages = state["messages"]
    response = llm.invoke([system_message] + messages)

    return {"messages": [response]}

graph = StateGraph(ResumeState)

tools = [extract_relevant_features]
tool_node = ToolNode(tools)

graph.add_node("extract_features", tool_node)

graph.add_edge(START, "initial_state")
graph.add_edge("initial_state", "extract_features")
graph.add_edge("extract_features", "improve_resume")
graph.add_edge("expert", END)

checkpointer = MemorySaver()
run = graph.compile(checkpointer=checkpointer)

# streamlit app
st.title("Resume Builder")
st.text("Improve Your Reusume")
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please Upload the pdf")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        resume_text = input_pdf_text(uploaded_file)
        initial_state = {
            "messages": [HumanMessage(content="Please suggest improvements to my resume.")],
            "resume": resume_text,
            "job_desc": jd
        }
        try:
            final_state = run.invoke(initial_state, config={
                                                            "configurable": {
                                                                "thread_id": "resume_builder_uy" 
                                                                            }
                                                            }
                                    )
            output_msg = final_state["messages"][-1].content
            st.subheader("Improved Resume Suggestions")
            st.write(output_msg)
        except Exception as e:
            st.error(f"Error - API failed: {e}")
