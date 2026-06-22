import os
import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient


# ENV SETUP
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not HUGGINGFACE_API_KEY:
    st.error("❌ HUGGINGFACE_API_KEY not found in .env")
    st.stop()

if not TAVILY_API_KEY:
    st.error("❌ TAVILY_API_KEY not found in .env")
    st.stop()


# LLM (STREAMING ENABLED)
endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    temperature=0.3,
    max_new_tokens=1024,
)
llm = ChatHuggingFace(llm=endpoint)


# TAVILY SEARCH CLIENT
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# LANGGRAPH STATE
class AgentState(TypedDict):
    question: str
    plan: NotRequired[str]
    search_results: NotRequired[str]
    research: NotRequired[str]
    critique: NotRequired[str]
    verdict: NotRequired[str]
    final_answer: NotRequired[str]
    loop_count: NotRequired[int]


# AGENTS
def planner_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Break the following question into clear research steps:\n\n{question}"
    )
    response = llm.invoke(prompt.format(question=state["question"]))
    return {
        "plan": response.content,
        "loop_count": 0
    }


def search_agent(state: AgentState):
    response = tavily_client.search(
        query=state["question"],
        search_depth="basic",
        max_results=5
    )

    results_text = ""
    for r in response["results"]:
        results_text += f"- {r['title']}: {r['content']}\n"

    return {"search_results": results_text}



def research_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """
        Question:
        {question}

        Research Plan:
        {plan}

        Web Search Results:
        {search_results}

        Use the web results for factual grounding.
        Write a detailed, accurate explanation.
        """
    )

    response = llm.invoke(
        prompt.format(
            question=state["question"],
            plan=state.get("plan", ""),
            search_results=state.get("search_results", "")
        )
    )

    return {"research": response.content}


def critic_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """
        Evaluate the following answer.

        Answer:
        {research}

        Decide if it is GOOD or BAD.

        Respond strictly in this format:
        Verdict: APPROVED or REJECTED
        Feedback: short reason
        """
    )

    response = llm.invoke(
        prompt.format(research=state.get("research", ""))
    )

    text = response.content
    verdict = "APPROVED" if "APPROVED" in text else "REJECTED"
    count = state.get("loop_count", 0) + 1

    return {
        "verdict": verdict,
        "critique": text,
        "loop_count": count
    }


def writer_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """
        Using the research below, write a clear, well-structured final answer
        suitable for beginners.

        Research:
        {research}
        """
    )

    response = llm.invoke(
        prompt.format(research=state.get("research", ""))
    )

    return {"final_answer": response.content}


# ROUTING LOGIC (VERSION-SAFE)
def route_after_critic(state: AgentState):
    # Stop looping after 3 attempts
    if state.get("loop_count", 0) >= 3:
        return "writer"

    if state.get("verdict") == "APPROVED":
        return "writer"

    return "research"


# BUILD LANGGRAPH
graph = StateGraph(AgentState)

graph.add_node("planner", planner_agent)
graph.add_node("search", search_agent)
graph.add_node("research", research_agent)
graph.add_node("critic", critic_agent)
graph.add_node("writer", writer_agent)

graph.set_entry_point("planner")

graph.add_edge("planner", "search")
graph.add_edge("search", "research")
graph.add_edge("research", "critic")

graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "writer": "writer",
        "research": "research"
    }
)

graph.add_edge("writer", END)

app_graph = graph.compile()


# STREAMLIT UI
st.set_page_config(page_title="LangGraph Research Assistant", layout="wide")

st.title("R&D Department")
st.caption("Planner → Search → Research → Critic → Writer")

user_input = st.text_area(
    "Enter your question or topic:",
    placeholder="Example: Explain LangGraph vs LangChain with real-world use cases"
)

if st.button("Run Agents"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        # Status indicator
        status = st.status("Agents are working...", expanded=False)

        # Dropdown sections
        with st.expander("Planner Agent", expanded=False):
            planner_box = st.empty()

        with st.expander("Search Agent (Tavily)", expanded=False):
            search_box = st.empty()

        with st.expander("Research Agent (Live)", expanded=False):
            research_box = st.empty()

        with st.expander("Critic Agent", expanded=False):
            critic_box = st.empty()

        # Final answer (always visible)
        final_box = st.empty()

        plan_text = ""
        search_text = ""
        research_text = ""
        critic_text = ""
        final_text = ""

        with st.spinner("Agents are collaborating..."):
            for event in app_graph.stream({"question": user_input}):
                for node, output in event.items():

                    if node == "planner" and "plan" in output:
                        plan_text = output["plan"]
                        planner_box.markdown(plan_text)

                    elif node == "search" and "search_results" in output:
                        search_text = output["search_results"]
                        search_box.markdown(search_text)

                    elif node == "research" and "research" in output:
                        research_text += output["research"]
                        research_box.markdown(research_text)

                    elif node == "critic" and "critique" in output:
                        critic_text = output["critique"]
                        critic_box.markdown(
                            f"**Attempt {output.get('loop_count')}**\n\n{critic_text}"
                        )

                    elif node == "writer" and "final_answer" in output:
                        final_text += output["final_answer"]
                        final_box.success(final_text)

        status.update(label="Agents completed successfully!", state="complete")
        st.balloons()
