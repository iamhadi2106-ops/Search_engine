import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import wikipedia
import arxiv


load_dotenv()

st.set_page_config(
    page_title="Nova Search - ReAct Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        color: #fafafa;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    .agent-thought {
        background-color: #1e2130;
        border-left: 5px solid #00d4ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .final-answer {
        background-color: #0e3025;
        border: 1px solid #00ff88;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search--v1.png", width=80)
    st.title("Nova Search Settings")
    
    api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key to start.") or os.getenv("GROQ_API_KEY", "")
    
    st.divider()
    
    model_name = st.selectbox(
        "🧠 LLM Model",
        [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "llama-3.1-70b-versatile",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "moonshotai/kimi-k2-instruct-0905"
        ],
        index=0,
        help="Select the AI model for reasoning."
    )
    
    max_steps = st.slider(
        "🔍 Iteration Limit",
        min_value=1,
        max_value=10,
        value=5,
        help="Max steps the agent takes to find an answer."
    )
    
    st.divider()
    st.info("Agent Tools: 🌐 WebSearch | 📚 Wikipedia | 📄 Arxiv")

st.title("🚀 Nova ReAct Search Engine")
st.markdown("""
<div style='background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff;'>
    Explore the web, Wikipedia, and scientific papers through an intelligent agent that <b>thinks, acts, and learns</b>.
</div>
""", unsafe_allow_html=True)

## Tool functions

def tool_web_search(query, k=5):
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddg:
            results = ddg.text(query, region="wt-wt", max_results=k)
            if not results:
                return "No web results found."
            
            lines = []
            for r in results:
                title = r.get("title", "Untitled")
                link = r.get("href", "#")
                body = r.get("body", "No content available.")
                lines.append(f"📌 {title}\n🔗 {link}\n📄 {body}\n")
            return "\n".join(lines)
    except Exception as e:
        return f"WebSearch error: {str(e)}"

def tool_wikipedia(query, sentences=3):
    """Get summaries from Wikipedia."""
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(query, results=1)
        if not search_results:
            return "No Wikipedia page found for this query."
        
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=sentences)
        return f"📚 Wikipedia Page: {page_title}\n\n{summary}"
    except wikipedia.DisambiguationError as e:
        return f"Wikipedia Disambiguation: Several options found. Try being more specific. Options: {', '.join(e.options[:5])}"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"
    
def tool_arxiv(query):
    """Search academic papers on Arxiv."""
    try:
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(search.results())
        if not results:
            return "No Arxiv papers found."
        
        lines = []
        for paper in results:
            summary = (paper.summary or "").replace("\n", " ")[:300]
            lines.append(f"📄 title: {paper.title}\n🔗 Link: {paper.entry_id}\n📝 Summary: {summary}...")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Arxiv error: {str(e)}"
    
# ReActStyle Prompt

SYSTEM_PROMPT = """

You are a helpful research assistant with access to 3 tools:
1) Websearch 2) Wikipedia 3) Arxiv

Follow this reasoning format exactly:

Thought: what you will do next,

Action: which tool  to use (Websearch or Wikipedia or Arxiv),

Action Input: search phrase,

(Then you get an observation with the tool result.)

Repeat this loop untill can answer.
When read, write:
Final Answer: <your short, clear answer in English>

"""

ACTION_RE = re.compile(r"Action:\s*(WebSearch|Wikipedia|Arxiv)", re.I)
INPUT_RE = re.compile(r"Action Input:\s*(.*)", re.I)
FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.I | re.DOTALL)

## Simple React Agent loop
def mini_agent(client, model, question, max_iters=5):
    """Run a ReAct loop with visual feedback in Streamlit."""
    transcript = [f"User Question: {question}"]
    observation = None 

    for step in range(1, max_iters + 1):
        convo = "\n".join(transcript)
        if observation:
            convo += f"\nObservation: {observation}"

        # Visual progress using st.status
        with st.status(f"Step {step}: Reasoning...", expanded=True) as status:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": convo},
                ],
                temperature=0.1,
                max_tokens=1024
            )
            text = resp.choices[0].message.content or ""
            
            # Show the model's thought process
            st.markdown(f"**AI Thought:**")
            st.code(text, language="markdown")
            
            # Parse actions
            action = None
            action_input = None
            
            final_match = FINAL_RE.search(text)
            if final_match:
                status.update(label=f"Step {step}: Completed!", state="complete", expanded=False)
                return final_match.group(1).strip()

            action_match = ACTION_RE.search(text)
            input_match = INPUT_RE.search(text)

            if action_match and input_match:
                action = action_match.group(1).strip().capitalize()
                action_input = input_match.group(1).strip()
                
                status.update(label=f"Step {step}: Executing {action}...", state="running")
                
                # Execute tool
                if action == "Websearch":
                    observation = tool_web_search(action_input)
                elif action == "Wikipedia":
                    observation = tool_wikipedia(action_input)
                elif action == "Arxiv":
                    observation = tool_arxiv(action_input)
                else:
                    observation = f"Unknown tool: {action}"
                
                st.markdown(f"**Observation:**")
                st.info(observation[:1000] + ("..." if len(observation) > 1000 else ""))
                
                transcript.append(text)
                transcript.append(f"Observation: {observation}")
                status.update(label=f"Step {step}: Tool used!", state="complete", expanded=False)
            else:
                status.update(label=f"Step {step}: Error parsing Action", state="error")
                return "Error: Could not parse Action/Action Input from model response."

    # If max steps reached
    status.update(label="Max iterations reached. Summarizing...", state="running")
    summary = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "The user's query remains unanswered. Summarize what you found so far briefly."},
            {"role": "user", "content": "\n".join(transcript)},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return summary.choices[0].message.content

# streamlit ui - ask a question

if not api_key:
    st.info("⚠️ Please enter your **Groq API Key** in the sidebar to unlock the search engine.")
    query = st.chat_input("Search is locked... Enter API Key", disabled=True)
else:
    query = st.chat_input("Ask me anything....")

if query:
    st.chat_message("user").write(query)

    try:
        client = Groq(api_key=api_key)
        answer = mini_agent(client, model=model_name, question=query, max_iters=max_steps)
        
        st.markdown("### ✨ Final Result")
        st.markdown(f"""
            <div class='final-answer'>
                {answer}
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
