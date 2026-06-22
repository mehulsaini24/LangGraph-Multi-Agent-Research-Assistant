#  LangGraph Multi-Agent Research Assistant

A **production-style multi-agent AI system** built using **LangGraph**, **Hugging Face Inference API**, and **Tavily Search**, with a clean **Streamlit UI** and **live streaming outputs**.

This project demonstrates how to build **stateful, decision-based AI workflows** using graphs instead of linear chains.

---

## 🚀 Features

- 🧠 **Multi-Agent Architecture**
  - Planner Agent
  - Search Agent (Tavily)
  - Research Agent
  - Critic Agent (with approval / rejection)
  - Writer Agent

- 🔄 **Iterative Improvement**
  - Critic validates answers
  - Automatic retry with loop control

- 🌐 **Web-Search Grounded (RAG-style)**
  - Uses Tavily for real-time web information
  - Reduces hallucinations

- 🔴 **Live Streaming Output**
  - Research and final answer stream token-by-token

- 🎨 **Clean Streamlit UI**
  - Expandable agent logs (dropdowns)
  - Final answer highlighted in green
  - Status indicators & animations

- 📦 **Single File Project**
  - Backend + frontend in one `app.py`
  - Easy to understand, teach, and deploy

---

## 🧠 Why LangGraph?

Traditional LLM pipelines are **linear**.  
Real-world problems are **iterative**.

**LangGraph enables:**
- Conditional routing
- Loops
- Shared state
- True agent collaboration

This project showcases **how AI agents work like a team**, not a single chatbot.

---

## 🏗️ Architecture Overview

User Question
↓
Planner Agent
↓
Search Agent (Tavily)
↓
Research Agent
↓
Critic Agent
├─ Reject → Research (loop)
└─ Approve → Writer Agent
↓
Final Answer

yaml
Copy code

All agents communicate through a **shared state (`AgentState`)**.

---

## 🧩 Tech Stack

- **Python**
- **LangGraph**
- **LangChain**
- **Hugging Face Inference API**
- **Tavily Search API**
- **Streamlit**
- **dotenv**

---

## 📂 Project Structure

.
├── app.py # Full backend + frontend
├── requirements.txt # Dependencies
├── .env.example # Environment variable template
├── .gitignore
└── README.md

yaml
Copy code

---

## 🔑 Environment Variables

Create a `.env` file (do NOT push to GitHub):

```env
HUGGINGFACE_API_KEY=your_huggingface_api_key
TAVILY_API_KEY=your_tavily_api_key
Use .env.example as a reference.

📦 Installation & Setup
1️⃣ Create virtual environment
bash
Copy code
python -m venv langgra
langgra\Scripts\activate   # Windows
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the app
bash
Copy code
streamlit run app.py
🧪 Example Use Cases
Explain complex technical topics

Interview preparation

Research summarization

Content generation with validation

Demonstrating multi-agent AI systems

🎥 Demo UI Behavior
Final answer is always visible

Agent logs are hidden inside dropdowns

Live streaming shows agents working in real time

Critic feedback and retry count are visible

🧠 Learning Outcomes
By studying this project, you’ll understand:

How agents collaborate

How shared state works

How to build loops with LangGraph

How to integrate tools (Search)

How to design production-grade AI workflows

🚀 Future Enhancements
Add citations to final answers

Add retry counter progress bar

Deploy to cloud (Streamlit Cloud / Hugging Face)

Chat-based interface

Memory across sessions

📜 License
MIT License

