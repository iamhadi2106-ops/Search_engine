# 🚀 Nova ReAct Search Engine

Nova is an intelligent search engine agent that uses the **ReAct (Reason -> Act)** framework to research complex queries across the web, Wikipedia, and academic papers through Arxiv. 

Powered by **Groq's** LPU technology, Nova provides near-instant reasoning and responses using the latest LLMs like Llama 3.3 and DeepSeek-R1.

![Nova Search UI](https://img.icons8.com/fluency/96/search--v1.png)

## ✨ Features

- **Intelligent Reasoning**: Uses a manual ReAct loop (Think -> Act -> Observe) to ensure accuracy and transparency in its research.
- **Multi-Tool Access**:
  - 🌐 **WebSearch**: Real-time information via DuckDuckGo.
  - 📚 **Wikipedia**: In-depth summaries of known topics.
  - 📄 **Arxiv**: Search and summarize scientific papers.
- **Premium UI**: Modern dark-themed Streamlit interface with real-time status updates and styled response cards.
- **Security First**: Search is locked until a valid Groq API key is provided, protecting your usage.
- **Model Flexibility**: Choose between several powerful models like `llama-3.3-70b-versatile` and `deepseek-r1-distill-llama-70b`.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/nova-search.git
   cd nova-search
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_key_here
   ```

## 🚀 Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Enter your API Key in the sidebar if not set in `.env`, and start searching!

## 📜 How it Works

Nova follows the **ReAct** pattern:
1. **Thought**: The model decides what information it needs.
2. **Action**: It selects a tool (WebSearch, Wikipedia, or Arxiv) to get that information.
3. **Action Input**: It provides the specific search query for the tool.
4. **Observation**: It reads the raw data returned by the tool.
5. **Final Answer**: Once it has enough information, it synthesizes the final response for you.

## ⚖️ License

MIT License - feel free to use and modify for your own projects!
