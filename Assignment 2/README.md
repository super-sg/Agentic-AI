# 🔬 Autonomous Research Agent (LangChain)

An AI-powered agent that automatically researches any topic and generates structured reports using **LangChain**, **Google Gemini**, and multiple research tools.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Research Agent (ReAct)              │
│         powered by Google Gemini LLM            │
├────────────────────┬────────────────────────────┤
│   🌐 Web Search   │   📚 Wikipedia Knowledge   │
│   (DuckDuckGo)    │   (WikipediaQueryRun)      │
├────────────────────┴────────────────────────────┤
│            📝 Report Generator                  │
│         (Markdown structured output)            │
└─────────────────────────────────────────────────┘
```

**Agent Type:** ReAct (Reasoning + Acting)  
**LLM:** Google Gemini 2.0 Flash  
**Tools:**
- 🌐 **Web Search** — DuckDuckGo (real-time web results, no API key needed)
- 📚 **Wikipedia** — Structured knowledge base for foundational information

## 📦 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Google API key
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

## 🚀 Usage

### Command Line

```bash
# Pass topic as argument
python research_agent.py "Impact of AI in Healthcare"

# Interactive mode
python research_agent.py
```

### Output

The agent will:
1. 🔍 Search the web for current information
2. 📚 Query Wikipedia for foundational knowledge
3. 🧠 Analyze and synthesize all findings
4. 📝 Generate a structured Markdown report in `sample_outputs/`

## 📄 Report Format

Each generated report includes:

| Section | Description |
|---------|-------------|
| **Cover Page** | Title, date, tools used |
| **Introduction** | Background and context |
| **Key Findings** | 5-7 major research findings |
| **Challenges** | Current issues and concerns |
| **Future Scope** | Emerging trends and predictions |
| **Conclusion** | Summary and balanced perspective |
| **References** | Sources consulted |

## 📁 Project Structure

```
Assignment 2/
├── research_agent.py      # Main agent entry point
├── tools.py               # Tool definitions (Search + Wikipedia)
├── report_generator.py    # Report formatting and output
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
├── README.md              # This file
└── sample_outputs/        # Generated reports
    ├── impact_of_ai_in_healthcare.md
    └── quantum_computing_current_state_and_future_prospects.md
```

## 🛠️ Tech Stack

- **LangChain** — Agent framework and tool orchestration
- **Google Gemini 2.0 Flash** — Large Language Model
- **DuckDuckGo Search** — Web search tool (free, no API key)
- **Wikipedia API** — Knowledge retrieval tool
- **Python dotenv** — Environment variable management

## 📊 Sample Outputs

Two sample reports are included in the `sample_outputs/` directory:

1. **Impact of AI in Healthcare** — Explores how artificial intelligence is transforming diagnostics, treatment, and healthcare delivery.
2. **Quantum Computing: Current State and Future Prospects** — Examines the current landscape and future potential of quantum computing technology.
