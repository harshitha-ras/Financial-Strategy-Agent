# AI Financial Strategist Agent 📈


This project implements a sophisticated AI agent designed to act as a financial strategist for a business. By synthesizing internal company data, competitor SEC filings, and real-time market trends, the agent provides actionable strategic recommendations to address specific business challenges like margin compression.

The agent uses a robust two-step "Researcher → Strategist" architecture to first gather high-quality data and then formulate a concrete, data-driven plan.

## Core Features
Dynamic Competitor Analysis: Automatically detects stock tickers (e.g., (F), (GM)) in your prompt and creates a specific analysis tool for each company's latest 10-K filing using the sec-api.

Internal Data Integration: Reads and analyzes your company's private operational data from a local file to understand your specific financial situation.

Real-Time Market Research: Uses the Tavily search API to gather the latest industry trends, news, and market projections relevant to your query.

Two-Step "Researcher → Strategist" Logic: A powerful architecture that first uses a "Researcher" agent to gather facts and then feeds that analysis to a "Strategist" AI to ensure a high-quality, actionable final plan.



## ⚙️ Setup and Installation

1. Install Dependencies
This project uses several Python libraries. You can install them all with the following command:

!pip install -U --upgrade llama-index llama-index-llms-openai llama-index-agent-openai \
               llama-index-readers-google llama-index-tools-tavily-research sec-api \
               google-api-python-client pydrive2 llama-index-core docx2txt

2. Configure API Keys
The agent requires three API keys to function, you'll need these keys:

- OpenAI API
- Tavily API
- SEC API

## ▶️ How to Run the Agent
- Step 1: Prepare Your Internal Data File
Create a simple text file (e.g., Q3_Financial_Review.txt) containing your company's key metrics. The agent is designed to read this file for context.

Example Q3_Financial_Review.txt:

```
Subject: Q3 2025 Performance Review
  
Our Gross Profit Margin dropped from 48.9% in Q2 to 45.2% in Q3.
This was primarily driven by a 15% price increase in cobalt and an 8% increase in semiconductor costs.
Inventory days remain stable at 55 days.
```

- Step 2: Upload Your Data File
If you are using an environment like Google Colab, upload your .txt file to the session storage (usually the /content/ directory). The script is currently configured to look for the file there.

- Step 3: Craft Your Prompt
Modify the prompt in the if __name__ == "__main__": block at the bottom of the script. For the best results, structure your prompt with three parts:
```
State your problem: "Based on my company's recent margin compression..."

Specify competitors: "...compare the strategies of Ford (F) and General Motors (GM)..." (ensure tickers are in parentheses).

Ask for market trends: "...what are the current market trends for EV charging infrastructure?"
```
- Step 4: Execute the Script

The agent will begin its two-step process, printing its reasoning, the research it finds, and finally, the strategic recommendations.

