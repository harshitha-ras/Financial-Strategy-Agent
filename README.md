# AI Financial Strategist Agent 📈


This project implements a sophisticated AI agent designed to act as a financial strategist for a business. By synthesizing internal company data, competitor SEC filings, and real-time market trends, the agent provides actionable strategic recommendations to address specific business challenges like margin compression.

The agent uses a robust two-step "Researcher → Strategist" architecture to first gather high-quality data and then formulate a concrete, data-driven plan.

![alt text](/UI_1.png)
![alt text](/UI_2.png)

## 🤖 Agentic Architecture: A Two-Stage Approach

![alt text](/Architecture.png)

To ensure accuracy and prevent premature conclusions, the agent operates in two distinct stages: Data Foraging and Strategy Synthesis. This separation of concerns is critical for producing high-quality, reliable output.

Stage 1: The Researcher Agent (Data Foraging)
The first stage is handled by a ReAct Agent. "ReAct" stands for Reasoning and Acting, meaning the agent iteratively thinks about its goal, selects the best tool, executes an action, and observes the outcome to inform its next step. The Researcher's sole purpose is to gather raw, unbiased data from multiple sources in response to the user's query. It is explicitly instructed not to perform analysis or draw conclusions.

The Researcher has access to a dynamic toolkit:

- SEC Filings Tool  Filing:

    - Function: Dynamically created for each stock ticker (e.g., (F), (GM)) found in the prompt.

    - Data Source: It targets Item 7: Management's Discussion and Analysis (MD&A) from the company's most recent 10-K filing, accessed via the sec-api.io service.

    - Strategic Value: The MD&A provides a direct narrative from the company's management about its financial performance, capital allocation, and key risk factors, offering insights that raw numbers alone cannot.

- Web Search Tool 🌐:

    - Function: Provides the ability to conduct real-time, optimized web searches.

    - Data Source: Powered by the Tavily Research API.

    - Strategic Value: This tool is crucial for capturing timely information that doesn't exist in periodic SEC filings, such as recent market trends, breaking news, competitive analysis, and emerging technologies (e.g., "market trends for EV charging infrastructure").

- Internal Data RAG Tool 📄:

    - Function: Performs Retrieval-Augmented Generation (RAG) on a private document provided by the user (.pdf, .docx, .txt).

    - Data Source: The user's uploaded file.

    - Strategic Value: Allows the agent to ground its research in the user's specific context, such as internal sales figures, operational metrics ("margin compression"), or proprietary reports. This connects external market data with the user's internal reality.

Stage 2: The Strategist LLM (Strategy Synthesis)
Once the Researcher Agent has gathered and presented all the relevant facts, its output is passed to the second stage.

- Function: This stage uses a powerful LLM (gpt-4o) with a carefully engineered prompt that instructs it to act as a "world-class financial strategist."

- Process: It receives the user's original problem and the complete, factual research summary from Stage 1. Its task is not to re-summarize the data, but to synthesize it. It must connect the dots between the various data points (internal metrics, competitor strategy, market trends) to develop a set of concrete, forward-looking strategic recommendations.

- Strategic Value: This separation ensures that the final strategy is directly and exclusively derived from the evidence collected in the first stage. It mitigates the risk of LLM hallucination and produces a final output that is both creative and data-grounded.

✨ Core Features
- Dynamic Toolbelt: The agent adapts its capabilities on the fly by creating specialized tools based on the user's specific query.

- Multi-Source Triangulation: It makes informed decisions by combining and cross-referencing data from official public filings (SEC), the live web (Tavily), and private user documents (RAG).

- Evidence-Based Reasoning: The two-stage design forces the final recommendations to be built upon a solid foundation of retrieved facts, enhancing the reliability of the output.

- Separation of Concerns: By cleanly separating the tasks of data gathering and strategic analysis, the agent minimizes bias and improves the quality of both steps.



## ⚙️ Setup and Installation

1. Install Dependencies
This project uses several Python libraries. You can install them all with the following command:

``` pip install -r requirements.txt ```

2. Launch the Interface from your terminal:
``` streamlit run app.py ```

3. Enter API Keys: In the sidebar, paste your API keys for OpenAI, Tavily, and SEC-API.io.

4. Upload Internal Data: If relevant, upload a .pdf, .docx, or .txt file for the agent to use as context.

5. Submit Your Query: Write your financial query in the main text area. Important: For the agent to create its SEC filing tool, you must enclose company tickers in parentheses, e.g., ...strategies of Ford (F) and General Motors (GM)....

6. Execute the Agent: Click the "Generate Strategy" button and observe the agent's progress.
