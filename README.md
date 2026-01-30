# Popo-RAG-Guardrail-AI-Chatbot
To build an Enterprise-Ready RAG(Retrieval-Augmented-Generation) Chatbot with Intent Guardrails that assist in question-answering based on its knowledge, in this case I used Apple 10K Finance Report 2025. But anyone could feed it with their own PDF document example FAQ for a customer service agent.

## 🍏Try out my Live Chatbot Here🍏:
https://popo-rag-guardrail-chatbot.streamlit.app/

<img width="500" height="500" alt="popo-rag-guardrail-ai-chatbot" src="https://github.com/user-attachments/assets/f104e5ff-4965-4f92-bbe0-0ff7960618ab" />


<br>
<br>

I was inspired by production-grade bots like [Scenario's AI](https://www.scenario.com/), I implemented a custom prompt-engineering layer to ensure the model remains grounded. If a user provides 'gibberish' or off-topic prompts, the system is designed to gracefully redirect them back to the documentation.

<br>
<br>

**Scenario AI Chatbot 1**

<img width="500" height="500" alt="Scenario AI Chatbot 1" src="https://github.com/user-attachments/assets/b9bb649e-0b62-49e4-8bbf-6a8163c7097f" />

<br>
<br>

**Scenario AI Chatbot 2**

<img width="500" height="500" alt="Scenario AI Chatbot 2" src="https://github.com/user-attachments/assets/d1fca5ea-b6ac-4901-a6a3-1237b88e261f" />

<br>
<br>
<br>

# 📈 Popo: Enterprise-Ready Apple 10-K Analyst
Popo is a specialized RAG (Retrieval-Augmented Generation) assistant designed to analyze Apple Inc.'s 2025 Fiscal 10-K filings with professional-grade precision and ethical guardrails.

<br>
<br>

🛠️ **Tech Stack**
* LLM: Llama 3 (70B) via Groq for high-speed inference.

* Orchestration: LangChain (Conversational Retrieval Chain).

* Vector Database: ChromaDB for persistent financial context.

* Embeddings: HuggingFace all-MiniLM-L6-v2.

* Frontend: Streamlit for a flowy, real-time chat experience.

<br>
<br>

🛡️ **Key Features**
* Intent Guardrails: Automatically identifies and refuses personal investment advice.

* Contextual Memory: Handles complex follow-up questions about specific fiscal years.

* Data Integrity: Restricts answers strictly to the provided 10-K document to prevent hallucinations.

<br>
<br>

🚀 **How to Run**
1. Clone this repo.

2. Unzip apple_chroma_db_export.zip.

3. Add your GROQ_API_KEY to a .env file.

4. Run pip install -r requirements.txt.

5. Launch with streamlit run app.py.
