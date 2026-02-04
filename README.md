# Popo-RAG-Guardrail-AI-Chatbot
To build an Enterprise-Ready RAG(Retrieval-Augmented-Generation) Chatbot with Intent Guardrails that assist in question-answering based on its knowledge, in this case I used Apple 10K Finance Report 2025. But anyone could feed it with their own PDF document example FAQ for a customer service agent.

## 🍏Try out my Live Chatbot Here🍏:
https://popo-rag-guardrail-ai-chatbot.streamlit.app/

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

---
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

---
🚀 **How to Run**

**Option 1: Standard Installation**
1. **Clone this repo**: git clone https://github.com/popolome/Popo-RAG-Guardrail-AI-Chatbot.git

2. **Setup Environment**: Create a .env file and add your GROQ_API_KEY.

3. **Install Dependencies**: pip install -r requirements.txt

4. **Launch**: streamlit run app.py

<br>
<br>

**Option 2: Docker (Recommended for Production)**

This project is fully containerized for consistency and easy deployment.
1. **Build the Image (Bash)**:

   docker build -t popo-analyst .

<br>
   
2. **Run the Container (Bash)**:
   
   docker run -p 8501:8501 --env-file .env popo-analyst

   _Access the app at http://localhost:8501_

<br>
<br>

📊 **Setting up the Rating System**

<img width="500" height="500" alt="Google Sheets" src="https://github.com/user-attachments/assets/ddee3ead-cae0-4a13-85ff-22ac36d505f9" />

To use the Google Sheets rating feature, add the following to your .streamlit/secrets.toml (or your Cloud provider's Secrets settings):

<br>

[connections.gsheets]

**spreadsheet** = "https://docs.google.com/spreadsheets/d/your-id-here"

**type** = "service_account"

**project_id** = "..."

**private_key_id** = "..."

**private_key** = "..."

**client_email** = "..."

... (other GCP fields)

---
🚢 **Deployment & Architecture**

Popo is designed to be platform-agnostic. While a live version is hosted on Streamlit Cloud, the project is fully containerized using Docker for production-grade deployment on Render. Project successfully tested on Render via Docker. Optimized for 1GB+ RAM environments; currently suspended to manage resources.

## Proof of Docker Deployment via Render

<img width="500" height="500" alt="Render Successful Docker Deployment 1" src="https://github.com/user-attachments/assets/07f0a9ee-bbce-416b-9e5f-9306448c59c1" />

<img width="500" height="500" alt="Render Successful Docker Deployment 2" src="https://github.com/user-attachments/assets/45a7784a-8ee2-43ef-8d74-67fb6c28ae10" />

<br>
<br>
<br>

However, I ran out of memory (OOM) due to Render's free tier of 512MB RAM (see images below). But, I still managed to containerize it and deployed it to production (cloud environment), still a successful job. Anyways, I already have working Popo on streamlit cloud (link on top).

<img width="500" height="500" alt="Render Successful Docker Deployment 3" src="https://github.com/user-attachments/assets/37bdd64e-0834-4554-a4e6-f5422600cf70" />

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/6c339225-7213-4795-84c5-c77bcbbe362f" />

<br>
<br>
<br>

📴 **Decommissioning**

I then decommissioned the Docker on Render after testing is done.

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4988e0cc-e7a1-43a1-a08f-98bd8878364f" />


<br>
<br>

🐳 **Why Docker?**

**Reproducibility**: Ensures the RAG pipeline (LangChain + ChromaDB) runs identically in local and cloud environments.

**Scalability**: The image can be deployed to any cloud provider (AWS, GCP, Azure) without code changes.

**Security**: System-level dependencies are isolated within the container.

<br>

🛠️ **Deploying to Render**

**Connect GitHub**: Linked this repository to a Render Web Service.

**Docker Runtime**: Render detects the Dockerfile and builds the image automatically.

**Secrets Management**: API keys (Groq, Google Sheets) are handled via Render's Environment Variables, keeping the code secure.

---
🌠 **Future Improvements**
* For production-scale deployment, this container is ready to be orchestrated via Kubernetes to handle high-concurrency financial queries.

---
📝 **Key Notes from Me**:
* I stumbled across Scenario from a email newsletter from Susan Shu Chang (Principal Data Scientist).
* I saw a chatbot at the web, so I tested it out, found out it was a good chatbot.
* Researched on how to create something similar and great like theirs.
* Used Colab to build the LLM with Llama-3 and Groq.
* I named it Popo, my online gaming nickname.
* Re-fined Popo's Prompt and added a few functionalities like reset, rating, memory, formating, etc.
* Spent like a week or more and finally deem it "great enough".
* Built a Docker for Popo, but ran out of memory(OOM) on Render.
* Popo requires more RAM if want to deploy it.
* Popo is also able to be trained on other corpus like customer service FAQ, just feed him with it and adjust his prompts.
* Anyways, already have a working Popo on Streamlit Cloud.
* Decommisioned Popo from Render after deployment testing.
* To be honest, if Groq API was not free, I may not have Popo.
* It was a great and semi-tough experience building it from ground up.
* This is a portfolio for my Data Scientist dream.

<br>

That's all from me.

<br>

Yours Truly,

Jun Kit Mak
