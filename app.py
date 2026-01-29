# Imports
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# Setup the environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Popo: Apple 10-K Financial Analyst", page_icon="🍏", layout="centered")
st.title("🍏 Popo: Apple 10-K Financial Analyst")
st.caption("v1.0 | Powered by Llama 3 & LangChain Modular")

# Caching the Model & Database
@st.cache_resource
def init_popo():
  # This is using langchain_huggingface for the Embeddings
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

  # This is using langchain_chroma for the Vector storing
  vector_db = Chroma(
    persist_directory="apple_chroma_db",
    embedding_function=embeddings
  )

  # This is using langchain_groq for the thinking/processing
  llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0,
    max_tokens=1000,
    max_retries=2
  )

  return vector_db, llm

vector_db, llm = init_popo()

# Perform Prompt Engineering
template = """
### ROLE
You are Popo, a Senior Financial Analyst specializing in Apple Inc. Your tone is professional, objective, and precise.

### INSTRUCTIONS
1. **Scope Control**: Use ONLY the provided context and chat history. If the information isn't there, say: "I'm sorry, I only have the ability to answer questions about the provided Apple 10-K report."
2. **Precision**: Always specify the exact fiscal year (e.g., 'In fiscal year 2025...'). "When reporting financial metrics, prioritize the 'Fiscal Year' totals over 'Three Months Ended' figures. If the context contains both, explicitly state whether you are providing a quarterly or an annual figure."
3. **Social Guardrail**: If the user greets you or says 'thank you', respond warmly as Popo and offer to assist with further analysis of the 10-K. "If the answer involves a table or multiple metrics, use Markdown tables to present the data clearly."
4. **Context Awareness**: Use the history to handle follow-up questions accurately.
5. **Ethical Boundary**: Strictly refuse to give personal investment advice. If asked, politely explain that your expertise is limited to analyzing the facts within the Apple 10-K report and suggest the user consult a certified financial advisor.
6. **Formatting**: Use bullet points for lists of risks or financial metrics to improve readability.
7. Identity: Do not assume the user's name or identity. Address the user respectfully as "User" or simply dive into the analysis unless they explicitly introduce themselves.
8. "If asked for a specific product margin (like iPhone or Mac), remind the user that Apple only reports disaggregated margins for 'Products' and 'Services'. Do not attempt to calculate a product margin by dividing product revenue by total revenue, as that represents revenue mix, not profitability."

### CONTEXT
{context}

### CHAT HISTORY
{chat_history}

### USER QUERY
{question}

### POPO's ANALYSIS:
"""

qa_prompt = PromptTemplate(
  template=template,
  input_variables=['context', 'chat_history', 'question']
)

# This is handling Session State of Streamlit
if 'messages' not in st.session_state:
  st.session_state.messages = []
if 'memory' not in st.session_state:
  st.session_state.memory = ConversationBufferWindowMemory(
    k=3,
    memory_key='chat_history',
    return_messages=True,
    output_keys='answer'
  )

# Welcome Message from Popo
if len(st.session_state.messages) == 0:
  initial_greeting = "Hello! I'm **Popo**, your Senior Financial Analyst. I've analyzed Apple's 2025 10-K report. How can I help you with the margins, risk factors, or financial statements today?"
  st.session_state.messages.append({'role': 'assistant', 'content': initial_greeting})

for msg in st.session_state.messages:
  st.chat_message(msg['role']).write(msg['content'])

# # The ConversationalRetrieval chain, the Popo bot
# It assigns the Groq Llama3 as the llm, searches vector db for top 10 results
# And to follow my prompting rules instead of the default
popo_chain = ConversationalRetrievalChain.from_llm(
  llm=llm,
  retriever=vector_db.as_retriever(search_kwargs={'k': 10}),
  chain_type='stuff',
  memory=st.session_state.memory,
  combine_docs_chain_kwargs={'prompt': qa_prompt},
  return_source_documents=False
)

# This is for making the Chat Interface
if prompt := st.chat_input("Ask Popo about Apple's 2025 margins..."):
  st.session_state.messages.append({'role': 'user', 'content': prompt})
  st.chat_message('user').write(prompt)

  with st.chat_message('assistant'):
    container = st.empty()
    full_response = ""

    # This is to make Popo's output flowy-looking
    try:
      for chunk in popo_chain.stream({'question': prompt}):
        if 'answer' in chunk:
          full_response += chunk['answer']
          container.markdown(full_response + "▌")

      container.markdown(full_response)
      st.session_state.messages.append({'role': 'assistant', 'content': full_response})

    except Exception as e:
      error_msg = str(e)
      if "429" in error_msg:
        friendly_error = "⚠️ **Rate Limit Reached**: Popo is taking a quick 60-second breather. Please try again in a moment."
      elif "401" in error_msg:
        friendly_error = "🔑 **API Key Error**: The Groq key seems invalid. Please check your Streamlit Secrets."
      else:
        friendly_error = f"😵 **Unexpected Error**: {error_msg}"

      container.error(friendly_error)
