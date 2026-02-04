# Imports
import streamlit as st
import os
import pandas as pd
import time
import datetime
from streamlit_gsheets import GSheetsConnection
from st_copy import copy_button
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# This is the reset sidebar
with st.sidebar:
  st.title("🍏 Popo's Control Panel")
  st.markdown("---")

  # This is the nuclear reset button to clear UI and LangChain memory
  if st.button("🗑️ Reset Analyst Session", use_container_width=True, type="primary"):
    st.session_state.messages = []
    if 'memory' in st.session_state:
      st.session_state.memory.clear()

    st.rerun()
    
  st.caption("This resets Popo's UI and Memory")
  st.markdown("---")
  st.caption("Popo v1.0 | Powered by Llama 3 & LangChain Modular")

# Setup the Google Sheet connection
conn = st.connection('gsheets', type=GSheetsConnection)

def log_to_sheets(query, response, score, key):
  # This checks if google sheets already have this duplicate record
  if st.session_state.get(f"logged_{key}"):
    return
    
  try:
    # This returns an error but popo will continue
    existing_data = conn.read(worksheet="Feedback", ttl=0)
    new_entry = pd.DataFrame([{
      "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "User_Query": query,
      "Popo_Response": response,
      "Score": "👍" if score == 1 else "👎"
    }])
    
    updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
    conn.update(worksheet="Feedback", data=updated_df)

    st.session_state[f"logged_{key}"] = True
    st.session_state["show_feedback_toast"] = True
      
  except Exception as e:
    # This logs the simple error to the developer in their console
    print(f"Feedback system unavailable: {e}")

# Setup the environment
st.set_page_config(page_title="Popo: Apple 10-K Financial Analyst", page_icon="🍏", layout="centered")
st.title("🍏 Popo: Apple 10-K Financial Analyst")

# This is for showing the feedback toast
if st.session_state.get("show_feedback_toast"):
  st.toast("Thank you for your feedback! Popo is getting smarter. 🧠")
  del st.session_state["show_feedback_toast"]       # And then deleting them after it is done
  
st.caption("v1.0 | Powered by Llama 3 & LangChain Modular")

# Caching the Model & Database
@st.cache_resource
def init_popo():
  # This is using langchain_huggingface for the Embeddings
  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  groq_api_key=st.secrets['GROQ_API_KEY']

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
    max_retries=2,
    streaming=True      # This makes the chatbot text flowy
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
3. **Social Guardrail (MANDATORY):**: 
   * **Session Reality Check**: Look at the start of the User Query for session status headers like [NEW SESSION] to determine if context exists.
   * **No Names**: NEVER assume or invent a user name.
   * **Warmth**: Respond warmly as Popo to greetings.
   * **Tables**: Use standard Markdown tables for any financial data comparison. Every table MUST include a header row and a separator row (e.g., | Category | Value | followed by | --- | --- |) to render correctly.
4. **Context Awareness**: Use the history to handle follow-up questions accurately.
5. **Ethical Boundary**: Strictly refuse to give personal investment advice. If asked, politely explain that your expertise is limited to analyzing the facts within the Apple 10-K report and suggest the user consult a certified financial advisor.
6. **Formatting**: 
   - Use bullet points for lists.
   - **MANDATORY**: Before starting a table, always include a double newline (\n\n).
   - **MANDATORY**: For tables, start the first character (the '|') at the very beginning of the line.
   - **MANDATORY**: Never skip the separator row (`| --- | --- |`). Without it, the table will not display.
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
    output_key='answer'
  )

# # The ConversationalRetrieval chain, the Popo bot
# It assigns the Groq Llama3 as the llm, searches vector db for top 10 results
# And to follow my prompting rules instead of the default
popo_chain = ConversationalRetrievalChain.from_llm(
  llm=llm,
  retriever=vector_db.as_retriever(search_kwargs={'k': 10}),
  chain_type='stuff',
  memory=st.session_state.memory,
  combine_docs_chain_kwargs={'prompt': qa_prompt},
  return_source_documents=True,
  output_key='answer'
)

# Welcome Message from Popo
if len(st.session_state.messages) == 0:
  initial_greeting = "Hello! I'm **Popo**, your Senior Financial Analyst. I've analyzed Apple's 2025 10-K report. How can I help you with the margins, risk factors, or financial statements today?"
  st.session_state.messages.append({'role': 'assistant', 'content': initial_greeting})

# This is the Display Loop % feedback
for i, msg in enumerate(st.session_state.messages):
  current_avatar = "🍏" if msg['role'] == 'assistant' else None
  with st.chat_message(msg['role'], avatar=current_avatar):
    clean_content = "\n".join([line.lstrip() for line in msg['content'].split('\n')])
    st.markdown(clean_content)

    # This is the copy to clipboard button
    if msg['role'] == 'assistant':
      msg_key = f"copy_{i}"
      
      col1, col2 = st.columns([0.85, 0.15])
      with col2:
        copy_button(
          msg['content'],
          tooltip="📋 Copy Analysis",
          copied_label="✅ Copied!",
          icon='st',
          key=msg_key
        )

    if msg['role'] == 'assistant' and msg.get('sources'):
      with st.expander("🧐 View Analyst Sources"):
        for s_idx, doc in enumerate(msg['sources']):
          page_num = doc.metadata.get('page', 'Unknown')
          # This grabs the first 250 characters and add ... if greater than 250 else show full content
          preview_text = (doc.page_content[:250] + '...') if len(doc.page_content) > 250 else doc.page_content
          
          st.caption(f"**Source {s_idx+1} (Page {page_num}):**")
          st.write(preview_text)
          st.divider()
    
    if msg['role'] == 'assistant' and i > 0:
      log_key = f"logged_{i}"
      fb_key = f"fb_{i}"

      is_already_logged = st.session_state.get(log_key, False)
      
      with st.popover("Rate this response", icon=":material/reviews:"):
        st.write("Was this helpful?")

        feedback = st.feedback("thumbs", key=fb_key, disabled=is_already_logged)
      
        if feedback is not None and not is_already_logged:
          # This saves to Google sheet if there is feedback
          user_q = st.session_state.messages[i-1]['content']
          log_to_sheets(user_q, msg['content'], feedback, fb_key)
          st.session_state[log_key] = True
          st.toast("Thank you for your feedback! Popo is getting smarter. 🧠")

          st.rerun()

# This handles the input
prompt = None

# Suggestions for the user to choose only if fresh start
if len(st.session_state.messages) == 1:
  suggestions = [
    "📈 2025 Revenue Mix",
    "🍏 iPhone Growth",
    "🛡️ Top Risk Factors"
  ]
  prompt = st.pills("Quick Analysis:",
                    suggestions,
                    default=None
                   )

# Used st.pills for a new 2026 pill-like look for the buttons
if not prompt:
  prompt = st.chat_input("Ask Popo about Apple's 2025 margins...")

if prompt:
  if len(st.session_state.messages) <= 1:
    status_header = "[NEW SESSION - NO PRIOR CONTEXT]"
  else:
    status_header = "[CONTINUING SESSION - REFER TO HISTORY]"

  enriched_query = f"{status_header}\nUser Question: {prompt}"
  
  st.session_state.messages.append({"role": "user", "content": prompt})
  st.chat_message('user').write(prompt)

  with st.chat_message("assistant", avatar="🍏"):
    container = st.empty()
    full_response = ""

  # This is to make Popo's output flowy-looking
    try:
      sources = []        # This is a blank list for sources        
      for chunk in popo_chain.stream({
        'question': enriched_query
      }):
        if 'answer' in chunk:
          answer_chunk = chunk['answer']

          # This loops thru each character in the chunk
          for char in answer_chunk:
            full_response += char
            # This prevents popo from streaming green text
            clean_preview = "\n".join([line.lstrip() for line in full_response.split('\n')])
            container.markdown(clean_preview + "▌")
            time.sleep(0.015)

        if 'source_documents' in chunk:
          sources = chunk['source_documents']

      final_clean = "\n".join([line.lstrip() for line in full_response.split('\n')])
      container.markdown(final_clean)
       
      st.session_state.messages.append({
        'role': 'assistant',
        'content': final_clean,
        'sources': sources
      })

      # This rerun here is to refresh the Display Loop above
      st.rerun()
    
    except Exception as e:
      error_msg = str(e)
      if "429" in error_msg:
        friendly_error = "⚠️ **Rate Limit Reached**: Popo is taking a quick 60-second breather. Please try again in a moment."
      elif "401" in error_msg:
        friendly_error = "🔑 **API Key Error**: The Groq key seems invalid. Please check your Streamlit Secrets."
      else:
        friendly_error = f"😵 **Unexpected Error**: {error_msg}"

      container.error(friendly_error)
