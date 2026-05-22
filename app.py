import streamlit as st
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="Honing Assistant", layout="wide")

st.title("🔧 Honing Troubleshooting Assistant")
st.markdown("Get expert guidance on honing issues using AI-powered troubleshooting")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to enable AI responses"
    )
    
    model_choice = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-3.5-turbo"],
        help="Choose the AI model for responses"
    )
    
    uploaded_file = "Honing Condition Matrix (v1).xlsx"
    st.info(f"📊 Using: {uploaded_file}")


@st.cache_data
def load_knowledge():
    """Load Excel file into formatted knowledge base"""
    try:
        xls = pd.ExcelFile(uploaded_file)
        all_text = []

        for sheet in xls.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            
            # Format sheet data
            text = df.fillna("").astype(str)
            combined = "\n".join(
                [" | ".join(row) for row in text.values.tolist()]
            )
            
            all_text.append(f"Sheet: {sheet}\n{combined}")

        return "\n\n".join(all_text)
    
    except FileNotFoundError:
        st.error(f"❌ File not found: {uploaded_file}")
        st.info("Please upload the Excel file to the repository root directory")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


@st.cache_data
def retrieve_relevant_sections(query, knowledge_base, num_sections=3):
    """Retrieve most relevant sections from knowledge base"""
    if not knowledge_base:
        return ""
    
    sections = knowledge_base.split("\n\n")
    
    # Simple relevance scoring (can be enhanced with embeddings)
    query_lower = query.lower()
    scored = []
    
    for section in sections:
        score = sum(1 for word in query_lower.split() if word in section.lower())
        scored.append((score, section))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([section for _, section in scored[:num_sections]])


# Load knowledge base
knowledge = load_knowledge()

if knowledge is None:
    st.warning("⚠️ Cannot proceed without knowledge base. Please check the file.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask a honing question...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not api_key:
            st.error("❌ Please enter your OpenAI API key in the sidebar")
        
        else:
            try:
                # Retrieve relevant context
                relevant_context = retrieve_relevant_sections(prompt, knowledge)
                
                system_prompt = f"""You are an expert honing engineer assistant.

Your role is to provide accurate, practical guidance on honing issues using the provided knowledge base.

IMPORTANT RULES:
- Use ONLY information from the knowledge base below
- If the knowledge base doesn't contain relevant information, say so clearly
- Provide step-by-step solutions when applicable
- Be specific and technical in your responses

Knowledge Base:
{relevant_context}
"""

                client = OpenAI(api_key=api_key)
                
                with st.spinner("🔄 Thinking..."):
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=1024
                    )

                reply = response.choices[0].message.content
                st.markdown(reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })

            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
                st.info("Check your API key and try again")
