import streamlit as st
import requests

# BACKEND_URL = "http://backend:8000"
BACKEND_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="Agentic Shopping", layout="centered")

st.title("🛍️ Agentic Shopping Assistant")

# --------------------------
# Initialize Session
# --------------------------

if "session_id" not in st.session_state:
    response = requests.post(f"{BACKEND_URL}/session")
    st.session_state.session_id = response.json()["session_id"]
    st.session_state.chat_history = []

# --------------------------
# Display Chat History
# --------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------
# User Input
# --------------------------

if prompt := st.chat_input("Describe what you need..."):
    
    # Add user message locally
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Send to backend
    response = requests.post(
        f"{BACKEND_URL}/chat/{st.session_state.session_id}",
        json={"message": prompt}
    )

    if response.status_code == 200:
        data = response.json()

        assistant_message = data["response"]

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        with st.chat_message("assistant"):
            st.markdown(assistant_message)

        # Show extracted constraints
        with st.expander("🔍 Current Constraints"):
            st.json(data["constraints"])

        # If satisfied → show reset button
        if data["satisfied"]:
            st.success("Session completed successfully!")

            if st.button("Start New Session"):
                del st.session_state.session_id
                del st.session_state.chat_history
                st.rerun()

    else:
        st.error("Backend error occurred.")
