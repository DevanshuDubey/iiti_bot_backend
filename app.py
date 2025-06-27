# app.py
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/v1/chat"

st.set_page_config(page_title="IITI BOT", page_icon="🧠")

st.title("🤖 Agent Query Router")
st.caption("Enter a query to see which specialized agent would handle it and get a response.")

user_query = st.text_input(
    "Your Query:",
    placeholder="e.g., 'Compare the CSE and EE departments'",
    help="Type your question and click 'Classify Query'",
)

if st.button("Send"):
    if not user_query:
        st.warning("Please enter a query to classify.")
    else:
        with st.spinner("🧠 Thinking... Contacting the Router Agent..."):
            try:
                payload = {"query": user_query}

                response = requests.post(BACKEND_URL, json=payload, timeout=10)

                response.raise_for_status()

                data = response.json()
                category = data.get("response", "N/A")

                st.success(f"**Agent Response:** `{category}`")
                st.write("The router agent determined this query should be handled by the agent responsible for its category.")
                
                with st.expander("Show Full API Response"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Connection Error: Could not connect to the backend service. "
                    "Is `run_agent_service.py` running in a separate terminal?"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"An API error occurred: {e}")