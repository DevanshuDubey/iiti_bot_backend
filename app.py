# app.py
import streamlit as st
import requests

# --- Configuration ---
BACKEND_URL = "http://localhost:8080/v1/respond"

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Agent Router UI", page_icon="🧠")

st.title("🤖 Agent Query Router")
st.caption("Enter a query to see which specialized agent would handle it.")

# --- User Input ---
user_query = st.text_input(
    "Your Query:",
    placeholder="e.g., 'Compare the CSE and EE departments'",
    help="Type your question and click 'Classify Query'",
)

# --- Button and Backend Interaction ---
if st.button("Send"):
    if not user_query:
        st.warning("Please enter a query to classify.")
    else:
        with st.spinner("🧠 Thinking... Contacting the Router Agent..."):
            try:
                # Prepare the data payload for the POST request
                payload = {"query": user_query}

                # Send the request to the Pathway backend service
                response = requests.post(BACKEND_URL, json=payload, timeout=10)

                # Raise an exception for bad status codes (4xx or 5xx)
                response.raise_for_status()

                # Parse the JSON response
                data = response.json()
                category = data.get("response", "N/A")

                # Display the result
                st.success(f"**Agent Response:** `{category}`")
                st.write("The router agent determined this query should be handled by the agent responsible for its category.")
                
                # Show the full JSON response for debugging/transparency
                with st.expander("Show Full API Response"):
                    st.json(data)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Connection Error: Could not connect to the backend service. "
                    "Is `run_agent_service.py` running in a separate terminal?"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"An API error occurred: {e}")