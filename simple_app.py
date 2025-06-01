import streamlit as st
import json
import os

st.write("Hello, Streamlit!")

# Check if gcreds.json exists and display its content (or a message)
if os.path.exists("gcreds.json"):
    try:
        with open("gcreds.json", "r") as f:
            gcp_creds = json.load(f)
        st.write("Google Cloud credentials loaded successfully!")
        st.json(gcp_creds) # Displaying the credentials for demonstration
    except Exception as e:
        st.error(f"Error loading Google Cloud credentials: {e}")
else:
    st.warning("gcreds.json not found. Please ensure it's in the root directory.")

# You can also access secrets from .streamlit/secrets.toml like this:
# st.write(f"GCP Service Account from secrets.toml: {st.secrets['gcp_service_account']}")
