import streamlit as st

# Function to display the chat history
def display_chat_history():
    chat_history = st.session_state.chat_history

    if chat_history:
        st.subheader("Chat History")
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                st.write(message.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(message.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Run the display_chat_history function
display_chat_history()
