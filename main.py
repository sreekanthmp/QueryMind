import streamlit as st
from config.enums import FAILURE_MESSAGES
from src.rag.rag_chain import RAGChain
from src.vectorstore.load_vectorstore import LoadVectorStore


class DocumentSearchApp:
    def __init__(self):
        # Initialize session state variables
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello, how can I assist you?"}]
        st.session_state.setdefault("vectorstore", LoadVectorStore())
        st.session_state.setdefault("rag_chain", RAGChain())

    def display_header(self):
        st.set_page_config(page_title="Chat Bot", page_icon=":robot_face:")
        st.markdown(
            "<h1 style='text-align: center;'>Document Search</h1>", unsafe_allow_html=True)

    def display_messages(self):
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    def handle_user_input(self):
        user_prompt = st.chat_input()
        if not user_prompt:
            return

        st.session_state.messages.append(
            {"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        with st.chat_message("assistant"):
            response_container = st.empty()
            streamed_response = []

            with st.spinner("Loading..."):
                status, chunks = st.session_state.rag_chain.rag_chain(
                    user_prompt)
                if not status:
                    failure_message = FAILURE_MESSAGES[0]
                else:
                    chunk_generator, _ = chunks
                    for chunk in chunk_generator:
                        streamed_response.append(chunk)
                        response_container.markdown("".join(streamed_response))
                        if "no response" in "".join(streamed_response).lower():
                            failure_message = FAILURE_MESSAGES[0]
                            break
                    else:
                        failure_message = None

            full_response = failure_message or "".join(streamed_response)
            response_container.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response})

    def run(self):
        self.display_header()
        self.display_messages()
        self.handle_user_input()


if __name__ == "__main__":
    app = DocumentSearchApp()
    app.run()
