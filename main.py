import csv
from datetime import datetime
import time
import streamlit as st
from config.enums import FAILURE_MESSAGES, Constants
from src.rag.rag_chain import RAGChain
from src.vectorstore.load_vectorstore import LoadVectorStore
from streamlit_star_rating import st_star_rating


class DocumentSearchApp:
    def __init__(self):
        # Initialize session state variables
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello, how can I assist you?"}]
        st.session_state.setdefault("vectorstore", LoadVectorStore())
        st.session_state.setdefault("rag_chain", RAGChain())
        if "pending_interaction" not in st.session_state:
            st.session_state.pending_interaction = None
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        self.csv_file = Constants.CSV_FILE.value
        self.initialize_csv()

    def initialize_csv(self):
        try:
            with open(
                    self.csv_file, mode="a+",
                    newline="", encoding="utf-8") as file:
                file.seek(0)
                if not file.read(1):
                    writer = csv.writer(file)
                    writer.writerow(
                        ["Timestamp", "Question", "Answer", "Rating"])
        except Exception as e:
            print("Error initializing CSV file:", e)

    def save_to_csv(self, timestamp, question, answer, rating):
        try:
            with open(
                    self.csv_file, mode="a",
                    newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, question, answer, rating])
                print("Data saved to CSV successfully.")
        except Exception as e:
            print("Error saving to CSV:", e)

    def display_header(self):
        st.set_page_config(page_title="Chat Bot", page_icon=":robot_face:")
        st.markdown(
            "<h1 style='text-align: center;'>Document Search</h1>", unsafe_allow_html=True)

    def display_sidebar(self):
        with st.sidebar:
            st.subheader("Available Pages")
            all_pages = st.session_state.vectorstore.get_all_documents()
            if all_pages:
                seen_urls = set()
                for i, metadata in enumerate(all_pages):
                    url = metadata.get("url", None)
                    if url not in seen_urls:
                        seen_urls.add(url)
                        page_name = metadata.get(
                            "title") or f"Document {i + Constants.ONE.value}"
                        st.markdown(f"[{page_name}]({url})")
            else:
                st.write("No pages found in the database.")

    def display_messages(self):
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    def handle_user_input(self):
        user_prompt = st.chat_input()
        if not user_prompt:
            return
        if st.session_state.pending_interaction:
            interaction = st.session_state.pending_interaction
            interaction["rating"] = "N/A"
            self.save_to_csv(**interaction)
            st.session_state.pending_interaction = None
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
                    # Get last two interactions
                    history = st.session_state.conversation_history[-2:]

                    # Only retry if there's history available
                    if history:
                        retry_prompt = (
                            f"Previous context: {history}. "
                            f"New question: '{user_prompt}'. "

                        )
                        status, chunks = st.session_state.rag_chain.rag_chain(
                            retry_prompt)
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
            st.session_state.conversation_history.append(
                            full_response.lower().replace(user_prompt, "", 1))
        st.session_state.conversation_history = (
            st.session_state.conversation_history[-2:]
        )
        st.session_state.pending_interaction = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": user_prompt,
                "answer": full_response,
                "rating": None
            }

    def submit_feedback(self, rating):
        if rating > (
                Constants.ZERO.value and st.session_state.pending_interaction):
            interaction = st.session_state.pending_interaction
            interaction["rating"] = rating
            self.save_to_csv(**interaction)
            st.session_state.pending_interaction = None
            success = st.success("Thank you for your feedback!")
            time.sleep(3)
            success.empty()
        else:
            st.warning("Please provide a valid rating.")

    def display_feedback_form(self):
        if st.session_state.pending_interaction:
            rating = st_star_rating(
                label="Rate your experience:", maxValue=5, defaultValue=0)

            if rating and rating > 0:  # Ensure a valid rating is selected
                if "feedback_submitted" not in st.session_state:
                    st.session_state.feedback_submitted = False

                if not st.session_state.feedback_submitted:
                    self.submit_feedback(rating)
                    st.session_state.feedback_submitted = True

    def run(self):
        self.display_header()
        self.display_sidebar()
        self.display_messages()
        self.handle_user_input()
        self.display_feedback_form()


if __name__ == "__main__":
    app = DocumentSearchApp()
    app.run()