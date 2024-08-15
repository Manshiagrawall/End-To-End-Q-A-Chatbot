import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr

# Define custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #333;
        color: #fff;
    }
    .sidebar .sidebar-content .element-container {
        color: #fff;
    }
    .stTextInput input {
        border-radius: 5px;
        padding: 10px;
        border: 1px solid #ccc;
    }
    .stSlider > div > div {
        border-radius: 5px;
        background-color: #eee;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens, openai_api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Initialize session state to store conversation history
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit app layout
st.title("Enhanced Q&A Chatbot With OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-3.5-turbo"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("### Go ahead and ask any question")
user_input = st.text_input("You:")

if st.button("Record"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.text_input("You:", value=user_input)
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Sorry, there was an issue with the speech recognition service.")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.session_state.history.append((user_input, response))
    st.write("### Response:")
    st.write(response)

    # Display conversation history
    st.write("### Conversation History:")
    for q, r in st.session_state.history:
        st.write(f"**You:** {q}")
        st.write(f"**Assistant:** {r}")

    # Export chat button
    if st.button("Export Chat"):
        chat_history = "\n".join([f"You: {q}\nAssistant: {r}" for q, r in st.session_state.history])
        st.download_button("Download Chat", chat_history, file_name="chat_history.txt")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.history = []
elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar.")
else:
    st.write("Please provide the user input.")
