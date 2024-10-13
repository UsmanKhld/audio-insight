import streamlit as st
import streamlit as st
from PIL import Image
import requests
from streamlit_lottie import st_lottie

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

def uiHeroSection():
    st.set_page_config(page_title="Audio-Insights", page_icon="🤖", layout="wide")

    # Apply custom CSS for column layout with different widths
    st.write(
        """<style>
        [data-testid="column"]:nth-child(1) {
            width: 35%;  /* Left column width */
            flex: 1 1 35%;
            min-width: 35%;
            display: flex;
            align-items: flex-start;
        }
        [data-testid="column"]:nth-child(2) {
            width: 65%;  /* Right column width */
            flex: 1 1 65%;
            min-width: 65%;
            display: flex;
            align-items: flex-start;
        }
        .big-title {
        font-size: 3em;
        font-weight: bold;
        color: #FFFFFF;
        }

        .big-title span {
        color: #FFA500;
        }
    
        </style>""",
        unsafe_allow_html=True,
    )
    text_column, image_column = st.columns((2, 1)) 
    with text_column:
        st.title(' Audio Insights 🎧 ')
    with image_column:
        lottie_coding = load_lottieurl("https://lottie.host/c328b028-512c-4d26-bfb4-3eadb686289a/zidbWrQeAG.json")
        st_lottie(lottie_coding, speed=1, height=300, key="coding")
    st.markdown("---")


# def image_custom():
#     lottie_coding=load_lottieurl("https://lottie.host/f9999102-82ea-48f3-a43e-555e1b508ad2/Fh853fMk3Z.json")
#     st_lottie(lottie_coding, speed=1, height=300, key="coding")

def uiSidebarInfo():
    """
        Displays information in sidebar
    """
    st.markdown("> version 1.0.0")
    # about this app and instructions CONTAINER
    # with st.container():
    #     st.write("## ℹ️ About this app and instructions")
    #     with st.expander("Details ...", expanded=True):
    #         st.markdown(
    #             "This app uses [OpenAI](https://beta.openai.com/docs/models/overview)'s API to answer questions about your PDF file. \nYou can find the source code on [GitHub](https://github.com/virajsabhaya23/load_N_ask)."
    #         )
    #         st.markdown("1. Enter OpenAI API key.\n 2. Upload your PDF file. \n 3. Ask your question.")
    st.write("> made by [Bardan Dhakal](https://www.linkedin.com/in/bardan-dhakal/), [Izaan Khalid](https://www.linkedin.com/in/izaankhalid/), [Usman Khalid](https://www.linkedin.com/in/usmankhld/), [Zaineel Mithani](https://www.linkedin.com/in/zaineel-mithani-19588025b/)")

def uiSidebarWorkingInfo():
    with st.container():
        st.write("## ℹ️ FAQ")
        with st.expander("How does Audio Insights work?", expanded=False):
            # st.write("## How does Load and Ask work?")
            st.write(":orange[When you upload an audio file, it will be divided into smaller clips or chunks. These will be transcribed into text and then converted into embeddings, which will be stored in a vector index. A vector index is a special type of database that allows for semantic search and retrieval. Semantic search takes into account the meaning of the words in a query, rather than just the words themselves, allowing for the most relevant document chunks to be found for a given question.]")
            st.write(":orange[When you ask a question, the system will search through the document chunks and find the most relevant ones using the vector index. It will then use GPT4o to generate a final answer. GPT4o is a large language model that can generate text, translate languages, produce various types of creative content, and provide informative answers to questions.]")

