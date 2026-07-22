import re
import streamlit as st

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "input_page"
if "transcript_ready" not in st.session_state:
    st.session_state.transcript_ready = False
if "summary_generated" not in st.session_state:
    st.session_state.summary_generated = False
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "extractive_summary" not in st.session_state:
    st.session_state.extractive_summary = ""
if "extractive_ready" not in st.session_state:
    st.session_state.extractive_ready = False
if "abstractive_summary" not in st.session_state:
    st.session_state.abstractive_summary = ""
if "abstractive_ready" not in st.session_state:
    st.session_state.abstractive_ready = False
if "results_page_ready" not in st.session_state:
    st.session_state.results_page_ready = False

YOUTUBE_REGEX = r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=|embed/|v/|e/|watch\?v%3D|)?([a-zA-Z0-9_-]{11})(.*)?$"


def reset_summary_state():
    st.session_state.summary_generated = False
    st.session_state.summary = ""
    st.session_state.model_ready = False
    st.session_state.extractive_summary = ""
    st.session_state.extractive_ready = False
    st.session_state.abstractive_summary = ""
    st.session_state.abstractive_ready = False
    st.session_state.results_page_ready = False


def fetch_transcript(video_url):
    try:
        from transcript.transcript_api import yt_transcript_api

        transcription = yt_transcript_api(video_url)
        if transcription:
            st.session_state.transcription = transcription
            st.session_state.transcript_ready = True
            st.success("Transcript retrieved successfully using API!")
            return
    except Exception as e:
        st.error(f"Error fetching transcript via API: {e}")

    try:
        from transcript.transcriber import transcribe_from_youtube

        transcription = transcribe_from_youtube(video_url)
        if transcription:
            st.session_state.transcription = transcription
            st.session_state.transcript_ready = True
            st.success("Transcript generated from audio!")
    except Exception as e:
        st.error(f"Audio Transcription Error: {e}")


def input_page():
    st.title("Welcome To SummariFi")
    st.write("Enter the YouTube video link below:")

    with st.form(key="video_form"):
        video_url = st.text_input(
            "YouTube Video Link",
            value=st.session_state.video_url,
            placeholder="https://www.youtube.com/watch?v=..."
        )
        submitted = st.form_submit_button(
            "Fetch Transcript",
            use_container_width=True
        )

    if submitted:
        st.session_state.video_url = video_url

        if not re.match(YOUTUBE_REGEX, video_url):
            st.error("Please provide a valid URL")
            return

        reset_summary_state()
        st.session_state.transcript_ready = False
        st.session_state.transcription = ""

        with st.spinner("Processing the video... Please wait."):
            fetch_transcript(video_url)

    if st.session_state.transcript_ready:
        st.info("Transcript is ready. Use the buttons below to view or generate each result.")
        st.text_area("Transcript Preview", value=st.session_state.transcription, height=250, disabled=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Print Transcript", use_container_width=True):
                st.session_state.results_page_ready = True
                st.session_state.current_page = "summary_page"
                st.rerun()

        with col2:
            if st.button("Generate Extractive Summary", use_container_width=True):
                with st.spinner("Generating extractive summary... Please wait."):
                    try:
                        from summarizer.extractive_summary import extractive_summary
                        extractive = extractive_summary(st.session_state.transcription)
                        st.session_state.extractive_summary = extractive
                        st.session_state.extractive_ready = True
                        st.session_state.results_page_ready = True
                        st.success("Extractive summary generated successfully!")
                    except Exception as e:
                        st.error(f"Error during extractive summary generation: {e}")

        with col3:
            if st.button("Generate Abstractive Summary", use_container_width=True):
                with st.spinner("Loading model and generating abstractive summary... Please wait."):
                    try:
                        from summarizer.extractive_summary import extractive_summary
                        extractive = extractive_summary(st.session_state.transcription)

                        from summarizer.abstractive_summary import summarize_text
                        abstractive = summarize_text(extractive)

                        st.session_state.extractive_summary = extractive
                        st.session_state.extractive_ready = True
                        st.session_state.abstractive_summary = abstractive
                        st.session_state.abstractive_ready = True
                        st.session_state.summary = abstractive
                        st.session_state.summary_generated = True
                        st.session_state.model_ready = True
                        st.session_state.results_page_ready = True
                        st.success("Abstractive summary generated successfully!")
                    except Exception as e:
                        st.session_state.model_ready = False
                        st.error(f"Error during summary generation: {e}")

    if st.session_state.summary_generated:
        with st.form(key="view_summary_form"):
            view_clicked = st.form_submit_button("Open Results Page", use_container_width=True)

        if view_clicked:
            st.session_state.current_page = "summary_page"
            st.rerun()


def summary_page():
    st.title("Results")

    transcript_tab, extractive_tab, abstractive_tab = st.tabs([
        "Transcript",
        "Extractive Summary",
        "Abstractive Summary",
    ])

    with transcript_tab:
        st.text_area("Transcript", value=st.session_state.transcription, height=280, disabled=True)

    with extractive_tab:
        st.text_area(
            "Extractive Summary",
            value=st.session_state.extractive_summary,
            height=280,
            disabled=True
        )

    with abstractive_tab:
        st.text_area(
            "Abstractive Summary",
            value=st.session_state.summary,
            height=280,
            disabled=True
        )

    import pyperclip

    with st.form(key="copy_summary_form"):
        copied = st.form_submit_button("Copy Abstractive Summary", use_container_width=True)

    if copied:
        pyperclip.copy(st.session_state.summary)
        st.success("Abstractive summary copied to clipboard!")

    with st.form(key="back_to_input_form"):
        go_back = st.form_submit_button("Go Back", use_container_width=True)

    if go_back:
        reset_summary_state()
        st.session_state.transcript_ready = False
        st.session_state.transcription = ""
        st.session_state.video_url = ""
        st.session_state.current_page = "input_page"
        st.rerun()


placeholder = st.empty()

if st.session_state.current_page == "input_page":
    with placeholder.container():
        input_page()
elif st.session_state.current_page == "summary_page":
    with placeholder.container():
        summary_page()
