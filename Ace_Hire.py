import streamlit as st
import os
import tempfile
import base64
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO, StringIO
from dotenv import load_dotenv
import sqlite3
import hashlib
from PIL import Image
import time
import google.generativeai as gai
import cv2
import PyPDF2
from docx import Document
from googletrans import Translator
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import uuid
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av
import pandas as pd
import io 
import openai
import unicodedata
import zipfile
import json
from utils.pdf_processor import extract_text_from_pdf, clean_text, extract_resume_sections
from utils.nlp_processor import analyze_job_description, analyze_resume, calculate_match_score
from utils.ui_components import display_match_score_gauge, display_keyword_match_bar, display_match_details_expander, display_recommendations
from utils.openai_helpers import initialize_openai, generate_interview_questions
from utils.audio_processor import get_indian_languages
from utils.analysis_processor import analyze_response, generate_comprehensive_report
from datetime import datetime, timedelta
from streamlit.components.v1 import html
# Add these imports if not already present
from typing import Optional
import pytesseract
from pdf2image import convert_from_bytes

# Configure page settings
st.set_page_config(page_title="ACE HIRE - AI Interview Prep", page_icon="📃", layout="wide")

# Database Initialization
import os
import sqlite3

def initialize_database():
    # Define the base directory and ensure it exists
    base_dir = r"C:\Users\A\Desktop\[FINAL_COMBINED]AI_Interviewer_Clean"
    db_path = os.path.join(base_dir, "ace_hire.db")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Connect to the database (this will create the file if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create all tables with proper schema in one transaction
        cursor.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL,
            security_question TEXT,
            security_answer TEXT,
            is_verified INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS interview_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            job_title TEXT NOT NULL,
            job_description TEXT,
            resume_text TEXT,
            questions TEXT,  
            responses TEXT,  
            report_content BLOB,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            feedback_text TEXT NOT NULL,
            submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS user_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_content BLOB NOT NULL,
            text_content TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );

        CREATE TABLE IF NOT EXISTS interview_recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question_number INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            recording BLOB NOT NULL,
            recording_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        ''')

        # Add any missing columns that might not exist in older versions
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN security_question TEXT")
            cursor.execute("ALTER TABLE users ADD COLUMN security_answer TEXT")
        except sqlite3.OperationalError:
            pass  # Columns already exist

        conn.commit()
        return True
        
    except PermissionError:
        print(f"Error: Permission denied when trying to create/open database at {db_path}")
        return False
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

# Video Recorder Class
class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.recording = False
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            self.frames.append(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def save_recording(self):
        if not self.frames:
            return None
            
        # Create video from frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        buffer = BytesIO()
        out = cv2.VideoWriter(buffer, fourcc, 20.0, (640, 480))
        
        for frame in self.frames:
            out.write(frame)
        out.release()
        
        video_bytes = buffer.getvalue()
        self.frames = []
        return video_bytes


# Helper Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_interview_session(user_id, job_title, job_description, resume_text, questions, responses, report_content):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO interview_sessions 
        (user_id, job_title, job_description, resume_text, questions, responses, report_content)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            job_title,
            job_description,
            resume_text,
            json.dumps(questions),
            json.dumps(responses),
            report_content
        ))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return False
    finally:
        conn.close()

def verify_login(email, password):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE email = ?', (email,))
    result = cursor.fetchone()
    conn.close()
    return result and result[0] == hash_password(password)

def register_user(first_name, last_name, email, phone, password):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
        INSERT INTO users (first_name, last_name, email, phone, password)
        VALUES (?, ?, ?, ?, ?)
        ''', (first_name, last_name, email, phone, hash_password(password)))
        conn.commit()
        st.success("Registration successful! Please verify your email.")
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_id(email):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def extract_text_from_pdf(file_content) -> Optional[str]:
    """
    Extract text content from PDF bytes with OCR fallback
    Args:
        file_content: PDF file bytes
    Returns:
        str: Extracted text or None if failed
    """
    # First try PyPDF2 extraction
    try:
        text = ""
        with BytesIO(file_content) as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        # If we got reasonable text, return it
        if text.strip() and len(text.strip()) > 50:
            return text.strip()
            
        # Otherwise try OCR fallback
        st.warning("Low text extraction - attempting OCR fallback...")
        return _extract_with_ocr(file_content)
        
    except Exception as e:
        st.warning(f"PDF extraction error: {str(e)} - attempting OCR fallback...")
        return _extract_with_ocr(file_content)

def _extract_with_ocr(pdf_bytes: bytes) -> str:
    """Extract text from image-based PDF using OCR"""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        
        text = ""
        with st.spinner("📷 Converting PDF pages to images..."):
            images = convert_from_bytes(pdf_bytes)
            
        progress_bar = st.progress(0)
        for i, image in enumerate(images):
            st.info(f"🔠 Extracting text from page {i+1}/{len(images)}...")
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
            progress_bar.progress((i + 1) / len(images))
        
        progress_bar.empty()
        return text.strip()
        
    except Exception as e:
        st.error(f"❌ OCR processing failed: {str(e)}")
        st.warning("Using fallback text extraction instead")
        return "Unable to extract full text. Please consider uploading a text-based PDF."
        

def extract_text_from_docx(file_content):
    """Expects bytes, returns text"""
    text = ""
    with BytesIO(file_content) as f:
        doc = Document(f)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text.strip()

def store_document(user_id, document_type, file_name, file_content, text_content=None):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    try:
        # If text_content is None, extract it based on the file type
        if text_content is None:
            if file_name.endswith('.pdf'):
                text_content = extract_text_from_pdf(file_content)
            elif file_name.endswith('.docx'):
                text_content = extract_text_from_docx(file_content)
            else:
                text_content = file_content.decode('utf-8')  # Fallback for text files

        cursor.execute('''
        INSERT INTO user_documents (user_id, document_type, file_name, file_content, text_content)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, document_type, file_name, file_content, text_content))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error storing document: {e}")
        return False
    finally:
        conn.close()

def generate_questions_from_text(text):
    model = gai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Generate relevant interview questions based on the following text:\n{text}"
    response = model.generate_content(prompt)
    return response.text.splitlines()  # Assuming the response is a list of questions

def translate_to_english(text, src_language):
    translator = Translator()
    translated = translator.translate(text, src=src_language, dest='en')
    return translated.text

def analyze_audio_response(audio_file):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Convert audio file to text
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            response_text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service"
    
    # Analyze sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(response_text)
    
    # Calculate fluency and hesitation patterns
    words = response_text.split()
    fluency_score = len(words) / (len(words) + response_text.count("uh") + response_text.count("um"))  # Simple fluency metric
    
    # Return analysis results
    return {
        "response_text": response_text,
        "sentiment": sentiment_scores,
        "fluency_score": fluency_score
    }



# Feedback Mechanism
def store_feedback(user_id, feedback_text):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO feedback (user_id, feedback_text)
            VALUES (?, ?)
        ''', (user_id, feedback_text))
        conn.commit()
        st.success("Thank you for your feedback!")
    except Exception as e:
        st.error(f"Error storing feedback: {e}")
    finally:
        conn.close()

# Audio recorder alternative using streamlit-webrtc
def audio_recorder_alternative(question_id):
    # Create a unique key for this recorder instance
    recorder_key = f"webrtc_recorder_{question_id}"
    
    # Create a class to handle audio processing
    class AudioProcessor:
        def __init__(self):
            self.audio_chunks = []
            self.recording = True
            self.start_time = datetime.now()
            
            # Create a temporary file path
            self.temp_dir = tempfile.gettempdir()
            self.audio_file_path = os.path.join(self.temp_dir, f"answer_{question_id}.wav")
            
            # Create file handles
            self.audio_file = open(self.audio_file_path, "wb")
            
        def recv(self, frame):
            # Check if we've been recording for more than 1 minute
            if (datetime.now() - self.start_time).total_seconds() >= 60:
                self.recording = False
                # Close the file if we're done recording
                if hasattr(self, 'audio_file') and self.audio_file and not self.audio_file.closed:
                    self.audio_file.close()
                
            if self.recording:
                try:
                    # Store the audio frame
                    self.audio_chunks.append(frame.to_ndarray())
                    # Write the audio chunk to file
                    self.audio_file.write(frame.to_ndarray().tobytes())
                except Exception as e:
                    print(f"Error processing audio frame: {str(e)}")
            
            # Return the frame unmodified
            return frame
                
        def get_file_path(self):
            # Close the file if needed
            if hasattr(self, 'audio_file') and self.audio_file and not self.audio_file.closed:
                try:
                    self.audio_file.close()
                except Exception as e:
                    print(f"Error closing audio file: {str(e)}")
                
            # Return the path if we have recorded data
            if hasattr(self, 'audio_chunks') and len(self.audio_chunks) > 0:
                return self.audio_file_path
            return None
    
    # Create a container for our state
    state_container = st.empty()
    
    # Track if recording is complete
    if f"recording_complete_{recorder_key}" not in st.session_state:
        st.session_state[f"recording_complete_{recorder_key}"] = False
        
    # Track audio file path
    if f"audio_file_path_{recorder_key}" not in st.session_state:
        st.session_state[f"audio_file_path_{recorder_key}"] = None
        
    # Check if we already have recorded audio for this question
    if st.session_state[f"recording_complete_{recorder_key}"] and st.session_state[f"audio_file_path_{recorder_key}"]:
        audio_path = st.session_state[f"audio_file_path_{recorder_key}"]
        if os.path.exists(audio_path):
            # Display success message
            state_container.success("✅ Recording completed!")
            
            # Display the recorded audio if possible
            try:
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.warning(f"Unable to preview audio: {str(e)}")
                
            return audio_path
            
    # Create a new processor
    processor = AudioProcessor()
    
    # Setup WebRTC component with updated parameters
    ctx = webrtc_streamer(
        key=recorder_key,
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=lambda: processor,
        frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    )
    
    # Recording status and controls
    if ctx.state.playing:
        # Display recording in progress message
        state_container.info("🔴 Recording in progress... (max 1 minute)")
        
        # Initialize start time if needed
        if f"start_time_{recorder_key}" not in st.session_state:
            st.session_state[f"start_time_{recorder_key}"] = datetime.now()
            
        # Calculate elapsed time
        elapsed_time = (datetime.now() - st.session_state[f"start_time_{recorder_key}"]).total_seconds()
        
        # Display countdown
        remaining_time = max(0, 60 - int(elapsed_time))
        st.write(f"⏱️ Time remaining: {remaining_time} seconds")
        
        # Auto-stop after 1 minute
        if elapsed_time >= 60:
            state_container.warning("⚠️ Maximum recording time reached! Click 'Stop' to continue.")
            st.session_state[f"recording_complete_{recorder_key}"] = True
            st.session_state[f"audio_file_path_{recorder_key}"] = processor.get_file_path()
            
    # If recording was just stopped or recording is complete
    elif not ctx.state.playing and (f"start_time_{recorder_key}" in st.session_state):
        # Recording was just stopped or completed
        audio_path = processor.get_file_path()
        if audio_path and os.path.exists(audio_path):
            # Mark as complete and store the path
            st.session_state[f"recording_complete_{recorder_key}"] = True
            st.session_state[f"audio_file_path_{recorder_key}"] = audio_path
            
            # Reset start time
            if f"start_time_{recorder_key}" in st.session_state:
                del st.session_state[f"start_time_{recorder_key}"]
                
            # Display success message
            state_container.success("✅ Recording completed!")
            
            # Display the recorded audio if possible
            try:
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.warning(f"Unable to preview audio: {str(e)}")
                
            return audio_path
    
    # If we have a path in session state, return it
    if st.session_state[f"recording_complete_{recorder_key}"] and st.session_state[f"audio_file_path_{recorder_key}"]:
        return st.session_state[f"audio_file_path_{recorder_key}"]
    
    return None

def circular_progress_bar(value, label):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect="equal"))
    ax.pie([value, 100 - value], labels=[f"{value}%", " "], colors=["#4CAF50", "#E0E0E0"], startangle=90)
    ax.text(0, 0, label, ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title(label, fontsize=16)
    st.pyplot(fig)

# Translations dictionary
translations = {
    "en": {
        "title": "ACE HIRE",
        "subtitle": "Become an ACE HIRE today, your gateway to interview success!",
        "help_desk": "Help Desk",
        "login": "Login",
        "register": "Register",
        "profile": "Profile",
        "how_it_works": "How It Works",
        "see_demo": "See Demo",
        "help": "Help",
        "testimonial": "ACE HIRE helped me prepare for my dream job! The practice sessions were incredibly realistic and the feedback was spot on.",
        "testimonial_author": "- Arnold S., Data Enthusiast"
    },
    "hi": {
        "title": "एस हायर"   ,
        "subtitle": "आपकी साक्षात्कार सफलता की ओर",
        "help_desk": "सहायता डेस्क",
        "login": "लॉगिन",
        "register": "पंजीकरण",
        "profile": "प्रोफ़ाइल",
        "how_it_works": "यह कैसे काम करता है",
        "see_demo": "डेमो देखें",
        "help": "सहायता",
        "testimonial": "एसी हायर ने मुझे मेरे सपनों की नौकरी के लिए तैयार करने में मदद की! प्रैक्टिस सत्र बेहद यथार्थवादी थे और फीडबैक बिल्कुल सही था।",
        "testimonial_author": "- अर्नोल्ड एस., डेटा उत्साही"
    },
    # Add other languages similarly...
}

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Initialize OpenAI client
openai_client = initialize_openai()

# ---- SESSION STATE INITIALIZATION ----
def initialize_session_state():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"
    if "resume_text" not in st.session_state:
        st.session_state["resume_text"] = None
    if "resume_sections" not in st.session_state:
        st.session_state["resume_sections"] = {}
    if "job_description" not in st.session_state:
        st.session_state["job_description"] = ""
    if "match_result" not in st.session_state:
        st.session_state["match_result"] = None
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    if "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = None
    if "resume_uploaded" not in st.session_state:
        st.session_state["resume_uploaded"] = False
    if "interview_responses" not in st.session_state:
        st.session_state["interview_responses"] = {}
    if "selected_language" not in st.session_state:
        st.session_state["selected_language"] = "English"
    if "comprehensive_report" not in st.session_state:
        st.session_state["comprehensive_report"] = None
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = 0
    if "recording" not in st.session_state:
        st.session_state["recording"] = False
    if "video_recorder" not in st.session_state:
        st.session_state["video_recorder"] = VideoRecorder()
    if "webrtc_ctx" not in st.session_state:
        st.session_state["webrtc_ctx"] = None
    if "question_visibility" not in st.session_state:
        st.session_state["question_visibility"] = [True, False, False]
    if "job_title" not in st.session_state:
        st.session_state["job_title"] = ""

# ---- PAGE RENDERING FUNCTIONS ----
def render_home_page():
    # Language selection moved to top-right
    language = st.selectbox("🌐 Language", 
                          options=list(translations.keys()), 
                          format_func=lambda x: translations[x]["title"],
                          key="lang_selector")
    
    lang = translations[language]
    
    with st.container():

        # ===== Header Section =====
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f'<h1 style="color:#2563EB; margin-bottom:0;">{lang["title"]}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1.2rem; color:#6B7280; margin-top:0;">{lang["subtitle"]}</p>', unsafe_allow_html=True)
        
        with col2:
            # Auth buttons container
            auth_container = st.container()
            btn_col1, btn_col2, btn_col3 = auth_container.columns([1, 1, 1], gap="small")
            
            # Profile button
            with btn_col1:
                if st.session_state.get("logged_in"):
                    if st.button("👤", 
                               help="View Profile", 
                               key="profile_button",
                               use_container_width=True):
                        st.session_state["current_page"] = "Profile"
                        st.rerun()
                else:
                    if st.button("👤", 
                               help="View Profile (requires login)", 
                               key="profile_button",
                               use_container_width=True):
                        st.session_state["redirect_to_profile"] = True
                        st.session_state["current_page"] = "Login"
                        st.rerun()
            
            # Login button
            with btn_col2:
                if st.button("🔑", 
                           help="Login",
                           key="home_login_btn",
                           use_container_width=True):
                    st.session_state["current_page"] = "Login"
                    st.rerun()
            
            # Register button
            with btn_col3:
                if st.button("📝", 
                           help="Register",
                           key="home_register_btn",
                           use_container_width=True):
                    st.session_state["current_page"] = "Register"
                    st.rerun()

        # ===== Hero Section =====
        hero_col1, hero_col2 = st.columns([1.5, 1], gap="large")
        
        with hero_col1:
            # Features list with icons
            st.markdown("""
            <div style="margin-top:1.5rem;">
                <div style="display:flex; align-items:center; margin-bottom:1rem;">
                    <span style="font-size:1.5rem; margin-right:0.5rem;">✨</span>
                    <span style="font-size:1.1rem;">AI-powered realistic interview simulations</span>
                </div>
                <div style="display:flex; align-items:center; margin-bottom:1rem;">
                    <span style="font-size:1.5rem; margin-right:0.5rem;">📊</span>
                    <span style="font-size:1.1rem;">Instant feedback on your responses</span>
                </div>
                <div style="display:flex; align-items:center; margin-bottom:2rem;">
                    <span style="font-size:1.5rem; margin-right:0.5rem;">📈</span>
                    <span style="font-size:1.1rem;">Track your progress over time</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Primary CTA button
            if st.session_state.get("logged_in"):
                if st.button("Start Practicing Now →", 
                            type="primary", 
                            use_container_width=True,
                            key="start_practicing"):
                    st.session_state["current_page"] = "Upload Resume"
                    st.rerun()
            else:
                if st.button("Start Practicing Now →", 
                            type="primary", 
                            use_container_width=True,
                            key="start_practicing"):
                    st.warning("Please log in to start practicing.")
                    st.session_state["current_page"] = "Login"
                    st.rerun()
            
               # 
    # Secondary buttons - Original exact layout
    cols = st.columns(2)
    with cols[0]:
        if st.button(lang["how_it_works"], 
                   use_container_width=True,
                   help="Learn about our process"):
            st.session_state["current_page"] = "About"
    with cols[1]:
        demo_url = "https://www.youtube.com/watch?v=TEh7-pOmXtA"
        if st.button(lang["see_demo"], 
                   use_container_width=True,
                   help="View a sample interview"):
            # Show redirect message in same position
            with cols[1]:
                st.markdown(f"""
                <div style="text-align:center; padding:0.5rem 1rem; border-radius:8px;
                           background-color:#f0f2f6; color:#333; margin-bottom:1rem;">
                    Opening demo...<br>
                    <a href="{demo_url}" target="_blank" style="color:#2563EB;">
                        Click here if nothing happens
                    </a>
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-open in new tab
                st.markdown(f"""
                <script>
                    window.open('{demo_url}', '_blank').focus();
                </script>
                """, unsafe_allow_html=True)
        # st.markdown(f"""<meta http-equiv="refresh" content="0; url={demo_url}">""", unsafe_allow_html=True)

                

        
        with hero_col2:
            # Hero image
            try:
                image = Image.open("C:\\Users\\A\\AI\\AI_Project-Ace_Higher\\cv.jpg")
                st.image(image, 
                        use_container_width=True,
                        caption="AI-powered interview preparation")
            except:
                st.error("Hero image not found")
            
            # Testimonial card
            st.markdown(f"""
            <div style="background:#F9FAFB; padding:1.5rem; border-radius:12px; margin-top:1rem;">
                <p style="font-style:italic; color:#4B5563;">"{lang["testimonial"]}"</p>
                <p style="font-weight:600; color:#111827;">{lang["testimonial_author"]}</p>
            </div>
            """, unsafe_allow_html=True)

    # ===== Features Section =====
    st.markdown("""
    <div style='margin-top:3rem;'>
        <h2 style='color:#2563EB; text-align:center;'>Why Choose ACE HIRE?</h2>
        <p style='text-align:center; color:#6B7280; margin-bottom:2rem;'>
            Our platform helps you ace your interviews through cutting-edge AI technology
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # First row of features
    features = st.columns(3, gap="medium")
    
    with features[0]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center; padding:1.5rem; height:100%;">
                <span style="font-size:2.5rem; display:block; margin-bottom:1rem;">🎯</span>
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">Personalized Practice</h3>
                <p style="color:#4B5563; margin:0;">
                    Get interview questions tailored specifically to your resume and target job role
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with features[1]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center; padding:1.5rem; height:100%;">
                <span style="font-size:2.5rem; display:block; margin-bottom:1rem;">🤖</span>
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">AI-Powered Analysis</h3>
                <p style="color:#4B5563; margin:0;">
                    Receive detailed feedback on your technical answers, communication skills, and body language
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with features[2]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center; padding:1.5rem; height:100%;">
                <span style="font-size:2.5rem; display:block; margin-bottom:1rem;">📊</span>
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">Performance Analytics</h3>
                <p style="color:#4B5563; margin:0;">
                    Track your progress with detailed metrics and improvement recommendations
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Second row of features
    features2 = st.columns(3, gap="medium")
    
    with features2[0]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center; padding:1.5rem; height:100%;">
                <span style="font-size:2.5rem; display:block; margin-bottom:1rem;">💬</span>
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">Real-time Feedback</h3>
                <p style="color:#4B5563; margin:0;">
                    Get instant suggestions during mock interviews to improve your responses
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with features2[1]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center; padding:1.5rem; height:100%;">
                <span style="font-size:2.5rem; display:block; margin-bottom:1rem;">📱</span>
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">Multi-language Support</h3>
                <p style="color:#4B5563; margin:0;">
                    Practice in multiple languages and get feedback in your preferred language
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with features2[2]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center; padding:1.5rem; height:100%;">
                <span style="font-size:2.5rem; display:block; margin-bottom:1rem;">🏆</span>
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">Industry Benchmarks</h3>
                <p style="color:#4B5563; margin:0;">
                    Compare your performance against industry standards for your target role
                </p>
            </div>
            """, unsafe_allow_html=True)

def render_profile_page():
    # Header with working back button
    col1, col2 = st.columns([4, 1])  # Adjust ratio as needed
    with col1:
        st.markdown('<h1 style="color:#2563EB; margin-bottom:0;">Your Profile</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Dashboard", 
                    key="profile_back_button",
                    help="Return to main dashboard"):
            st.session_state["current_page"] = "Home"  # Change "Home" to your dashboard page name if different
            st.rerun()

    st.markdown("---")  # Horizontal line for visual separation

    st.markdown("""
<style>
    div[data-testid="column"]:nth-of-type(2) button {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        background: white;
        color: #2563EB;
        margin-top: 0.5rem;
    }
    div[data-testid="column"]:nth-of-type(2) button:hover {
        background-color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)
    
    # Get user details from database
    user_id = get_user_id(st.session_state["username"])
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    
    # Fetch user profile information
    cursor.execute('SELECT first_name, last_name, email, phone FROM users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()
    
    if user_data:
        first_name, last_name, email, phone = user_data
        
        # Personal Information Section
        with st.container(border=True):
            st.subheader("Personal Information")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("First Name", value=first_name, disabled=True)
                st.text_input("Email", value=email, disabled=True)
            with col2:
                st.text_input("Last Name", value=last_name, disabled=True)
                st.text_input("Phone", value=phone, disabled=True)
        
        # Interview History Section
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Interview History")
            
            # Get all interview sessions for this user
            cursor.execute('''
                SELECT id, document_type, file_name, upload_date 
                FROM user_documents 
                WHERE user_id = ? 
                ORDER BY upload_date DESC
            ''', (user_id,))
            sessions = cursor.fetchall()
            
            if sessions:
                for session in sessions:
                    session_id, doc_type, file_name, upload_date = session
                    
                    with st.expander(f"{upload_date} - {file_name}"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Type:** {doc_type.replace('_', ' ').title()}")
                            st.write(f"**Date:** {upload_date}")
                        
                        with col2:
                            # Download button for PDF reports
                            if file_name.endswith('.pdf'):
                                cursor.execute('SELECT file_content FROM user_documents WHERE id = ?', (session_id,))
                                file_content = cursor.fetchone()[0]
                                
                                st.download_button(
                                    label="Download Report",
                                    data=file_content,
                                    file_name=file_name,
                                    mime="application/pdf",
                                    use_container_width=True
                                )
            else:
                st.info("No interview history found")
    
    conn.close()
    
    # Delete Account Button (with confirmation)
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    if st.button("Delete Account", type="secondary", help="Permanently delete your account and all data"):
        if st.checkbox("I understand this will permanently delete all my data"):
            if st.button("Confirm Deletion", type="primary"):
                conn = sqlite3.connect('ace_hire.db')
                cursor = conn.cursor()
                try:
                    # Delete all user data
                    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
                    cursor.execute('DELETE FROM user_documents WHERE user_id = ?', (user_id,))
                    cursor.execute('DELETE FROM interview_recordings WHERE user_id = ?', (user_id,))
                    cursor.execute('DELETE FROM feedback WHERE user_id = ?', (user_id,))
                    conn.commit()
                    
                    # Clear session and redirect to home
                    st.session_state.clear()
                    st.session_state["current_page"] = "Home"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting account: {str(e)}")
                finally:
                    conn.close()
        
def render_login_page():
    # Custom CSS for the login page
    st.markdown("""
    <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .login-title {
            color: #2563EB;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .login-form {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }
        .login-btn {
            width: 100%;
            margin-top: 1rem;
            padding: 0.75rem;
            border-radius: 8px;
            font-weight: 600;
        }
        .forgot-password {
            text-align: center;
            margin-top: 1rem;
            color: #2563EB;
            cursor: pointer;
        }
        .divider {
            display: flex;
            align-items: center;
            margin: 1.5rem 0;
            color: #6B7280;
        }
        .divider::before, .divider::after {
            content: "";
            flex: 1;
            border-bottom: 1px solid #E5E7EB;
        }
        .divider::before {
            margin-right: 1rem;
        }
        .divider::after {
            margin-left: 1rem;
        }
        .alternative-actions {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1.5rem;
        }
        .error-message {
            color: #EF4444;
            text-align: center;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main login container
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Login title and description
        st.markdown('<h2 class="login-title">Welcome Back</h2>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #6B7280; margin-bottom: 2rem;">Sign in to continue to ACE HIRE</p>', unsafe_allow_html=True)
        
        # Initialize error message
        login_error = None
        
        # Login form
        with st.form("login_form"):
            email = st.text_input(
                "Email Address*",
                placeholder="Enter your email address",
                key="login_email"
            )
            
            password = st.text_input(
                "Password*",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            # Remember me checkbox
            col1, col2 = st.columns([1, 2])
            with col1:
                remember_me = st.checkbox("Remember me", value=True)
            with col2:
                st.markdown('<p class="forgot-password" onclick="alert(\'Password reset link will be sent to your email\')">Forgot password?</p>', unsafe_allow_html=True)
            
            # Login button
            login_btn = st.form_submit_button(
                "Login",
                type="primary",
                use_container_width=True
            )
            
            if login_btn:
                if not email or not password:
                    login_error = "Please fill in all required fields"
                elif not verify_login(email, password):
                    login_error = "Invalid email or password"
                else:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = email
                    st.session_state["current_page"] = "Upload Resume"
                    st.rerun()
        
        # Display error message if any
        if login_error:
            st.markdown(f'<p class="error-message">{login_error}</p>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="divider">or</div>', unsafe_allow_html=True)
        
        # Social login options
        st.markdown('<p style="text-align: center; margin-bottom: 1rem;">Sign in with</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Google", use_container_width=True):
                st.warning("Google login coming soon!")
        with col2:
            if st.button("LinkedIn", use_container_width=True):
                st.warning("LinkedIn login coming soon!")
        with col3:
            if st.button("GitHub", use_container_width=True):
                st.warning("GitHub login coming soon!")
        
        # Alternative actions
        st.markdown('<div class="alternative-actions">', unsafe_allow_html=True)
        st.markdown('<p style="color: #6B7280; margin-right: 0.5rem;">Don\'t have an account?</p>', unsafe_allow_html=True)
        if st.button("Register", key="register_from_login"):
            st.session_state["current_page"] = "Register"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close login container
        
# Update the register_user function to include security questions
def register_user(first_name, last_name, email, phone, password):
    SECURITY_QUESTIONS = {
        "What was your first pet's name?": "pet",
        "In what city were you born?": "city",
        "What is your mother's maiden name?": "maiden",
        "What was your first car's model?": "car",
        "What is your favorite book title?": "book"
    }

    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    
    # Let user select and answer a security question
    security_question = st.selectbox(
        "Select a security question for password recovery",
        list(SECURITY_QUESTIONS.keys())
    )
    security_answer = st.text_input("Your answer (case sensitive)", type="password")
    
    if not security_answer.strip():
        st.error("Please provide an answer to the security question")
        return False
        
    try:
        cursor.execute('''
        INSERT INTO users (first_name, last_name, email, phone, password, security_question, security_answer)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            first_name, 
            last_name, 
            email, 
            phone, 
            hash_password(password),
            security_question,
            security_answer.strip()  # Store the exact case-sensitive answer
        ))
        conn.commit()
        st.success("Registration successful! Please verify your email.")
        return True
    except sqlite3.IntegrityError:
        st.error("Email already exists.")
        return False
    finally:
        conn.close()

def render_register_page():
    # Custom CSS for the registration page
    st.markdown("""
    <style>
        .register-container {
            max-width: 600px;
            margin: 2rem auto;
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            background: white;
        }
        .register-title {
            color: #2563EB;
            text-align: center;
            margin-bottom: 1.25rem;
            font-size: 1.8rem;
            font-weight: 600;
        }
        .register-subtitle {
            text-align: center;
            color: #6B7280;
            margin-bottom: 2rem;
            font-size: 1rem;
        }
        .register-form {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }
        .name-fields {
            display: flex;
            gap: 1rem;
        }
        .stTextInput>div>div>input, 
        .stTextInput>div>div>input:focus {
            border-radius: 8px;
            padding: 0.75rem;
        }
        .register-btn {
            width: 100%;
            margin-top: 1rem;
            padding: 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
        }
        .terms-container {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 1rem 0;
        }
        .terms-text {
            font-size: 0.9rem;
            color: #6B7280;
        }
        .terms-link {
            color: #2563EB;
            text-decoration: none;
        }
        .divider {
            display: flex;
            align-items: center;
            margin: 1.5rem 0;
            color: #9CA3AF;
            font-size: 0.9rem;
        }
        .divider::before, .divider::after {
            content: "";
            flex: 1;
            border-bottom: 1px solid #E5E7EB;
        }
        .divider::before {
            margin-right: 1rem;
        }
        .divider::after {
            margin-left: 1rem;
        }
        .login-prompt {
            text-align: center;
            margin-top: 1.5rem;
            font-size: 0.95rem;
            color: #6B7280;
        }
        .error-message {
            color: #EF4444;
            text-align: center;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }
        .success-message {
            color: #10B981;
            text-align: center;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }
        .password-hints {
            font-size: 0.8rem;
            color: #6B7280;
            margin-top: -0.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main registration container
    with st.container():
        st.markdown('<div class="register-container">', unsafe_allow_html=True)
        
        # Registration header
        st.markdown('<h2 class="register-title">Create Your Account</h2>', unsafe_allow_html=True)
        st.markdown('<p class="register-subtitle">Join ACE HIRE to start preparing for your dream job</p>', unsafe_allow_html=True)
        
        # Initialize messages
        register_error = None
        register_success = None
        
        # Registration form
        with st.form("register_form"):
            # Name fields in a row
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input(
                    "First Name*",
                    placeholder="Enter your first name",
                    key="register_first_name"
                )
            with col2:
                last_name = st.text_input(
                    "Last Name*",
                    placeholder="Enter your last name",
                    key="register_last_name"
                )
            
            # Contact information
            email = st.text_input(
                "Email Address*",
                placeholder="Enter your email address",
                key="register_email"
            )
            
            phone = st.text_input(
                "Phone Number*",
                placeholder="Enter your phone number",
                key="register_phone"
            )
            
            # Password fields
            password = st.text_input(
                "Create Password*",
                type="password",
                placeholder="Create a strong password",
                key="register_password"
            )
            st.markdown('<p class="password-hints">Use 8+ characters with a mix of letters, numbers & symbols</p>', unsafe_allow_html=True)
            
            confirm_password = st.text_input(
                "Confirm Password*",
                type="password",
                placeholder="Re-enter your password",
                key="register_confirm_password"
            )
            
            # Terms and conditions with single checkbox
            terms_accepted = st.checkbox("I agree to the [Terms of Service](https://docs.google.com/document/d/1MKqy6coL1Az25eTOQtYZ26NBqm0pYn8q1U7uAnxPEr0/edit?usp=sharing) and [Privacy Policy](https://docs.google.com/document/d/1UEV8MeIFGCBem_DvGRT2ZHHajOZyf-Gcslv24Fw2Vtk/edit?usp=sharing)",
            value=False,
            key="terms_checkbox")
            
            # Register button
            register_btn = st.form_submit_button(
                "Create Account",
                type="primary",
                use_container_width=True
            )
            
            if register_btn:
                # Validate form
                if not all([first_name, last_name, email, phone, password, confirm_password]):
                    register_error = "Please fill in all required fields"
                elif not terms_accepted:
                    register_error = "You must accept the terms and conditions"
                elif password != confirm_password:
                    register_error = "Passwords do not match"
                elif len(password) < 8:
                    register_error = "Password must be at least 8 characters"
                else:
                    # Try to register the user
                    if register_user(first_name, last_name, email, phone, password):
                        register_success = "Registration successful! Please verify your email."
                        # Auto-redirect after 2 seconds
                        time.sleep(2)
                        st.session_state["current_page"] = "Login"
                        st.rerun()
                    else:
                        register_error = "Email already exists"
        
        # Display messages
        if register_error:
            st.markdown(f'<p class="error-message">{register_error}</p>', unsafe_allow_html=True)
        elif register_success:
            st.markdown(f'<p class="success-message">{register_success}</p>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="divider">Already have an account?</div>', unsafe_allow_html=True)
        
        # Login prompt
        st.markdown('<div style="text-align: center; margin-top: 1.5rem;">', unsafe_allow_html=True)
        if st.button("Sign In", 
                    key="login_from_register",
                    use_container_width=True,
                    type="secondary"):
            st.session_state["current_page"] = "Login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close register container

def render_password_reset_page():
    st.header("Reset Password")
    email = st.text_input("Enter your email:")
    
    if st.button("Send Reset Link"):
        reset_password(email)

def reset_password(email):
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
    user = cursor.fetchone()
    if user:
        # Simulate sending a password reset link
        st.success("Password reset link has been sent to your email.")
    else:
        st.error("Email not found.")
    conn.close()

def render_upload_resume_page():
    # Prevent going back if analysis has started
    if st.session_state.get("analysis_started", False):
        st.error("You cannot go back once the analysis process has started.")
        if st.button("Continue to Resume-Job Match", use_container_width=True):
            st.session_state["current_page"] = "Resume-Job Match"
            st.rerun()
        return

    # Page header with progress indicator
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:1rem;">
        <h1 style="margin:0;">📄 Upload Documents</h1>
        <span style="margin-left:auto; background:#EFF6FF; color:#2563EB; 
                    padding:0.3rem 0.8rem; border-radius:20px; font-weight:500;">
            Step 1 of 4
        </span>
    </div>
    <p style="color:#6B7280; margin-top:-0.5rem; margin-bottom:1.5rem;">
        Upload your documents to begin your interview preparation
    </p>
    """, unsafe_allow_html=True)

    # Back button at top with better styling
    if st.button("← Back to Home", 
                key="back_to_home_top",
                use_container_width=True,
                type="secondary"):
        st.session_state["current_page"] = "Home"
        st.rerun()

    # Enhanced disclaimer with better styling
    with st.container(border=True):
        st.markdown("""
        <div style="background:#FFF3E0; padding:1rem; border-radius:8px;">
            <div style="display:flex; align-items:center; margin-bottom:0.5rem;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">⚠️</span>
                <h4 style="margin:0; color:#E65100;">Important Notice</h4>
            </div>
            <p style="color:#6B7280; margin:0;">
                Once you begin the analysis process by clicking "Analyze Documents", 
                you won't be able to go back and change your uploaded documents. 
                Please review carefully before proceeding.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Horizontal layout for upload sections with improved spacing
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Job Description Upload - Enhanced Card
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">📋</span>
                <h3 style="margin:0; color:#2563EB;">Job Description</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Tabs with better styling
            tab1, tab2 = st.tabs(["📁 Upload File", "✏️ Paste Text"])
            
            with tab1:
                job_description_file = st.file_uploader(
                    "Upload job description (PDF/DOCX/TXT):",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=False,
                    key="job_desc_upload",
                    help="Upload the job description file you're applying for"
                )
                if job_description_file:
                    try:
                        file_content = job_description_file.read()
                        if job_description_file.name.endswith('.pdf'):
                            text_content = extract_text_from_pdf(file_content)
                        elif job_description_file.name.endswith('.docx'):
                            text_content = extract_text_from_docx(file_content)
                        else:  # For .txt files
                            text_content = file_content.decode('utf-8')
                        
                        st.session_state["job_description"] = text_content
                        st.session_state["jd_file_content"] = file_content
                        st.success("✅ Job description processed!")
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
            
            with tab2:
                job_description = st.text_area(
                    "Or paste job description text:",
                    value=st.session_state.get("job_description", ""),
                    height=200,
                    placeholder="Paste the job description here...",
                    key="job_desc_text",
                    help="Alternatively, you can paste the job description text directly"
                )
                if job_description:
                    st.session_state["job_description"] = job_description

    with col2:
        # Resume Upload - Enhanced Card
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">📄</span>
                <h3 style="margin:0; color:#2563EB;">Your Resume</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # File uploader must be defined before processing
            resume_file = st.file_uploader(
                "Upload your resume (PDF/DOCX):",
                type=["pdf", "docx"],
                accept_multiple_files=False,
                key="resume_uploader",
                help="Upload your current resume in PDF or DOCX format"
            )
            
            # Now process the file if it exists
            if resume_file:
                if (st.session_state.get("uploaded_file_name") != resume_file.name) or (st.session_state.get("resume_text") is None):
                    with st.spinner("🔍 Analyzing your resume..."):
                        try:
                            file_content = resume_file.read()
                            if resume_file.name.endswith('.pdf'):
                                try:
                                    # First attempt standard extraction
                                    resume_text = extract_text_from_pdf(file_content)
                                    
                                    # If text is too short, warn user about possible image PDF
                                    if resume_text and len(resume_text.strip()) < 100:
                                        st.warning("⚠️ The PDF may be image-based. For better results, try a text-based PDF or we'll attempt OCR.")
                                    
                                except Exception as pdf_error:
                                    st.warning(f"⚠️ Standard extraction failed: {str(pdf_error)}. Attempting OCR...")
                                    resume_text = _extract_with_ocr(file_content)
                                    
                            else:  # .docx
                                resume_text = extract_text_from_docx(file_content)
                            
                            if resume_text:
                                cleaned_text = clean_text(resume_text)
                                resume_sections = extract_resume_sections(cleaned_text)
                                
                                st.session_state.update({
                                    "resume_text": cleaned_text,
                                    "resume_sections": resume_sections,
                                    "uploaded_file_name": resume_file.name,
                                    "resume_uploaded": True,
                                    "resume_file_content": file_content
                                })
                                st.success("✅ Resume processed!")
                            else:
                                st.error("❌ Couldn't extract text from file. Please try a different file format.")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing resume: {str(e)}")
                            if "OCR" in str(e):
                                st.info("💡 Tip: Try uploading a text-based PDF or DOCX file for better results")

                # Enhanced preview expander
                if st.session_state.get("resume_text"):
                    with st.expander("🔍 Preview extracted content", expanded=False):
                        preview_text = st.session_state["resume_text"]
                        st.text_area(
                            "Resume Content Preview",
                            value=preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text,
                            height=200,
                            label_visibility="collapsed",
                            key="resume_preview"
                        )

    # Target Position - Centered card with better styling
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">🎯</span>
                <h3 style="margin:0; color:#2563EB;">Target Position</h3>
            </div>
            """, unsafe_allow_html=True)
            
            job_title = st.text_input(
                "Job Title*",
                placeholder="e.g., Senior Data Scientist",
                value=st.session_state.get("job_title", ""),
                key="job_title_input",
                label_visibility="collapsed",
                help="Enter the exact job title you're applying for"
            )
            st.session_state["job_title"] = job_title

    # Validation and action section with improved layout
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    
    # Check requirements
    requirements_met = all([
        bool(st.session_state.get("job_title", "").strip()),
        bool(st.session_state.get("job_description", "").strip()),
        st.session_state.get("resume_uploaded", False)
    ])
    
    # Enhanced status indicator
    if not requirements_met:
        missing = []
        if not st.session_state.get("job_title", "").strip():
            missing.append("Job Title")
        if not st.session_state.get("job_description", "").strip():
            missing.append("Job Description")
        if not st.session_state.get("resume_uploaded", False):
            missing.append("Resume")
        
        with st.container(border=True):
            st.markdown(f"""
            <div style="display:flex; align-items:center; color:#D97706;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">⚠️</span>
                <div>
                    <h4 style="margin:0;">Missing Required Information</h4>
                    <p style="margin:0;">Please provide: {', '.join(missing)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Centered action buttons with improved styling
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if st.button("← Back to Home", 
                        use_container_width=True,
                        key="back_to_home_bottom",
                        type="secondary"):
                st.session_state["current_page"] = "Home"
                st.rerun()
        with col2:
            if st.button(
                "Analyze Documents →",
                type="primary",
                use_container_width=True,
                disabled=not requirements_met,
                help="Analyze your resume against the job description"
            ):
                with st.spinner("🧠 Performing analysis..."):
                    try:
                        # Perform analysis
                        job_analysis = analyze_job_description(st.session_state["job_description"])
                        resume_analysis = analyze_resume(
                            st.session_state["resume_text"],
                            st.session_state.get("resume_sections", {})
                        )
                        match_result = calculate_match_score(resume_analysis, job_analysis)
                        
                        # Store results
                        st.session_state.update({
                            "job_analysis": job_analysis,
                            "resume_analysis": resume_analysis,
                            "match_result": match_result,
                            "analysis_started": True  # Flag that analysis has started
                        })
                        
                        # Store documents in database
                        user_id = get_user_id(st.session_state["username"])
                        if resume_file:
                            store_document(
                                user_id=user_id,
                                document_type='resume',
                                file_name=resume_file.name,
                                file_content=st.session_state["resume_file_content"]
                            )
                        
                        # Handle job description storage
                        if job_description_file:
                            store_document(
                                user_id=user_id,
                                document_type='job_description',
                                file_name=job_description_file.name,
                                file_content=st.session_state["jd_file_content"]
                            )
                        else:  # Pasted text
                            store_document(
                                user_id=user_id,
                                document_type='job_description',
                                file_name='job_description.txt',
                                file_content=st.session_state["job_description"].encode('utf-8'),
                                text_content=st.session_state["job_description"]
                            )
                        
                        # Redirect to results page
                        st.session_state["current_page"] = "Resume-Job Match"
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")

# Add this helper function (place it near your other utility functions)
def _extract_with_ocr(pdf_bytes: bytes) -> str:
    """Extract text from image-based PDF using OCR"""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        
        text = ""
        with st.spinner("📷 Converting PDF pages to images..."):
            images = convert_from_bytes(pdf_bytes)
            
        progress_bar = st.progress(0)
        for i, image in enumerate(images):
            st.info(f"🔠 Extracting text from page {i+1}/{len(images)}...")
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
            progress_bar.progress((i + 1) / len(images))
        
        progress_bar.empty()
        return text.strip()
        
    except Exception as e:
        st.error(f"❌ OCR processing failed: {str(e)}")
        st.warning("Using fallback text extraction instead")
        return "Unable to extract full text. Please consider uploading a text-based PDF."
    
def render_resume_job_match_page():
    
    st.markdown(f"""
    <div style="background:#FFC601; padding:1rem; border-radius:8px; margin-bottom:1.5rem;">
        <h3 style="margin:0;">Applying for: <span style="color:#2563EB">{st.session_state.get("job_title", "Not specified")}</span></h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Page header with progress indicator
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:1rem;">
        <h1 style="margin:0;">📊 Resume Analysis</h1>
        <span style="margin-left:auto; background:#FFC601; color:#2563EB; padding:0.3rem 0.8rem; border-radius:20px; font-weight:500;">
            Step 2 of 4
        </span>
    </div>
    <p style="color:#6B7280; margin-top:-0.5rem; margin-bottom:1.5rem;">
        See how well your resume matches the job requirements
    </p>
    """, unsafe_allow_html=True)

    # Check if we have analysis results
    if not st.session_state.get("match_result"):
        st.warning("Please analyze your documents first!")
        st.button("Go Back to Upload", on_click=lambda: st.session_state.update({"current_page": "Upload Resume"}))
        return

    match_result = st.session_state["match_result"]
    
    # Overall match score card
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            # Enhanced circular progress chart
            if 'match_result' in st.session_state:
                score = st.session_state["match_result"].get("overall_score", 0) * 100
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#4C78A8"},
                        'steps': [
                            {'range': [0, 50], 'color': "#E45756"},  # Red
                            {'range': [50, 75], 'color': "#F58518"},  # Orange
                            {'range': [75, 100], 'color': "#54A24B"}  # Green
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': score
                        }
                    },
                    number={
                        'font': {'size': 28, 'color': 'black'},
                        'suffix': '%'
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(t=0, b=0, l=20, r=20),
                    font={'family': "Arial"}
                )
                
                # Add annotation for score interpretation
                fig.add_annotation(
                    x=0.5,
                    y=-0.2,
                    text=f"{'Excellent' if score >= 80 else 'Good' if score >= 50 else 'Needs Improvement'}",
                    showarrow=False,
                    font=dict(size=14, color="black")
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

        with col2:
            st.markdown("""
            <h3 style="margin-top:0;">Overall Match Score</h3>
            <p style="color:#6B7280;">
                This score represents how well your resume aligns with the job requirements.
                Higher scores indicate better alignment with the position.
            </p>
            """, unsafe_allow_html=True)
            
            # Score interpretation with colored indicators
            score = st.session_state["match_result"].get("overall_score", 0) * 100
            if score >= 80:
                st.success("""
                **Excellent match!**  
                Your resume strongly aligns with the job requirements.
                """)
            elif score >= 50:
                st.warning("""
                **Good match.**  
                Consider some improvements to better align with the position.
                """)
            else:
                st.error("""
                **Needs improvement.**  
                Significant gaps exist between your resume and job requirements.
                """)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    
    # Key metrics in columns
    st.subheader("Detailed Analysis", divider="blue")
    cols = st.columns(3)
    
    with cols[0]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center;">
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">{}/10</h3>
                <p style="color:#6B7280; margin-top:0;">Keyword Match</p>
            </div>
            """.format(len(match_result.get("matching_keywords", []))), unsafe_allow_html=True)
    
    with cols[1]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center;">
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">{}/10</h3>
                <p style="color:#6B7280; margin-top:0;">Experience Alignment</p>
            </div>
            """.format(int(match_result.get("experience_score", 0) * 10)), unsafe_allow_html=True)
    
    with cols[2]:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align:center;">
                <h3 style="color:#2563EB; margin-bottom:0.5rem;">{}/10</h3>
                <p style="color:#6B7280; margin-top:0;">Skills Coverage</p>
            </div>
            """.format(int(match_result.get("skills_score", 0) * 10)), unsafe_allow_html=True)

    # Keyword analysis expander
    with st.expander("🔍 **Keyword Analysis**", expanded=True):
        tab1, tab2 = st.tabs(["✅ Matching Keywords", "❌ Missing Keywords"])
        
        with tab1:
            if match_result.get("matching_keywords"):
                cols = st.columns(4)
                for i, keyword in enumerate(match_result["matching_keywords"]):
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div style="background:#D1FAE5; color:#065F46; padding:0.3rem 0.8rem; 
                                    border-radius:20px; margin:0.2rem; text-align:center;">
                            {keyword}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No matching keywords found")
        
        with tab2:
            if match_result.get("missing_keywords"):
                cols = st.columns(4)
                for i, keyword in enumerate(match_result["missing_keywords"]):
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div style="background:#FEE2E2; color:#B91C1C; padding:0.3rem 0.8rem; 
                                    border-radius:20px; margin:0.2rem; text-align:center;">
                            {keyword}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No important keywords missing")

    # Section-by-section analysis
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.subheader("Section Analysis", divider="blue")
    
    sections = st.columns(2)
    with sections[0]:
        with st.container(border=True):
            st.markdown("""
            <h4 style="color:#2563EB; margin-bottom:0.5rem;">Strengths</h4>
            <ul style="color:#6B7280; padding-left:1.2rem;">
            """, unsafe_allow_html=True)
            
            for strength in match_result.get("strengths", ["No significant strengths identified"]):
                st.markdown(f"<li>{strength}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul>", unsafe_allow_html=True)
    
    with sections[1]:
        with st.container(border=True):
            st.markdown("""
            <h4 style="color:#2563EB; margin-bottom:0.5rem;">Areas for Improvement</h4>
            <ul style="color:#6B7280; padding-left:1.2rem;">
            """, unsafe_allow_html=True)
            
            for improvement in match_result.get("improvements", ["No major improvements needed"]):
                st.markdown(f"<li>{improvement}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul>", unsafe_allow_html=True)

    # Navigation buttons
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state["current_page"] = "Upload Resume"
            st.rerun()
    
    with col3:
        if st.button("Generate Questions →", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Generate Questions"
            st.rerun()

def render_generate_questions_page():
    # Page header with progress indicator
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:1rem;">
        <h1 style="margin:0;">❓ Generate Interview Questions</h1>
        <span style="margin-left:auto; background:#FFC601; color:#2563EB; padding:0.3rem 0.8rem; border-radius:20px; font-weight:500;">
            Step 3 of 4
        </span>
    </div>
    <p style="color:#6B7280; margin-top:-0.5rem; margin-bottom:1.5rem;">
        Personalized questions based on your resume and job description
    </p>
    """, unsafe_allow_html=True)

    # Verify requirements are met
    if not st.session_state.get("resume_text") or not st.session_state.get("resume_uploaded", False):
        st.warning("Please upload your resume first!")
        st.button("Go to Resume Upload", on_click=lambda: st.session_state.update({"current_page": "Upload Resume"}))
        return
    elif not st.session_state.get("match_result"):
        st.warning("Please analyze your resume-job match first!")
        st.button("Go to Resume-Job Match", on_click=lambda: st.session_state.update({"current_page": "Resume-Job Match"}))
        return

    # Display match summary in a card
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            match_score = st.session_state["match_result"].get("overall_score", 0) * 100
            st.metric("Overall Match Score", f"{match_score:.1f}%")
        with col2:
            matching_count = len(st.session_state["match_result"].get("matching_keywords", []))
            missing_count = len(st.session_state["match_result"].get("missing_keywords", []))
            st.metric("Keyword Analysis", f"{matching_count} ✅ | {missing_count} ❌")

    st.divider()

    # Question generation section
    st.markdown("""
    <h3 style="margin-bottom:0.5rem;">Generate Your Interview Questions</h3>
    <p style="color:#6B7280; margin-top:0;">
        Click below to generate personalized questions based on your resume and the job requirements.
    </p>
    """, unsafe_allow_html=True)

    #Generate questions button with loading state
    if st.button("✨ Generate Questions", 
                type="primary", 
                use_container_width=True,
                key="generate_questions"):
        with st.spinner("Generating personalized interview questions..."):
            try:
                # Corrected function call with proper arguments
                questions = generate_interview_questions(
                    client=openai_client,  # The initialized OpenAI client
                    job_title=st.session_state.get("job_title", ""),
                    job_description=st.session_state.get("job_description", ""),
                    resume_text=st.session_state.get("resume_text", ""),
                    match_result=st.session_state.get("match_result", {}),
                    num_questions=3
                )
                
                if questions:
                    st.session_state["questions"] = questions
                    st.toast("Questions generated successfully!", icon="✅")
                else:
                    st.error("Failed to generate questions. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Display generated questions
    if st.session_state.get("questions"):
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        st.subheader("Your Personalized Questions", divider="blue")
        
        for i, question in enumerate(st.session_state["questions"], 1):
            with st.container(border=True):
                st.markdown(f"""
                <div style="display:flex; align-items:flex-start; margin-bottom:0.5rem;">
                    <span style="background:#2563EB; color:white; border-radius:50%; 
                                width:24px; height:24px; display:flex; align-items:center; 
                                justify-content:center; margin-right:0.8rem; flex-shrink:0;">
                        {i}
                    </span>
                    <p style="margin:0; font-size:1.05rem;">
                        {question.lstrip('- *').strip()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a small space between questions
                if i < len(st.session_state["questions"]):
                    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # Navigation buttons
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Analysis", use_container_width=True):
            st.session_state["current_page"] = "Resume-Job Match"
            st.rerun()
    
    with col3:
        if st.button("Start Interview →", 
                   type="primary", 
                   use_container_width=True,
                   disabled=not st.session_state.get("questions")):
            st.session_state["current_page"] = "Interview Session"
            st.rerun()
    
    # Help text if no questions generated yet
    if not st.session_state.get("questions"):
        st.markdown("""
        <div style="text-align:center; margin-top:4rem; color:#6B7280;">
            <p>Your personalized questions will appear here after generation</p>
        </div>
        """, unsafe_allow_html=True)

def render_interview_session_page():

    st.markdown("""
    <style>
        /* Remove white background from question cards */
        .question-card {
            background: #FFFFFF !important;
            border: none !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        /* Set proper text color for questions */
        .question-card h3, .question-card p {
            color: #1A1A1A !important;
        }
        /* Keep other styling for audio options */
        .audio-option-card {
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            margin-bottom: 1rem;
            background: #F8FAFC;
        }
        .audio-option-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .audio-option-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-left: 0.5rem;
            color: #1A1A1A;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page header with progress indicator
    st.markdown("""
    <style>
        .question-card {
            padding: 1.5rem;
            background: #F8FAFC;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            margin-bottom: 1.5rem;
        }
        .audio-option-card {
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            margin-bottom: 1rem;
        }
        .audio-option-header {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        .audio-option-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-left: 0.5rem;
        }
        .tab-content {
            padding: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="display:flex; align-items:center; margin-bottom:1rem;">
            <h1 style="margin:0;">🎤 Interview Practice Session</h1>
            <span style="margin-left:auto; background:#FFC601; color:#2563EB; padding:0.3rem 0.8rem; border-radius:20px; font-weight:500;">
                Question {current}/{total}
            </span>
        </div>
        """.format(
            current=st.session_state.get("current_question", 0) + 1,
            total=len(st.session_state.get("questions", []))
        ),
        unsafe_allow_html=True
    )

    # Verify requirements
    if not st.session_state.get("questions"):
        st.error("No interview questions found. Please generate questions first!")
        if st.button("Go to Generate Questions"):
            st.session_state["current_page"] = "Generate Questions"
            st.rerun()
        return

    questions = st.session_state["questions"]
    current_q_index = st.session_state.get("current_question", 0)
    current_question = questions[current_q_index].lstrip('- *').strip()
    question_id = f"q_{current_q_index}"

    # Current Question Card - Very Prominent
    st.markdown(f"""
    <div class="question-card">
        <h3 style="color:#2563EB; margin-bottom:0.8rem;">Question {current_q_index + 1}</h3>
        <p style="font-size:1.2rem; margin-bottom:0;">{current_question}</p>
    </div>s
    """, unsafe_allow_html=True)

    # Language selection
    languages = get_indian_languages()
    selected_language = st.selectbox(
        "Select your preferred language for answering", 
        languages,
        index=languages.index(st.session_state.get("selected_language", "English"))
    )
    st.session_state["selected_language"] = selected_language

    # Check if this question has already been answered
    existing_response = st.session_state["interview_responses"].get(question_id, {})

    # Main Response Section - Audio Options First
    st.subheader("Record Your Answer", divider="blue")
    
    # Create two prominent cards for audio options
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Live Recording Option
        with st.container():
            st.markdown("""
            <div class="audio-option-card">
                <div class="audio-option-header">
                    <span style="font-size:1.5rem;">🎙️</span>
                    <span class="audio-option-title">Record Live</span>
                </div>
                <div class="tab-content">
            """, unsafe_allow_html=True)
            
            st.write("Record your answer in real-time (max 1 minute):")
            audio_file_path = audio_recorder_alternative(question_id)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    with col2:
        # Upload Audio Option
        with st.container():
            st.markdown("""
            <div class="audio-option-card">
                <div class="audio-option-header">
                    <span style="font-size:1.5rem;">📁</span>
                    <span class="audio-option-title">Upload Audio</span>
                </div>
                <div class="tab-content">
            """, unsafe_allow_html=True)
            
            st.write("Upload a pre-recorded answer (max 1 minute):")
            uploaded_audio = st.file_uploader(
                "Choose audio file", 
                type=["mp3", "wav", "m4a", "ogg"], 
                key=f"audio_upload_{question_id}",
                label_visibility="collapsed"
            )
            
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Handle audio processing (for both recording and upload)
    audio_submitted = False
    transcription_text = ""
    
    if (audio_file_path and os.path.exists(audio_file_path)) or uploaded_audio:
        if uploaded_audio:
            # Save the uploaded file temporarily
            temp_dir = tempfile.gettempdir()
            audio_file_path = os.path.join(temp_dir, f"upload_{question_id}_{uploaded_audio.name}")
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
        
        # Display audio player
        st.audio(audio_file_path, format="audio/wav")
        
        # Attempt automatic transcription
        with st.spinner("Transcribing your answer..."):
            try:
                transcription = analyze_audio_response(audio_file_path)
                if isinstance(transcription, dict) and "response_text" in transcription:
                    transcription_text = transcription["response_text"]
                    st.success("🎉 Transcription successful!")
                    st.text_area("Transcribed Answer", 
                                value=transcription_text,
                                height=150,
                                key=f"transcription_{question_id}")
                else:
                    st.warning("⚠️ Couldn't transcribe automatically. Please type your answer below.")
                    manual_text = st.text_area("Please type your answer:",
                                             height=200,
                                             key=f"manual_input_{question_id}")
                    transcription_text = manual_text
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
                transcription_text = ""
        
        # Submit button for audio responses
        if st.button("Submit Audio Answer", 
                    type="primary",
                    disabled=not transcription_text.strip(),
                    use_container_width=True):
            audio_submitted = True

    # Process the submitted answer
    if transcription_text and (audio_submitted or st.session_state.get(f"submit_text_{question_id}")):
        with st.spinner("Analyzing your answer..."):
            try:
                # Calculate duration (estimate for audio, count words for text)
                duration = 0
                if audio_file_path and os.path.exists(audio_file_path):
                    file_size = os.path.getsize(audio_file_path) / 1024  # KB
                    duration = min(60, file_size * 0.064)  # Rough estimate
                else:
                    duration = len(transcription_text.split()) / 3  # Words per second estimate
                
                # Store the response data
                response_data = {
                    "question": current_question,
                    "transcription": transcription_text,
                    "translation": transcription_text,
                    "response_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": duration,
                    "response_type": "audio" if audio_submitted else "text"
                }
                
                # Analyze the response
                analysis = analyze_response(
                    openai_client, 
                    transcription_text,
                    current_question, 
                    st.session_state.get("job_description", ""),
                    duration
                )
                
                response_data["analysis"] = analysis
                st.session_state["interview_responses"][question_id] = response_data
                
                # Show analysis results immediately
                st.subheader("Analysis Results", divider="blue")
                
                cols = st.columns(4)
                metrics = [
                    ("Relevance", analysis.get('relevance_score', 0)),
                    ("Clarity", analysis.get('clarity_score', 0)),
                    ("Technical", analysis.get('technical_accuracy', 0)),
                    ("Professional", analysis.get('professionalism', 0))
                ]
                
                for col, (label, value) in zip(cols, metrics):
                    with col:
                        st.metric(label, f"{value}/10")
                
                st.write("**Feedback:**")
                st.write(analysis.get("feedback", "No feedback available."))
                
                # Move to next question if available
                if current_q_index + 1 < len(questions):
                    st.session_state["current_question"] += 1
                    time.sleep(2)  # Let user see the results
                    st.rerun()
                else:
                    st.session_state["current_page"] = "Final Report"
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing answer: {str(e)}")

    # Navigation buttons at bottom
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Question", 
                    disabled=current_q_index == 0,
                    use_container_width=True):
            st.session_state["current_question"] -= 1
            st.rerun()
    
    with col2:
        # Next button shows "Finish" if on last question
        next_text = "Finish" if current_q_index + 1 >= len(questions) else "Next Question →"
        next_disabled = question_id not in st.session_state["interview_responses"]
        
        if st.button(next_text, 
                    type="primary",
                    disabled=next_disabled,
                    use_container_width=True):
            if current_q_index + 1 < len(questions):
                st.session_state["current_question"] += 1
                st.rerun()
            else:
                st.session_state["current_page"] = "Final Report"
                st.rerun()
                    
def render_final_report_page():
    """Render the optimized final report with single PDF download"""
    # Page header with progress indicator
    st.markdown("""
    <div style="display:flex; align-items:center; margin-bottom:1rem;">
        <h1 style="margin:0;">📊 Interview Performance Report</h1>
        <span style="margin-left:auto; background:#EFF6FF; color:#2563EB; 
                    padding:0.3rem 0.8rem; border-radius:20px; font-weight:500;">
            Step 4 of 4
        </span>
    </div>
    <p style="color:#6B7280; margin-top:-0.5rem; margin-bottom:1.5rem;">
        Detailed analysis of your interview performance
    </p>
    """, unsafe_allow_html=True)

    # Verify requirements
    if not st.session_state.get("interview_responses"):
        st.error("Please complete the interview session first!")
        st.button("Go to Interview", on_click=lambda: st.session_state.update({"current_page": "Interview Session"}))
        return

    # Generate report if needed
    if not st.session_state.get("comprehensive_report"):
        with st.spinner("Generating your comprehensive analysis..."):
            try:
                report = generate_comprehensive_report(
                    openai_client,
                    st.session_state["interview_responses"],
                    st.session_state.get("job_title", ""),
                    st.session_state.get("job_description", "")
                )
                st.session_state["comprehensive_report"] = report or {}
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
                st.session_state["comprehensive_report"] = {}

    report = st.session_state.get("comprehensive_report", {})
    avg_scores = report.get("average_scores", {})

    # ========== Report Sections ========== #
    
    # 1. Overall Assessment Card
    with st.container(border=True):
        st.subheader("Overall Performance")
        st.write(report.get("overall_assessment", "No assessment available"))
        
        # Calculate and display overall score
        if avg_scores:
            overall_score = (avg_scores.get('technical', 0) * 10 + 
                           avg_scores.get('clarity', 0) * 10 + 
                           avg_scores.get('relevance', 0) * 10 + 
                           avg_scores.get('professionalism', 0) * 10) / 4
            st.metric("Overall Score", f"{overall_score:.1f}/10")

    # 2. Performance Metrics Section
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.subheader("Detailed Metrics", divider="blue")
    
    if avg_scores:
        # Visual metrics display
        cols = st.columns(4)
        metrics = [
            ("Technical", f"{avg_scores.get('technical', 0) * 10:.1f}/10"),
            ("Clarity", f"{avg_scores.get('clarity', 0) * 10:.1f}/10"),
            ("Relevance", f"{avg_scores.get('relevance', 0) * 10:.1f}/10"),
            ("Professionalism", f"{avg_scores.get('professionalism', 0) * 10:.1f}/10")
        ]
        
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label, value)

    # 3. Strengths & Improvements
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.subheader("Key Insights", divider="blue")
    
    with st.expander("✅ Strengths", expanded=True):
        for strength in report.get("technical_strengths", ["No strengths identified"]):
            st.write(f"- {strength}")
    
    with st.expander("🔍 Areas for Improvement"):
        for improvement in report.get("technical_improvements", ["No improvements needed"]):
            st.write(f"- {improvement}")

    # 4. Question Analysis
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.subheader("Question-by-Question Review", divider="blue")
    
    for q_id, data in st.session_state.get("interview_responses", {}).items():
        question_index = int(q_id.split("_")[1]) + 1
        with st.expander(f"Question {question_index}: {data.get('question', '')[:60]}..."):
            st.write(f"**Question:** {data.get('question', '')}")
            st.write(f"**Your Answer:** {data.get('translation', '')[:500]}{'...' if len(data.get('translation', '')) > 500 else ''}")
            
            analysis = data.get("analysis", {})
            cols = st.columns(3)
            metrics = [
                ("Relevance", analysis.get('relevance_score', 0)),
                ("Clarity", analysis.get('clarity_score', 0)),
                ("Technical", analysis.get('technical_accuracy', 0))
            ]
            
            for col, (label, value) in zip(cols, metrics):
                with col:
                    st.metric(label, f"{value}/10")
            
            st.write("**Feedback:**")
            st.write(analysis.get("feedback", "No feedback available."))

    # At the end of the function, before the navigation buttons:
    if st.session_state.get("logged_in"):
        user_id = get_user_id(st.session_state["username"])
        pdf_buffer = create_pdf_report()
        
        # Save the session to database - ADD UNIQUE KEY
        if st.button("Save This Session", key="save_session_button"):
            success = save_interview_session(
                user_id=user_id,
                job_title=st.session_state.get("job_title", ""),
                job_description=st.session_state.get("job_description", ""),
                resume_text=st.session_state.get("resume_text", ""),
                questions=st.session_state.get("questions", []),
                responses=st.session_state.get("interview_responses", {}),
                report_content=pdf_buffer.getvalue()
            )
            if success:
                st.success("Session saved successfully!")
                st.rerun()  # Refresh to show the updated state

    # ========== Navigation & Actions ========== #
    
    st.markdown("<div style='margin-top:3rem;'></div>", unsafe_allow_html=True)
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        if st.button("Start New Session", type="primary", use_container_width=True):
            # Reset session while preserving login info
            keys_to_preserve = ['logged_in', 'username']
            preserved_state = {k: st.session_state[k] for k in keys_to_preserve}
            st.session_state.clear()
            st.session_state.update(preserved_state)
            st.session_state["current_page"] = "Upload Resume"
            st.rerun()
    
    with action_col2:
        # Single PDF download button
        now = datetime.now()
        pdf_filename = f"Ace_Report_{now.strftime('%H%M%S%d%m%Y')}.pdf"
        
        with st.spinner("Preparing PDF report..."):
            pdf_buffer = create_pdf_report()
            st.download_button(
                label="📄 Download Full Report",
                data=pdf_buffer,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True
            )

def create_pdf_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import io
    import base64
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles with unique names
    custom_styles = {
        'ReportTitle': {'fontSize': 16, 'alignment': 1, 'spaceAfter': 12, 'fontName': 'Helvetica-Bold'},
        'SectionHeader': {'fontSize': 14, 'spaceBefore': 12, 'spaceAfter': 6, 'fontName': 'Helvetica-Bold'},
        'SubHeader': {'fontSize': 12, 'spaceBefore': 10, 'spaceAfter': 6, 'fontName': 'Helvetica-Bold'},
        'BodyText': {'fontSize': 10, 'spaceAfter': 6, 'fontName': 'Helvetica'},
        'BulletPoint': {'fontSize': 10, 'leftIndent': 10, 'spaceAfter': 3, 'fontName': 'Helvetica'}
    }
    
    for style_name, style_params in custom_styles.items():
        if not hasattr(styles, style_name):
            styles.add(ParagraphStyle(name=style_name, **style_params))
    
    elements = []
    
    # Get report data
    report = st.session_state.get("comprehensive_report", {})
    avg_scores = report.get("average_scores", {})
    responses = st.session_state.get("interview_responses", {})
    
    # Helper function to safely convert content to string
    def safe_to_string(content, default=""):
        if content is None:
            return default
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return "\n".join([f"{k}: {v}" for k, v in content.items()])
        if isinstance(content, list):
            return "\n".join([str(item) for item in content])
        return str(content)
    
    # ========== Report Sections ========== #
    
    # 1. Report Header
    elements.append(Paragraph("INTERVIEW PERFORMANCE REPORT", styles['ReportTitle']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Metadata
    meta_data = [
        f"Date: {datetime.now().strftime('%B %d, %Y')}",
        f"Candidate: {st.session_state.get('username', '')}",
        f"Job Title: {st.session_state.get('job_title', '')}"
    ]
    
    for meta in meta_data:
        elements.append(Paragraph(meta, styles['BodyText']))
    elements.append(Spacer(1, 0.3*inch))
    
    # 2. Overall Assessment Card
    elements.append(Paragraph("Overall Assessment", styles['SectionHeader']))
    overall_assessment = safe_to_string(report.get("overall_assessment", "No assessment available yet."))
    elements.append(Paragraph(overall_assessment, styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Calculate and display overall score
    if avg_scores:
        overall_score = (avg_scores.get('technical', 0) * 10 + 
                       avg_scores.get('clarity', 0) * 10 + 
                       avg_scores.get('relevance', 0) * 10 + 
                       avg_scores.get('professionalism', 0) * 10) / 4
        elements.append(Paragraph(f"Overall Score: {overall_score:.1f}/10", styles['SubHeader']))
    
    # 3. Performance Metrics Section
    elements.append(PageBreak())
    elements.append(Paragraph("Detailed Metrics", styles['SectionHeader']))
    
    if avg_scores:
        # Scores Table
        data = [
            ["Metric", "Score"],
            ["Technical", f"{avg_scores.get('technical', 0) * 10:.1f}/10"],
            ["Clarity", f"{avg_scores.get('clarity', 0) * 10:.1f}/10"],
            ["Relevance", f"{avg_scores.get('relevance', 0) * 10:.1f}/10"],
            ["Professionalism", f"{avg_scores.get('professionalism', 0) * 10:.1f}/10"]
        ]
        
        t = Table(data, colWidths=[2*inch, 1*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
    
    # 4. Visualization (Plotly with fallback to Matplotlib)
    if avg_scores:
        categories = ['Technical', 'Clarity', 'Relevance', 'Professionalism']
        values = [
            avg_scores.get('technical', 0) * 10,
            avg_scores.get('clarity', 0) * 10,
            avg_scores.get('relevance', 0) * 10,
            avg_scores.get('professionalism', 0) * 10
        ]
        
        # Try Plotly first
        plotly_success = False
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(37, 99, 235, 0.3)',
                line=dict(color='rgba(37, 99, 235, 0.8)')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50),
                width=500,
                height=400
            )
            
            try:
                img_bytes = fig.to_image(format="png")
                img = Image(io.BytesIO(img_bytes), width=5*inch, height=4*inch)
                elements.append(img)
                plotly_success = True
            except Exception:
                plotly_success = False
        except ImportError:
            plotly_success = False
        
        # Fallback to Matplotlib if Plotly fails
        if not plotly_success:
            try:
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.fill(categories, values, 'b', alpha=0.2)
                ax.plot(categories, values, 'b-', linewidth=2)
                ax.set_ylim(0, 10)
                ax.set_theta_offset(np.pi/2)
                ax.set_theta_direction(-1)
                ax.set_rlabel_position(0)
                
                img_bytes = io.BytesIO()
                plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                img_bytes.seek(0)
                img = Image(img_bytes, width=5*inch, height=4*inch)
                elements.append(img)
            except Exception:
                # Final fallback to table
                elements.append(Paragraph("Performance Scores Visualization", styles['SubHeader']))
                elements.append(t)
    
    # 5. Strengths & Improvements
    elements.append(PageBreak())
    elements.append(Paragraph("Strengths & Improvements", styles['SectionHeader']))
    
    elements.append(Paragraph("Technical Strengths:", styles['SubHeader']))
    strengths = report.get("technical_strengths", ["No strengths identified"])
    if isinstance(strengths, list):
        for s in strengths:
            elements.append(Paragraph(f"• {safe_to_string(s)}", styles['BulletPoint']))
    else:
        elements.append(Paragraph(f"• {safe_to_string(strengths)}", styles['BulletPoint']))
    
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Areas for Improvement:", styles['SubHeader']))
    improvements = report.get("technical_improvements", ["No improvements needed"])
    if isinstance(improvements, list):
        for i in improvements:
            elements.append(Paragraph(f"• {safe_to_string(i)}", styles['BulletPoint']))
    else:
        elements.append(Paragraph(f"• {safe_to_string(improvements)}", styles['BulletPoint']))
    
    # 6. Detailed Assessments
    elements.append(PageBreak())
    elements.append(Paragraph("Detailed Assessments", styles['SectionHeader']))
    
    sections = [
        ("Communication Assessment", "communication_assessment"),
        ("Vocabulary & Speech Patterns", "vocabulary_speech"), 
        ("Behavioral Assessment", "behavioral_assessment")
    ]
    
    for title, key in sections:
        elements.append(Paragraph(title, styles['SubHeader']))
        content = safe_to_string(report.get(key, "Not available"))
        elements.append(Paragraph(content, styles['BodyText']))
        elements.append(Spacer(1, 0.2*inch))
    
    # 7. Final Recommendation
    elements.append(Paragraph("Final Recommendation", styles['SectionHeader']))
    final_recommendation = safe_to_string(report.get("final_recommendation", "No recommendation available"))
    elements.append(Paragraph(final_recommendation, styles['BodyText']))
    
    # 8. Question Analysis
    elements.append(PageBreak())
    elements.append(Paragraph("Question Analysis", styles['SectionHeader']))
    
    question_data = []
    for q_id, data in responses.items():
        question_index = int(q_id.split("_")[1]) + 1
        analysis = data.get("analysis", {})
        
        elements.append(Paragraph(f"Question {question_index}", styles['SubHeader']))
        elements.append(Paragraph(safe_to_string(data.get('question', '')), styles['BodyText']))
        
        elements.append(Paragraph("Your Answer:", styles['SubHeader']))
        elements.append(Paragraph(safe_to_string(data.get('translation', '')), styles['BodyText']))
        
        # Scores table
        score_data = [
            ["Metric", "Score"],
            ["Relevance", f"{analysis.get('relevance_score', 0)}/10"],
            ["Clarity", f"{analysis.get('clarity_score', 0)}/10"],
            ["Technical", f"{analysis.get('technical_accuracy', 0)}/10"]
        ]
        
        t = Table(score_data, colWidths=[1.5*inch, 1*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        
        elements.append(Paragraph("Feedback:", styles['SubHeader']))
        feedback = safe_to_string(analysis.get("feedback", "No feedback available."))
        elements.append(Paragraph(feedback, styles['BodyText']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Store for comparison chart
        question_data.append({
            "Question": f"Q{question_index}",
            "Relevance": analysis.get('relevance_score', 0),
            "Clarity": analysis.get('clarity_score', 0),
            "Technical": analysis.get('technical_accuracy', 0)
        })
    
    # 9. Performance Comparison Chart
    if question_data:
        elements.append(PageBreak())
        elements.append(Paragraph("Performance Comparison", styles['SectionHeader']))
        
        # Try Plotly first
        plotly_success = False
        try:
            import plotly.express as px
            df = pd.DataFrame(question_data)
            fig = px.bar(df, 
                        x='Question', 
                        y=['Relevance', 'Clarity', 'Technical'],
                        barmode='group',
                        labels={'value': 'Score', 'variable': 'Metric'},
                        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            fig.update_layout(
                yaxis_range=[0, 10],
                width=500,
                height=400
            )
            
            try:
                img_bytes = fig.to_image(format="png")
                img = Image(io.BytesIO(img_bytes), width=5*inch, height=4*inch)
                elements.append(img)
                plotly_success = True
            except Exception:
                plotly_success = False
        except ImportError:
            plotly_success = False
        
        # Fallback to Matplotlib if Plotly fails
        if not plotly_success:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                df = pd.DataFrame(question_data)
                x = np.arange(len(df))
                width = 0.25
                
                ax.bar(x - width, df['Relevance'], width, label='Relevance', color='#1f77b4')
                ax.bar(x, df['Clarity'], width, label='Clarity', color='#ff7f0e')
                ax.bar(x + width, df['Technical'], width, label='Technical', color='#2ca02c')
                
                ax.set_xticks(x)
                ax.set_xticklabels(df['Question'])
                ax.set_ylim(0, 10)
                ax.legend()
                ax.set_ylabel('Score')
                ax.set_title('Question Performance Comparison')
                
                img_bytes = io.BytesIO()
                plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
                plt.close()
                img_bytes.seek(0)
                img = Image(img_bytes, width=5*inch, height=4*inch)
                elements.append(img)
            except Exception:
                # Final fallback to table
                comparison_table = [
                    ["Question", "Relevance", "Clarity", "Technical"]
                ]
                for q in question_data:
                    comparison_table.append([
                        q['Question'],
                        q['Relevance'],
                        q['Clarity'],
                        q['Technical']
                    ])
                
                t = Table(comparison_table)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(t)
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_report_text(report, responses, overall_score, avg_scores):
    """Generate formatted text report"""
    try:
        buffer = StringIO()
        
        # Header
        buffer.write("# INTERVIEW PERFORMANCE REPORT\n\n")
        buffer.write(f"Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
        
        # Overall Assessment
        buffer.write("## OVERALL ASSESSMENT\n")
        buffer.write(f"{report.get('overall_assessment', 'No assessment available.')}\n\n")
        
        # Performance Metrics
        buffer.write("## PERFORMANCE METRICS\n")
        buffer.write(f"- Overall Score: {overall_score:.1f}/10\n")
        buffer.write(f"- Technical: {avg_scores.get('technical', 0) * 10:.1f}/10\n")
        buffer.write(f"- Clarity: {avg_scores.get('clarity', 0) * 10:.1f}/10\n")
        buffer.write(f"- Relevance: {avg_scores.get('relevance', 0) * 10:.1f}/10\n")
        buffer.write(f"- Professionalism: {avg_scores.get('professionalism', 0) * 10:.1f}/10\n\n")
        
        # Strengths and Improvements
        buffer.write("## STRENGTHS & IMPROVEMENTS\n")
        buffer.write("### Technical Strengths:\n")
        for s in report.get("technical_strengths", ["No strengths identified"]):
            buffer.write(f"- {s}\n")
        
        buffer.write("\n### Areas for Improvement:\n")
        for i in report.get("technical_improvements", ["No improvements needed"]):
            buffer.write(f"- {i}\n")
        
        # Detailed Assessments
        buffer.write("\n## DETAILED ASSESSMENTS\n")
        sections = [
            ("Communication", "communication_assessment"),
            ("Vocabulary & Speech", "vocabulary_speech"), 
            ("Behavioral", "behavioral_assessment")
        ]
        
        for title, key in sections:
            buffer.write(f"### {title} Assessment:\n")
            buffer.write(f"{report.get(key, 'Not available')}\n\n")
        
        # Questions Analysis
        buffer.write("## QUESTION ANALYSIS\n")
        for q_id, data in responses.items():
            question_index = int(q_id.split("_")[1]) + 1
            analysis = data.get("analysis", {})
            
            buffer.write(f"### Question {question_index}\n")
            buffer.write(f"**Question:** {data.get('question', '')}\n")
            buffer.write(f"**Your Answer:** {data.get('translation', '')}\n")
            
            buffer.write("**Scores:**\n")
            buffer.write(f"- Relevance: {analysis.get('relevance_score', 0)}/10\n")
            buffer.write(f"- Clarity: {analysis.get('clarity_score', 0)}/10\n")
            buffer.write(f"- Technical: {analysis.get('technical_accuracy', 0)}/10\n\n")
            
            buffer.write("**Feedback:**\n")
            buffer.write(f"{analysis.get('feedback', 'No feedback available.')}\n\n")
        
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None
    
def render_profile_page():
    # Header with back button
 # Header with working back button
    col1, col2 = st.columns([4, 1])  # Adjust ratio as needed
    with col1:
        st.markdown('<h1 style="color:#2563EB; margin-bottom:0;">Your Profile</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("← Back to Dashboard", 
                    key="profile_back_button",
                    help="Return to main dashboard"):
            st.session_state["current_page"] = "Home"
            st.rerun()
    
    if not st.session_state.get("logged_in"):
        st.warning("Please log in to view your profile")
        return
    
    user_id = get_user_id(st.session_state["username"])
    conn = sqlite3.connect('ace_hire.db')
    cursor = conn.cursor()
    
    # Fetch user profile information
    cursor.execute('SELECT first_name, last_name, email, phone FROM users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()
    
    if user_data:
        first_name, last_name, email, phone = user_data
        
        # ===== Personal Information Card =====
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">👤</span>
                <h2 style="margin:0; color:#2563EB;">Personal Information</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("**First Name**", value=first_name, disabled=True)
                st.text_input("**Email**", value=email, disabled=True)
            with col2:
                st.text_input("**Last Name**", value=last_name, disabled=True)
                st.text_input("**Phone**", value=phone, disabled=True)
        
                # ===== Interview History Section =====
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <span style="font-size:1.5rem; margin-right:0.5rem;">📅</span>
                <h2 style="margin:0; color:#2563EB;">Interview History</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Get all interview sessions for this user
            cursor.execute('''
                SELECT id, session_date, job_title, report_content 
                FROM interview_sessions 
                WHERE user_id = ? 
                ORDER BY session_date DESC
                LIMIT 20
            ''', (user_id,))
            sessions = cursor.fetchall()
            
            if sessions:
                # Add search and filter controls with unique keys
                search_col, filter_col = st.columns(2)
                with search_col:
                    search_term = st.text_input(
                        "Search sessions", 
                        placeholder="Job title or date",
                        key="profile_session_search"  # Unique key
                    )
                with filter_col:
                    date_filter = st.selectbox(
                        "Filter by", 
                        ["All", "Last 7 days", "Last 30 days", "Last 6 months"],
                        key="profile_date_filter"  # Unique key
                    )
                
                filtered_sessions = sessions
                if search_term:
                    filtered_sessions = [s for s in filtered_sessions 
                                       if search_term.lower() in s[2].lower() 
                                       or search_term in str(s[1])]
                
                if date_filter != "All":
                    now = datetime.now()
                    if date_filter == "Last 7 days":
                        cutoff = now - timedelta(days=7)
                    elif date_filter == "Last 30 days":
                        cutoff = now - timedelta(days=30)
                    else:  # Last 6 months
                        cutoff = now - timedelta(days=180)
                    filtered_sessions = [s for s in filtered_sessions 
                                       if datetime.strptime(s[1], '%Y-%m-%d %H:%M:%S') > cutoff]
                
                if not filtered_sessions:
                    st.info("No sessions match your filters")
                else:
                    # Display 2 sessions per row
                    cols_per_row = 2
                    session_cols = st.columns(cols_per_row)
                    
                    for i, session in enumerate(filtered_sessions):
                        session_id, session_date, job_title, report_content = session
                        col = session_cols[i % cols_per_row]
                        
                        with col:
                            with st.container(border=True):
                                st.markdown(f"""
                                <div style="padding:1rem;">
                                    <h4 style="margin-top:0; color:#2563EB;">{job_title}</h4>
                                    <p style="color:#6B7280; margin-bottom:0.5rem;">
                                        <span style="font-weight:600;">Date:</span> {session_date}
                                    </p>
                                    <div style="display:flex; gap:0.5rem; margin-top:1rem;">
                                """, unsafe_allow_html=True)
                                
                                # Download button with unique key
                                st.download_button(
                                    label="Download Report",
                                    data=report_content,
                                    file_name=f"Interview_Report_{session_date}_{job_title}.pdf",
                                    mime="application/pdf",
                                    key=f"download_{session_id}",  # Unique key per session
                                    use_container_width=True
                                )
                                
                                # Delete button with unique key
                                if st.button(
                                    "Delete",
                                    key=f"delete_{session_id}",  # Unique key per session
                                    use_container_width=True,
                                    type="secondary"
                                ):
                                    cursor.execute('DELETE FROM interview_sessions WHERE id = ?', (session_id,))
                                    conn.commit()
                                    st.rerun()
                                
                                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.info("You haven't completed any interview sessions yet")

    
    conn.close()
    
    # ===== Account Management Section =====
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; align-items:center; margin-bottom:1rem;">
            <span style="font-size:1.5rem; margin-right:0.5rem;">⚠️</span>
            <h2 style="margin:0; color:#DC2626;">Account Management</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Delete account flow
        if st.checkbox("I want to delete my account and all data"):
            st.warning("This action cannot be undone. All your data will be permanently deleted.")
            if st.text_input("Type 'DELETE' to confirm") == "DELETE":
                if st.button("Permanently Delete Account", type="primary"):
                    conn = sqlite3.connect('ace_hire.db')
                    cursor = conn.cursor()
                    try:
                        # Delete all user data
                        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
                        cursor.execute('DELETE FROM user_documents WHERE user_id = ?', (user_id,))
                        cursor.execute('DELETE FROM interview_recordings WHERE user_id = ?', (user_id,))
                        cursor.execute('DELETE FROM feedback WHERE user_id = ?', (user_id,))
                        cursor.execute('DELETE FROM interview_sessions WHERE user_id = ?', (user_id,))
                        conn.commit()
                        
                        # Clear session and redirect to home
                        st.session_state.clear()
                        st.session_state["current_page"] = "Home"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting account: {str(e)}")
                    finally:
                        conn.close()

def render_help_desk_page():
    st.header("Help Desk")
    st.write("Welcome to the Help Desk. Please describe your issue or question below.")

    # Display common queries
    st.subheader("Common Queries:")
    common_queries = [
        "1. How do I reset my password?",
        "2. How can I upload my resume?",
        "3. What should I do if I encounter technical issues?",
        "4. How can I prepare for the interview?",
        "5. Where can I find the terms and conditions?"
    ]
    for query in common_queries:
        st.write(query)

    # AI Integration for user queries
    gemini_api_key = "AIzaSyDYQZ9lPnU2z73PydAIYVxd_4m3V8p1nKQ"  # Replace with your actual API key
    gai.configure(api_key=gemini_api_key)

    model = gai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[])
 
    def get_gemini_response(question):
        response = chat.send_message(question)
        return response

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    input = st.text_input("Ask your question:", key="input")
    submit = st.button("Submit")

    if submit and input:
        response = get_gemini_response(input)
        st.session_state['chat_history'].append(("You", input))
        st.subheader("AI Response:")
        st.write(response.text)
        st.session_state["chat_history"].append(("AI", response.text))
    
    st.subheader("Chat History:")
    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")
    
    if st.button("Back to Home", key="back_from_help"):
        st.session_state["current_page"] = "Home"

def render_feedback_page():
    st.header("Feedback")
    feedback_text = st.text_area("Please provide your feedback or report an issue:")
    
    if st.button("Submit Feedback"):
        user_id = get_user_id(st.session_state["username"])
        if feedback_text:
            store_feedback(user_id, feedback_text)
        else:
            st.error("Feedback cannot be empty.")

def main():
    initialize_database()
    initialize_session_state()

    st.markdown("""
<style>
    /* Custom styling for question cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Button hover effects */
    .stButton>button:hover {
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    
    /* Consistent metric cards */
    [data-testid="stMetric"] {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Better divider styling */
    [data-testid="stHorizontalBlock"] hr {
        margin: 2rem 0;
        border-color: #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<style>
    /* Custom styling for the gauge chart */
    .stPlotlyChart {
        border-radius: 12px;
    }
    
    /* Better spacing for expanders */
    .stExpander {
        margin-bottom: 1rem;
    }
    
    /* Consistent card shadows */
    [data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Improved tab styling */
    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }
    .stTabs [role="tab"] {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<style>
    /* Improved file uploader styling */
    .stFileUploader > label {
        border: 2px dashed #E5E7EB;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stFileUploader > label:hover {
        border-color: #2563EB;
        background-color: #F8FAFC;
    }
    
    /* Better radio buttons */
    .stRadio > div {
        display: flex;
        gap: 1rem;
    }
    
    /* Container shadows */
    [data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<style>
    /* Improved button styling */
    .stButton>button {
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Better input fields */
    .stTextInput>div>div>input, 
    .stTextArea>div>div>textarea {
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    /* Consistent card shadows */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

    # Custom CSS for optimized layout
    st.markdown("""
        <style>
            .main-container {
                padding: 2rem 1rem;
            }
            .title {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                color: #1f3d7a;
            }
            .subtitle {
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                color: #555;
            }
            .description {
                font-size: 1.1rem;
                line-height: 1.6;
                margin-bottom: 2rem;
            }
            .feature-point {
                margin-bottom: 0.8rem;
                display: flex;
                align-items: center;
            }
            .feature-icon {
                margin-right: 0.5rem;
                color: #1f3d7a;
            }
            .image-container {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            .primary-btn {
                background-color: #1f3d7a !important;
                color: white !important;
                font-weight: bold !important;
                margin-bottom: 1rem !important;
            }
            .secondary-btn {
                border: 1px solid #1f3d7a !important;
                color: #1f3d7a !important;
                margin-right: 0.5rem !important;
            }
            .auth-buttons {
                position: absolute;
                top: 1rem;
                right: 1rem;
            }
            .testimonial {
                font-style: italic;
                color: #666;
                background: #f9f9f9;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # Clear the page completely before rendering the selected page
    if st.session_state["current_page"] == "Home":
        render_home_page()
    elif st.session_state["current_page"] == "Login":
        render_login_page()
    elif st.session_state["current_page"] == "Register":
        render_register_page()
    elif st.session_state["current_page"] == "Password Reset":
        render_password_reset_page()
    elif st.session_state["current_page"] == "Upload Resume":
        render_upload_resume_page()
    elif st.session_state["current_page"] == "Resume-Job Match":
        render_resume_job_match_page()
    elif st.session_state["current_page"] == "Generate Questions":
        render_generate_questions_page()
    elif st.session_state["current_page"] == "Interview Session":
        render_interview_session_page()
    elif st.session_state["current_page"] == "Final Report":
        render_final_report_page()
    elif st.session_state["current_page"] == "Profile":
        render_profile_page()
    elif st.session_state["current_page"] == "Help":
        render_help_desk_page()
    elif st.session_state["current_page"] == "Feedback":
        render_feedback_page()

if __name__ == "__main__":
    main()