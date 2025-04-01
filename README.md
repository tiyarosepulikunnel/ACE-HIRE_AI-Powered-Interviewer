# ACE HIRE - AI Interview Preparation Platform

<img src="Ace_Hire.jpg" alt="Alt Text" width="200" height="200" text-align=center>

Become an ACE HIRE today - your gateway to interview success with AI-powered practice sessions and feedback.

## 🌟 Features

- **Resume-Job Matching**: Analyze how well your resume matches job descriptions
- **Personalized Questions**: AI-generated interview questions tailored to your resume
- **Practice Sessions**: Realistic interview simulations with voice recording
- **Performance Analytics**: Detailed feedback on your responses
- **Multi-language Support**: Practice in multiple Indian languages
- **Progress Tracking**: Save and review your interview history

<p align="center">
  <a href="https://www.youtube.com/watch?v=TEh7-pOmXtA">
    <img src="https://img.youtube.com/vi/TEh7-pOmXtA/maxresdefault.jpg" alt="Demo Video">
  </a>
</p>

<p align="center"><b>Demo Video for Ace Hire</b></p>

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Google Gemini API key (for question generation)
- OpenAI API key (for analysis)

### Installation

1. Clone the repository:

```
   git clone https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer
   cd Ace-Hire_AI-Powered-Interviewer
```

2. Install dependencies:
   1. Basic installation
   ```
      pip install -r requirements.txt
   ```
   2. For GPU support (recommended)
   ```
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   3. Linux audio dependencies
   ```
      sudo apt-get install portaudio19-dev python3-pyaudio
   ```
3. Create a .env file with your API keys:

```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

4. Initialize the database:

```
python -c "from utils.db import initialize_database; initialize_database()"
```

5. Run the application:

```
streamlit run Ace_Hire.py
```

## 🗂️ Project Structure

    ace-hire/
    ├── app.py                # Main application file
    ├── utils/
    │   ├── db.py             # Database operations
    │   ├── nlp_processor.py  # NLP analysis functions
    │   ├── pdf_processor.py  # Resume parsing
    │   └── openai_helpers.py # OpenAI integration
    ├── static/               # Static assets
    ├── requirements.txt      # Dependencies
    └── README.md             # This file

## 📋 Usage Guide

1. Upload Your Resume

   - Upload your resume (PDF/DOCX)
   - Paste or upload the job description

2. Resume Analysis

   - View your resume-job match score
   - See keyword matches and gaps

3. Generate Questions

   - Get personalized interview questions
   - Customize question difficulty

4. Practice Interview

   - Record your answers to questions
   - Get instant feedback on responses

5. Review Performance
   - View comprehensive feedback reports
   - Track progress over time

# 📚 Documentation

For detailed documentation, please visit our [report](https://docs.google.com/document/d/1Ya14k2mQhb7dpngNkGItMhwa9rPPxyGOU7FP-cOv6OI/edit?usp=sharing).

# 🤝 Contributing

We welcome contributions!
Steps to contribute:

1. Fork the repository

2. Create your feature branch:

```
git checkout -b feature/AmazingFeature
```

3. Commit your changes:

```
git commit -m "Add AmazingFeature"
```

4. Push to the branch:

```
git push origin feature/AmazingFeature
```

5. Open a Pull Request<br><br>

## 📸 Screenshots

### 🔐 Authentication

_🔑 Login Page_
![Login Page](https://github.com/user-attachments/assets/c6c69fef-129f-47b8-a1f8-721e4274c91c)

_📝 Registration Page_
![Registration Page](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/Registration.jpg?raw=true)

### 🏠 Main Application

_🏡 Home Page_
![Home Page](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/HomePage.jpg?raw=true)

_📤 Upload Documents Page_
![Upload Documents](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/UploadDocuments.jpg?raw=true)

### ✨ Features

_📄 Resume Analysis_
![Resume Analysis](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/ResumeAnalysis.jpg?raw=true)

_❓ Question Generation_
![Question Generation](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/QuestionGeneration.jpg?raw=true)

_💼 Interview Practice Session_
![Interview Practice Session](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/InterviewPracticeSession.jpg?raw=true)

### 📊 Results

_📑 Final Report_
![Final Report](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/FinalReport.jpg?raw=true)

_👤 User Profile Page_
![Profile](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer/blob/main/UI_Screenshots/FinalReport.jpg?raw=true)

# 📦 Packages & System Requirements

## Minimum System Requirements

| **Component**  | **Requirement**                                    |
| -------------- | -------------------------------------------------- |
| **OS**         | Windows 10+ / macOS 10.15+ / Linux (Ubuntu 20.04+) |
| **Python**     | 3.8+ (64-bit recommended)                          |
| **RAM**        | 4GB minimum (8GB recommended)                      |
| **Storage**    | 500MB available space                              |
| **Webcam**     | Required for video interviews                      |
| **Microphone** | Required for audio recording                       |

## Core Dependencies

| **Package**             | **Version** | **Purpose**                     |
| ----------------------- | ----------- | ------------------------------- |
| **streamlit**           | >=1.22.0    | Web application framework       |
| **python-dotenv**       | >=0.21.0    | Environment variable management |
| **PyPDF2**              | >=3.0.0     | PDF text extraction             |
| **python-docx**         | >=0.8.11    | DOCX file processing            |
| **openai**              | >=0.27.0    | AI question generation          |
| **google-generativeai** | >=0.3.0     | Alternative AI service          |
| **sqlite3**             | Included    | Database operations             |
| **nltk**                | >=3.8.0     | Natural language processing     |
| **pyaudio**             | >=0.2.13    | Audio recording                 |
| **speechrecognition**   | >=3.10.0    | Speech-to-text                  |
| **plotly**              | >=5.15.0    | Data visualization              |
| **pandas**              | >=2.0.0     | Data analysis                   |

## ✅ Verified Configurations

| **OS**               | **Python** | **Status**  | **Notes**                 |
| -------------------- | ---------- | ----------- | ------------------------- |
| **Windows 11**       | 3.10       | ✅ Verified | Best with NVIDIA GPU      |
| **macOS Ventura**    | 3.9        | ✅ Verified | M1/M2 native support      |
| **Ubuntu 22.04 LTS** | 3.8        | ✅ Verified | Requires `libasound2-dev` |

> **Note:** For M1/M2 Mac users, we recommend using **Conda** for better compatibility with audio packages.

## 🚀 Future Roadmap

### ⏳ Near-Term (0-6 months)

- 🎯 Domain expansion to finance/engineering roles
- 🔗 LinkedIn API integration
- 🔇 Advanced noise cancellation algorithms

### ⌛ Mid-Term (6-12 months)

- 😊 Real-time facial expression analysis
- 🎚️ Dynamic difficulty adjustment
- ♾️ Neurodiversity support modules

### 🔭 Long-Term Vision

- 🫸🏽➡️ Full interview lifecycle platform
- 🏷️ Employer-branded customization
- 🔮 Predictive hiring success scoring
- 🕶️ VR interview simulations

## 📜 License

Distributed under the MIT `(License)`. See LICENSE for more information.

## Team Members and Roles

| **Member**           | **Role**                            | **Responsibilities**                                                                           | **Contact**                        |
| -------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Priyanka Bhat**    | Backend Developer                   | Backend Development                                                                            | priyankabhatt0022@gmail.com        |
| **Amrutheshwari V.** | Documentation                       | Report Formation and Parametric Design                                                         | amrutheshwariv@gmail.com           |
| **Tiya Rose**        | Frontend Designer                   | UI Design                                                                                      | tiyarosepulikunnel@gmail.com       |
| **Lavanya HS**       | Researcher                          | Problem statement and existing work research                                                   | lavanyahs865@gmail.com             |
| **Gajendiran**       | User-Centric Functionality Designer | User-centric designs (Help Desk page & Feedback session page)                                  | 2023gajendiran.a@vidyashilp.edu.in |
| **Sairaj Bhise**     | Full-Stack Developer                | Integration of frontend and backend code, frontend development & optimization, UX optimization | sairajbhise2005@gmail.com          |

### Contributors

- **Priyanka Bhat** - Backend Development ([@pbhatt0022](https://github.com/pbhatt0022))
- **Amrutheshwari V.** - Documentation ([@Amrutheshwari01](https://github.com/Amrutheshwari01))
- **Tiya Rose** - Frontend Design ([@tiyarosepulikunnel](https://github.com/tiyarosepulikunnel))
- **Lavanya HS** - Research ([@lavanya-hs15](https://github.com/lavanya-hs15))
- **Gajendiran** - User-Friendly Design Implementation([@GajendiranA](https://github.com/GajendiranA))
- **Sairaj Bhise** - Integration of Frontend-centric and backend-centric code([@SairajBhise2005](https://github.com/SairajBhise2005))

## 📧 Contact

Project Team - Team Ace Hire<br>
Project Link: [ACE-HIRE_AI-Powered-Interviewer](https://github.com/SairajBhise2005/ACE-HIRE_AI-Powered-Interviewer)
