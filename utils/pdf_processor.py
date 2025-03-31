import io
import streamlit as st
import re
from typing import Optional, Dict, Any

# Try to import fitz (PyMuPDF), but have a fallback if it fails
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF not available. PDF extraction will use fallback method.")

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """
    Extract text content from a PDF file
    
    Args:
        pdf_file: The uploaded PDF file
    
    Returns:
        str: The extracted text content or a placeholder if extraction failed
    """
    # Check if the PDF file was actually uploaded
    if pdf_file is None:
        print("No PDF file was provided")
        return None
    
    # If PyMuPDF is not available, provide a sample resume text
    if not PYMUPDF_AVAILABLE:
        print("PyMuPDF not available - using sample resume text")
        # Let the user know about the issue using streamlit
        st.warning("PDF extraction library (PyMuPDF) is not properly configured. Using sample resume text instead.")
        
        # Generate a sample resume text for testing
        sample_text = """
        RESUME
        
        SUMMARY
        Experienced data analyst with strong background in Python, SQL, and data visualization.
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology - 2018-2022
        
        EXPERIENCE
        Data Analyst - ABC Corporation
        June 2022 - Present
        - Analyzed large datasets using Python and SQL
        - Created visualizations with Tableau and PowerBI
        - Developed machine learning models for predictive analytics
        
        Junior Developer - XYZ Tech
        January 2020 - May 2022
        - Assisted with web application development
        - Worked with databases and REST APIs
        
        SKILLS
        Programming: Python, SQL, R
        Tools: Pandas, NumPy, Scikit-learn, TensorFlow
        Visualization: Tableau, PowerBI, Matplotlib
        Soft Skills: Problem-solving, Communication, Teamwork
        """
        return sample_text
    
    # Use PyMuPDF if it's available
    try:
        # Get the bytes from the uploaded file
        pdf_bytes = pdf_file.getvalue()
        
        # Open the PDF document
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Initialize an empty string to store text
        text = ""
        
        # Check if document is empty
        if doc.page_count == 0:
            return "This PDF document has no pages. Please upload a valid resume."
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text
        
        # If no text was extracted but the PDF has pages
        if not text.strip() and doc.page_count > 0:
            st.warning("The PDF appears to contain no extractable text (possibly a scanned document). Using sample resume text instead.")
            return """
            RESUME
            
            SUMMARY
            Experienced professional with background in the field.
            
            EDUCATION
            Bachelor's Degree - University
            
            EXPERIENCE
            Professional Position - Company
            - Responsibility 1
            - Responsibility 2
            
            SKILLS
            Technical skills, Soft skills, Industry knowledge
            """
        
        return text
    except Exception as e:
        error_message = str(e)
        print(f"PDF extraction error: {error_message}")
        st.error(f"Error processing PDF: {error_message}")
        
        # Show a more friendly message to the user
        st.info("Using a sample resume for demonstration. Upload a different PDF or continue with the sample.")
        
        # Return a sample resume
        return """
        RESUME
        
        SUMMARY
        Professional with experience in relevant field.
        
        EDUCATION
        Bachelor's Degree - University
        
        EXPERIENCE
        Position - Company
        - Achievements
        - Responsibilities
        
        SKILLS
        Technical skills, Tools, Software, Soft skills
        """

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace, newlines, etc.
    
    Args:
        text: The text to clean
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters and clean up
    text = text.strip()
    
    return text

def extract_resume_sections(text: str) -> Dict[str, str]:
    """
    Attempt to extract common resume sections
    
    Args:
        text: The resume text
    
    Returns:
        dict: Dictionary of resume sections
    """
    sections = {
        'education': '',
        'experience': '',
        'skills': '',
        'summary': '',
        'full_text': text
    }
    
    # Simple pattern matching for common sections
    # This could be enhanced with more sophisticated NLP techniques
    lower_text = text.lower()
    
    # Extract education section
    edu_matches = re.search(r'education(.*?)(experience|skills|projects)', lower_text, re.DOTALL | re.IGNORECASE)
    if edu_matches:
        sections['education'] = clean_text(edu_matches.group(1))
    
    # Extract experience section
    exp_matches = re.search(r'experience(.*?)(education|skills|projects)', lower_text, re.DOTALL | re.IGNORECASE)
    if exp_matches:
        sections['experience'] = clean_text(exp_matches.group(1))
    
    # Extract skills section
    skills_matches = re.search(r'skills(.*?)(education|experience|projects)', lower_text, re.DOTALL | re.IGNORECASE)
    if skills_matches:
        sections['skills'] = clean_text(skills_matches.group(1))
    
    # Extract summary section
    summary_matches = re.search(r'(summary|profile|objective)(.*?)(education|experience|skills)', lower_text, re.DOTALL | re.IGNORECASE)
    if summary_matches:
        sections['summary'] = clean_text(summary_matches.group(2))
    
    return sections
