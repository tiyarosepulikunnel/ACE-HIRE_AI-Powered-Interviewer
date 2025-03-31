import re
from typing import Dict, List, Optional
import spacy
from spacy.matcher import PhraseMatcher
from collections import defaultdict

# Load English language model
nlp = spacy.load("en_core_web_sm")

def analyze_resume(resume_text: str, sections: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Analyze resume text and extract key information
    
    Args:
        resume_text (str): Cleaned resume text
        sections (dict): Dictionary of resume sections
        
    Returns:
        dict: Dictionary containing extracted information
    """
    result = {
        'skills': extract_skills(resume_text),
        'experience': extract_experience(sections.get('experience', '')),
        'education': extract_education(sections.get('education', '')),
        'certifications': extract_certifications(sections.get('certifications', '')),
        'projects': extract_projects(sections.get('projects', ''))
    }
    return result

def extract_skills(text: str) -> List[str]:
    """
    Extract skills from resume text
    
    Args:
        text (str): Resume text
        
    Returns:
        list: List of skills
    """
    # Common skills to look for
    skills = [
        "Python", "Java", "C++", "JavaScript", "SQL", "HTML", "CSS",
        "Machine Learning", "Data Analysis", "AWS", "Azure", "Docker",
        "Kubernetes", "Git", "React", "Angular", "Node.js", "TensorFlow",
        "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Tableau", "Power BI"
    ]
    
    # Create PhraseMatcher object
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp(text) for text in skills]
    matcher.add("SKILLS", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text)
    
    return list(found_skills)

def extract_experience(text: str) -> List[str]:
    """
    Extract work experience from resume text
    
    Args:
        text (str): Text from experience section of resume
        
    Returns:
        list: List of job positions
    """
    experience = []
    
    # Pattern to match job titles and companies
    pattern = r'(?:Senior|Junior)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:at|@|,)\s*([A-Z][a-zA-Z0-9\s&]+)'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        # Changed from match.text to match.group()
        experience.append(match.group().strip())
    
    return experience

def extract_education(text: str) -> List[str]:
    """
    Extract education information from resume text
    
    Args:
        text (str): Text from education section of resume
        
    Returns:
        list: List of education entries
    """
    education = []
    
    # Patterns for different degree types
    patterns = [
        r'(?:B\.?A\.?|B\.?S\.?|B\.?E\.?|Bachelor(?:[\'\']s)?)\s*(?:of\s*)?\w*\s*(?:in|,)?\s*([A-Za-z]+(?: [A-Za-z]+)*)',
        r'(?:M\.?A\.?|M\.?S\.?|M\.?E\.?|Master(?:[\'\']s)?)\s*(?:of\s*)?\w*\s*(?:in|,)?\s*([A-Za-z]+(?: [A-Za-z]+)*)',
        r'(?:Ph\.?D\.?|Doctorate)\s*(?:of\s*)?\w*\s*(?:in|,)?\s*([A-Za-z]+(?: [A-Za-z]+)*)',
        r'Associate(?:[\'\']s)?\s*(?:of\s*)?\w*\s*(?:in|,)?\s*([A-Za-z]+(?: [A-Za-z]+)*)'
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Changed from match.text to match.group()
            education.append(match.group().strip())
    
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in education if not (x in seen or seen.add(x))]

def extract_certifications(text: str) -> List[str]:
    """
    Extract certifications from resume text
    
    Args:
        text (str): Text from certifications section
        
    Returns:
        list: List of certifications
    """
    certifications = []
    
    # Pattern to match certifications
    pattern = r'(?:Certified|Certification)\s*(?:in)?\s*([A-Z][a-zA-Z0-9\s]+)'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        # Changed from match.text to match.group()
        certifications.append(match.group().strip())
    
    return certifications

def extract_projects(text: str) -> List[str]:
    """
    Extract projects from resume text
    
    Args:
        text (str): Text from projects section
        
    Returns:
        list: List of projects
    """
    projects = []
    
    # Pattern to match project names and descriptions
    pattern = r'(?:Project|Developed|Created)\s*:\s*([A-Z][a-zA-Z0-9\s\-]+)'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        # Changed from match.text to match.group()
        projects.append(match.group().strip())
    
    return projects

def analyze_job_description(jd_text: str) -> Dict[str, List[str]]:
    """
    Analyze job description text and extract key information
    
    Args:
        jd_text (str): Job description text
        
    Returns:
        dict: Dictionary containing extracted information
    """
    result = {
        'required_skills': extract_skills(jd_text),
        'preferred_skills': extract_preferred_skills(jd_text),
        'education_requirements': extract_education_requirements(jd_text),
        'experience_requirements': extract_experience_requirements(jd_text)
    }
    return result

def extract_preferred_skills(text: str) -> List[str]:
    """
    Extract preferred skills from job description
    
    Args:
        text (str): Job description text
        
    Returns:
        list: List of preferred skills
    """
    preferred_skills = []
    
    # Look for phrases indicating preferred skills
    pattern = r'(?:Preferred|Bonus|Nice to have)\s*:\s*([A-Za-z0-9\s,]+)'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        # Changed from match.text to match.group(1) to get the captured group
        preferred_skills.extend([s.strip() for s in match.group(1).split(',')])
    
    return preferred_skills

def extract_education_requirements(text: str) -> List[str]:
    """
    Extract education requirements from job description
    
    Args:
        text (str): Job description text
        
    Returns:
        list: List of education requirements
    """
    requirements = []
    
    # Pattern to match education requirements
    pattern = r'(?:Degree|Education)\s*(?:in|requirements?)?\s*:\s*([A-Za-z0-9\s,]+)'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        # Changed from match.text to match.group(1)
        requirements.append(match.group(1).strip())
    
    return requirements

def extract_experience_requirements(text: str) -> List[str]:
    """
    Extract experience requirements from job description
    
    Args:
        text (str): Job description text
        
    Returns:
        list: List of experience requirements
    """
    requirements = []
    
    # Pattern to match experience requirements
    pattern = r'(?:Experience|Years)\s*(?:requirements?)?\s*:\s*([0-9+\s+years?]+)'
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        # Changed from match.text to match.group(1)
        requirements.append(match.group(1).strip())
    
    return requirements

def calculate_match_score(resume_analysis: Dict, job_analysis: Dict) -> Dict:
    """
    Calculate match score between resume and job description
    
    Args:
        resume_analysis (dict): Analysis of resume
        job_analysis (dict): Analysis of job description
        
    Returns:
        dict: Match results including score and details
    """
    result = {
        'overall_score': 0,
        'skill_match': {
            'matching': [],
            'missing': [],
            'score': 0
        },
        'education_match': {
            'matching': [],
            'missing': [],
            'score': 0
        },
        'experience_match': {
            'matching': [],
            'missing': [],
            'score': 0
        }
    }
    
    # Calculate skill match
    required_skills = job_analysis.get('required_skills', [])
    resume_skills = resume_analysis.get('skills', [])
    
    matching_skills = [skill for skill in required_skills if skill in resume_skills]
    missing_skills = [skill for skill in required_skills if skill not in resume_skills]
    
    result['skill_match']['matching'] = matching_skills
    result['skill_match']['missing'] = missing_skills
    result['skill_match']['score'] = len(matching_skills) / max(1, len(required_skills))
    
    # Calculate education match
    required_education = job_analysis.get('education_requirements', [])
    resume_education = resume_analysis.get('education', [])
    
    matching_education = []
    for req in required_education:
        for edu in resume_education:
            if req.lower() in edu.lower():
                matching_education.append(edu)
                break
    
    missing_education = [req for req in required_education 
                        if not any(req.lower() in edu.lower() for edu in resume_education)]
    
    result['education_match']['matching'] = matching_education
    result['education_match']['missing'] = missing_education
    result['education_match']['score'] = len(matching_education) / max(1, len(required_education))
    
    # Calculate overall score (weighted average)
    skill_weight = 0.5
    education_weight = 0.3
    experience_weight = 0.2
    
    result['overall_score'] = (
        result['skill_match']['score'] * skill_weight +
        result['education_match']['score'] * education_weight
    )
    
    # Add matching and missing keywords for display
    result['matching_keywords'] = matching_skills + matching_education
    result['missing_keywords'] = missing_skills + missing_education
    
    return result