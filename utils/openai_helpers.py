import os
import streamlit as st
from openai import OpenAI
def initialize_openai(api_key=None, github_token=None):
    """
    Initialize the OpenAI client with either:
    - A GitHub token via Azure inference endpoint
    - An OpenAI API key directly if Azure endpoint not used
    Args:
        api_key: Optional OpenAI API key for direct API usage
        github_token: Optional GitHub token for Azure inference
    Returns:
        OpenAI client instance
    """
    try:
        # Check if GitHub token is provided or available in environment
        github_token = github_token or os.getenv("GITHUB_TOKEN")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if github_token:
            print("✅ Using GitHub token with Azure inference endpoint.")
            endpoint = "https://models.inference.ai.azure.com"
            model_name = "gpt-4o"
            
            client = OpenAI(
                base_url=endpoint,
                api_key=github_token,  # Pass GitHub token for Azure
            )
            return client
        elif api_key:
            print("✅ Using OpenAI API key for direct OpenAI integration.")
            client = OpenAI(api_key=api_key)
            return client
        
        else:
            raise ValueError("❗️ No valid API key or GitHub token provided!")
    
    except Exception as e:
        print(f"⚠️ Failed to initialize OpenAI client: {str(e)}")
        return None

def generate_interview_questions(client, job_title, job_description, resume_text, match_result, num_questions=3):
    """
    Generate interview questions based on resume and job description
    
    Args:
        client: OpenAI client instance
        job_title: Job title
        job_description: Job description text
        resume_text: Resume text
        match_result: Match analysis result
        num_questions: Number of questions to generate
        
    Returns:
        List of interview questions or None if generation fails
    """
    try:
        # Prepare context for question generation
        missing_keywords = match_result.get("missing_keywords", [])
        matching_keywords = match_result.get("matching_keywords", [])
        
        # Create prompt
        system_prompt = f"""
        You are an expert interviewer for {job_title} positions. Generate exactly {num_questions} interview questions based on the candidate's resume and the job description. The questions should be:

        1. One technical question based on the candidate's resume experience
        2. One technical question related to the job description
        3. One behavioral question
        
        Make the questions sound natural and conversational, not robotic or formulaic.
        
        Format your response as 3 individual questions (not numbered), one per line, starting with a dash (-).
        """
        
        user_prompt = f"""
        JOB TITLE: {job_title}
        
        JOB DESCRIPTION: {job_description}
        
        RESUME: {resume_text[:1500]}
        
        MATCHING KEYWORDS: {', '.join(matching_keywords[:10])}
        
        MISSING KEYWORDS: {', '.join(missing_keywords[:10])}
        
        Please generate {num_questions} interview questions based on this information.
        """
        
        # Make OpenAI API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract and process questions
        answer_text = response.choices[0].message.content
        questions = [q.strip() for q in answer_text.split('\n') if q.strip()]
        
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None