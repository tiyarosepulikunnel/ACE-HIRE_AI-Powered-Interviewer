import streamlit as st
import json

def analyze_response(client, response_text, question, job_description=None, response_duration=0):
    """
    Analyze interview response using AI
    
    Args:
        client: OpenAI client instance
        response_text: Transcribed and translated response
        question: The interview question
        job_description: Optional job description for context
        response_duration: Duration of the response in seconds
        
    Returns:
        Dictionary containing analysis results
    """
    if not client or not response_text:
        return {
            "relevance_score": 0,
            "clarity_score": 0,
            "technical_accuracy": 0,
            "sentiment": "neutral",
            "emotion": "neutral",
            "professionalism": 0,
            "words_per_minute": 0,
            "feedback": "No response to analyze."
        }
    
    try:
        # Calculate words per minute
        word_count = len(response_text.split())
        words_per_minute = 0
        if response_duration > 0:
            words_per_minute = int((word_count / response_duration) * 60)
        
        # Create prompt for OpenAI
        system_prompt = """
        You are an expert interview assessor. Analyze the candidate's response to the interview question and provide:
        
        1. Relevance Score (0-10): How relevant the response is to the question asked.
        2. Clarity Score (0-10): How clear and well-articulated the response is.
        3. Technical Accuracy Score (0-10): For technical questions, how accurate and knowledgeable the response is.
        4. Sentiment (positive/negative/neutral): The overall sentiment of the response.
        5. Emotion (confident/nervous/enthusiastic/hesitant/neutral): The dominant emotion conveyed.
        6. Professionalism Score (0-10): How professional and appropriate the response is.
        7. Words Per Minute: The speaking rate (already calculated, just confirm the number).
        8. Feedback: 2-3 sentences of constructive feedback including strengths and areas for improvement.
        
        Format your response as a JSON object with the keys: relevance_score, clarity_score, technical_accuracy, sentiment, emotion, professionalism, words_per_minute, feedback.
        """
        
        user_prompt = f"""
        QUESTION: {question}
        
        RESPONSE: {response_text}
        
        JOB CONTEXT: {job_description[:500] if job_description else ""}
        
        CALCULATED WORDS PER MINUTE: {words_per_minute}
        
        Please analyze this interview response.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        analysis_text = response.choices[0].message.content
        
        # For safety, provide default values in case parsing fails
        analysis = {
            "relevance_score": 5,
            "clarity_score": 5,
            "technical_accuracy": 5,
            "sentiment": "neutral",
            "emotion": "neutral",
            "professionalism": 5,
            "words_per_minute": words_per_minute,
            "feedback": "Analysis unavailable."
        }
        
        # Try to parse the JSON
        try:
            parsed = json.loads(analysis_text)
            # Update with parsed values
            analysis.update(parsed)
        except Exception as json_error:
            st.warning(f"⚠️ Error parsing analysis response: {str(json_error)}")
        
        return analysis
    
    except Exception as e:
        st.error(f"❌ Error analyzing response: {str(e)}")
        return {
            "relevance_score": 0,
            "clarity_score": 0,
            "technical_accuracy": 0,
            "sentiment": "neutral",
            "emotion": "neutral",
            "professionalism": 0,
            "words_per_minute": words_per_minute if 'words_per_minute' in locals() else 0,
            "feedback": f"Error during analysis: {str(e)}"
        }

def generate_comprehensive_report(client, responses, job_title, job_description):
    """
    Generate a comprehensive interview report
    
    Args:
        client: OpenAI client instance
        responses: Dictionary of analyzed responses
        job_title: Job title
        job_description: Job description
        
    Returns:
        Dictionary containing report results
    """
    if not client or not responses:
        return {"report": "Insufficient data to generate report."}
    
    try:
        # Compile all response data
        response_summaries = []
        tech_scores = []
        clarity_scores = []
        relevance_scores = []
        prof_scores = []
        
        for q_id, data in responses.items():
            if "analysis" in data:
                a = data["analysis"]
                
                response_summaries.append({
                    "question": data["question"],
                    "answer": data["translation"][:300] + "..." if len(data["translation"]) > 300 else data["translation"],
                    "relevance": a.get("relevance_score", 0),
                    "clarity": a.get("clarity_score", 0),
                    "technical": a.get("technical_accuracy", 0),
                    "sentiment": a.get("sentiment", "neutral"),
                    "emotion": a.get("emotion", "neutral"),
                    "professionalism": a.get("professionalism", 0),
                    "wpm": a.get("words_per_minute", 0),
                })
                
                tech_scores.append(a.get("technical_accuracy", 0))
                clarity_scores.append(a.get("clarity_score", 0))
                relevance_scores.append(a.get("relevance_score", 0))
                prof_scores.append(a.get("professionalism", 0))
        
        # Calculate averages
        avg_technical = sum(tech_scores) / len(tech_scores) if tech_scores else 0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        avg_prof = sum(prof_scores) / len(prof_scores) if prof_scores else 0
        overall_score = (avg_technical + avg_clarity + avg_relevance + avg_prof) / 4
        
        # Generate comprehensive report
        system_prompt = f"""
        You are an expert interview coach and talent evaluator. Create a comprehensive interview assessment report for a candidate who applied for a {job_title} position.
        
        The report should include:
        
        1. OVERALL ASSESSMENT: An executive summary of the candidate's interview performance (3-4 sentences)
        
        2. TECHNICAL STRENGTHS: 2-3 specific technical strengths demonstrated
        
        3. TECHNICAL AREAS FOR IMPROVEMENT: 2-3 specific technical areas for improvement
        
        4. COMMUNICATION ASSESSMENT: Analysis of clarity, confidence, and articulation (3-4 sentences)
        
        5. VOCABULARY & SPEECH PATTERNS: Commentary on word choice, speaking pace, and speech patterns (2-3 sentences)
        
        6. BEHAVIORAL ASSESSMENT: How well the candidate demonstrated soft skills and cultural fit (2-3 sentences)
        
        7. FINAL RECOMMENDATION: Whether the candidate should proceed in the hiring process, including a confidence level (high/medium/low) and brief justification
        
        Format your response as JSON with these exact keys: overall_assessment, technical_strengths, technical_improvements, communication_assessment, vocabulary_speech, behavioral_assessment, final_recommendation
        """
        
        # Prepare context with all response data
        context = json.dumps({
            "job_title": job_title,
            "response_data": response_summaries,
            "averages": {
                "technical": avg_technical,
                "clarity": avg_clarity,
                "relevance": avg_relevance,
                "professionalism": avg_prof,
                "overall": overall_score
            }
        })
        
        # Generate report
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please generate a comprehensive report based on this interview data: {context}"}
            ],
            temperature=0.7,
            max_tokens=1200,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        report_text = response.choices[0].message.content
        
        # Default report structure
        report = {
            "overall_assessment": "Unable to generate assessment.",
            "technical_strengths": ["Not available"],
            "technical_improvements": ["Not available"],
            "communication_assessment": "Not available",
            "vocabulary_speech": "Not available",
            "behavioral_assessment": "Not available", 
            "final_recommendation": "Not available"
        }
        
        # Try to parse the JSON
        try:
            parsed = json.loads(report_text)
            # Update with parsed values
            report.update(parsed)
        except Exception as json_error:
            st.warning(f"⚠️ Error parsing report response: {str(json_error)}")
        
        # Add the averages to the report
        report["average_scores"] = {
            "technical": avg_technical,
            "clarity": avg_clarity,
            "relevance": avg_relevance,
            "professionalism": avg_prof,
            "overall": overall_score
        }
        
        return report
    
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")
        return {"report": f"Error generating report: {str(e)}"}