import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any

def display_match_score_gauge(score: float, title: str = "Overall Match Score") -> None:
    """
    Display a gauge chart for match score visualization
    
    Args:
        score: Score value between 0 and 1
        title: Chart title
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'red'},
                {'range': [30, 70], 'color': 'orange'},
                {'range': [70, 100], 'color': 'green'},
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, b=20, t=40),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_keyword_match_bar(matching_keywords: List[str], 
                              missing_keywords: List[str]) -> None:
    """
    Display a horizontal bar chart for keyword matches
    
    Args:
        matching_keywords: List of matching keywords
        missing_keywords: List of missing keywords
    """
    if not matching_keywords and not missing_keywords:
        st.info("No keywords identified for comparison.")
        return
    
    total = len(matching_keywords) + len(missing_keywords)
    matched_pct = len(matching_keywords) / total if total > 0 else 0
    
    # Create data for visualization
    categories = ["Matching", "Missing"]
    values = [len(matching_keywords), len(missing_keywords)]
    colors = ['green', 'red']
    
    fig = px.bar(
        x=values,
        y=categories,
        orientation='h',
        color=categories,
        color_discrete_sequence=colors,
        labels={'x': 'Count', 'y': 'Keywords'},
        title='Keyword Match Analysis'
    )
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, b=20, t=40),
        showlegend=False,
        paper_bgcolor="white"
    )
    
    # Add percentages as text
    fig.add_annotation(
        x=len(matching_keywords)/2 if len(matching_keywords) > 0 else 0, 
        y="Matching",
        text=f"{matched_pct:.0%}",
        showarrow=False,
        font=dict(color="white", size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_match_details_expander(match_result: Dict[str, Any]) -> None:
    """
    Display expandable section with match details
    
    Args:
        match_result: Dictionary containing match analysis
    """
    with st.expander("View Match Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matching Keywords")
            if match_result.get("matching_keywords"):
                for keyword in sorted(match_result.get("matching_keywords", [])):
                    st.markdown(f"✅ {keyword}")
            else:
                st.info("No matching keywords found.")
        
        with col2:
            st.subheader("Missing Keywords")
            if match_result.get("missing_keywords"):
                for keyword in sorted(match_result.get("missing_keywords", [])):
                    st.markdown(f"❌ {keyword}")
            else:
                st.success("No missing keywords. Great job!")
        
        st.divider()
        
        # Display different scores
        st.subheader("Score Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Keyword Match Score", f"{match_result.get('keyword_match_score', 0) * 100:.1f}%")
        
        with col2:
            st.metric("Content Similarity Score", f"{match_result.get('tfidf_score', 0) * 100:.1f}%")

def display_recommendations(match_result: Dict[str, Any], job_title: str) -> None:
    """
    Display recommendations based on match analysis
    
    Args:
        match_result: Dictionary containing match analysis
        job_title: The job title
    """
    st.subheader("Recommendations")
    
    missing_keywords = match_result.get("missing_keywords", [])
    overall_score = match_result.get("overall_score", 0)
    
    if overall_score >= 0.8:
        st.success("Great job! Your resume is well-aligned with the job requirements.")
        
        if missing_keywords:
            st.info("Consider adding these missing keywords to further strengthen your resume:")
            st.write(", ".join(missing_keywords[:5]))
    
    elif overall_score >= 0.6:
        st.info("Your resume is moderately aligned with the job requirements.")
        
        if missing_keywords:
            st.warning("To improve your match, consider adding these key skills to your resume:")
            st.write(", ".join(missing_keywords[:7]))
    
    else:
        st.warning("Your resume may need significant enhancements for this position.")
        
        if missing_keywords:
            st.error("Critical skills missing from your resume:")
            st.write(", ".join(missing_keywords[:10]))
        
        st.info(f"Consider gaining experience or training in these areas for {job_title} positions.")