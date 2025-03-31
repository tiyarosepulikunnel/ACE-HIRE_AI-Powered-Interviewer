import json
from datetime import datetime
from sqlalchemy.orm import Session as DbSession
from models.database import SessionLocal, User, Session, Analysis, Interview

class DataStore:
    def __init__(self):
        try:
            self.db: DbSession = SessionLocal()
        except Exception as e:
            print(f"Failed to initialize database connection: {e}")
            raise

    def __del__(self):
        if hasattr(self, 'db'):
            try:
                self.db.close()
            except Exception as e:
                print(f"Error closing database connection: {e}")

    def save_session(self, session_id: str, data: dict):
        """Save session data to database."""
        try:
            # Create a new user if this is the first session
            user = User()
            self.db.add(user)
            self.db.flush()

            # Check if session exists
            db_session = self.db.query(Session).filter(Session.session_id == session_id).first()

            if not db_session:
                # Create new session
                db_session = Session(
                    session_id=session_id,
                    user_id=user.id,
                    job_title=data.get('job_title', ''),
                    job_description=data.get('job_description', ''),
                    resume_text=data.get('resume_text', '')
                )
                self.db.add(db_session)
            else:
                # Update existing session
                db_session.job_title = data.get('job_title', db_session.job_title)
                db_session.job_description = data.get('job_description', db_session.job_description)
                db_session.resume_text = data.get('resume_text', db_session.resume_text)

            self.db.commit()
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            self.db.rollback()
            return False

    def load_session(self, session_id: str):
        """Load session data from database."""
        try:
            db_session = self.db.query(Session).filter(Session.session_id == session_id).first()
            if db_session:
                return {
                    'job_title': db_session.job_title,
                    'job_description': db_session.job_description,
                    'resume_text': db_session.resume_text
                }
            return None
        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    def save_analysis(self, session_id: str, analysis_data: dict):
        """Save resume analysis results."""
        try:
            # Get session
            db_session = self.db.query(Session).filter(Session.session_id == session_id).first()
            if not db_session:
                print(f"No session found with id: {session_id}")
                return False

            # Create analysis
            analysis = Analysis(
                session_id=db_session.id,
                overall_score=analysis_data.get('match_result', {}).get('overall_score', 0),
                skill_match_score=analysis_data.get('match_result', {}).get('skill_match_score', 0),
                matching_keywords=json.dumps(analysis_data.get('match_result', {}).get('matching_keywords', [])),
                missing_keywords=json.dumps(analysis_data.get('match_result', {}).get('missing_keywords', []))
            )

            self.db.add(analysis)
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error saving analysis: {e}")
            self.db.rollback()
            return False

    def load_analysis(self, session_id: str):
        """Load resume analysis results."""
        try:
            db_session = self.db.query(Session).filter(Session.session_id == session_id).first()
            if not db_session:
                return None

            analysis = self.db.query(Analysis).filter(Analysis.session_id == db_session.id).first()
            if analysis:
                return {
                    'match_result': {
                        'overall_score': analysis.overall_score,
                        'skill_match_score': analysis.skill_match_score,
                        'matching_keywords': json.loads(analysis.matching_keywords),
                        'missing_keywords': json.loads(analysis.missing_keywords)
                    }
                }
            return None
        except Exception as e:
            print(f"Error loading analysis: {e}")
            return None

    def save_interview(self, session_id: str, interview_data: dict):
        """Save interview session data."""
        try:
            db_session = self.db.query(Session).filter(Session.session_id == session_id).first()
            if not db_session:
                print(f"No session found with id: {session_id}")
                return False

            interview = Interview(
                session_id=db_session.id,
                questions=json.dumps(interview_data.get('questions', [])),
                answers=json.dumps(interview_data.get('answers', {})),
                feedback=json.dumps(interview_data.get('feedback', {}))
            )

            self.db.add(interview)
            self.db.commit()
            return True
        except Exception as e:
            print(f"Error saving interview: {e}")
            self.db.rollback()
            return False

    def load_interview(self, session_id: str):
        """Load interview session data."""
        try:
            db_session = self.db.query(Session).filter(Session.session_id == session_id).first()
            if not db_session:
                return None

            interview = self.db.query(Interview).filter(Interview.session_id == db_session.id).first()
            if interview:
                return {
                    'questions': json.loads(interview.questions),
                    'answers': json.loads(interview.answers),
                    'feedback': json.loads(interview.feedback)
                }
            return None
        except Exception as e:
            print(f"Error loading interview: {e}")
            return None