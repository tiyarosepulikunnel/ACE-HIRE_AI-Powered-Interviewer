from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy import text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from datetime import datetime

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Create engine and session
try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Failed to create database engine: {e}")
    raise

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    job_title = Column(String)
    job_description = Column(Text)
    resume_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,
                        default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    analysis = relationship("Analysis", back_populates="session")
    interview = relationship("Interview", back_populates="session")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    overall_score = Column(Float)
    skill_match_score = Column(Float)
    matching_keywords = Column(Text)  # Stored as JSON string
    missing_keywords = Column(Text)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="analysis")


class Interview(Base):
    __tablename__ = "interviews"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    questions = Column(Text)  # Stored as JSON string
    answers = Column(Text)  # Stored as JSON string
    feedback = Column(Text)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="interview")


def init_db():
    """Initialize database with proper error handling."""
    try:
        # Create a test session to verify connection
        session = SessionLocal()

        # Corrected to use text() for raw SQL
        session.execute(text("SELECT 1"))

        session.close()

        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database initialization successful.")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise
