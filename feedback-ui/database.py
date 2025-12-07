"""Database models and setup for feedback storage."""
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, CheckConstraint, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database path
DB_PATH = Path(__file__).parent / "data" / "feedback.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Feedback(Base):
    """Feedback model for storing question ratings."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    role_context = Column(Text, nullable=True)
    role_id = Column(String(50), nullable=True)
    relevance = Column(Integer, nullable=False)
    clarity = Column(Integer, nullable=False)
    discriminative = Column(Integer, nullable=False)
    comments = Column(Text, nullable=True)
    source = Column(String(20), default="generated")  # 'generated' or 'user_provided'
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        CheckConstraint('relevance >= 0 AND relevance <= 10', name='check_relevance'),
        CheckConstraint('clarity >= 0 AND clarity <= 10', name='check_clarity'),
        CheckConstraint('discriminative >= 0 AND discriminative <= 10', name='check_discriminative'),
    )

    def to_dict(self):
        """Convert to dictionary for JSON response."""
        return {
            "id": self.id,
            "question": self.question,
            "role_context": self.role_context,
            "role_id": self.role_id,
            "relevance": self.relevance,
            "clarity": self.clarity,
            "discriminative": self.discriminative,
            "comments": self.comments,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "avg_score": (self.relevance + self.clarity + self.discriminative) / 3
        }


class PendingQuestion(Base):
    """Queue for questions waiting for human feedback (online RL)."""
    __tablename__ = "pending_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(50), unique=True, nullable=False)  # Unique ID from training
    question = Column(Text, nullable=False)
    role_context = Column(Text, nullable=True)
    role_id = Column(String(50), nullable=True)
    status = Column(String(20), default="pending")  # pending, rated, expired
    relevance = Column(Integer, nullable=True)
    clarity = Column(Integer, nullable=True)
    discriminative = Column(Integer, nullable=True)
    reward = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    rated_at = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary for JSON response."""
        return {
            "id": self.id,
            "request_id": self.request_id,
            "question": self.question,
            "role_context": self.role_context,
            "role_id": self.role_id,
            "status": self.status,
            "relevance": self.relevance,
            "clarity": self.clarity,
            "discriminative": self.discriminative,
            "reward": self.reward,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "rated_at": self.rated_at.isoformat() if self.rated_at else None,
        }


def init_db():
    """Initialize the database, creating tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
