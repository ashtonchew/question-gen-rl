"""Database models and setup for feedback storage."""
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, CheckConstraint
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
