"""FastAPI application for collecting human feedback on generated questions."""
import json
import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Form, Depends, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import init_db, get_db, Feedback
from generator import load_roles, get_random_role, generate_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load roles on startup."""
    init_db()
    load_roles()
    logger.info("Feedback UI started - database initialized, roles loaded")
    yield


app = FastAPI(title="Question Feedback UI", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index(request: Request):
    """Main rating page - displays role info, generated question, and rating form."""
    role = get_random_role()
    question = generate_question(role)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "role": role,
        "question": question,
        "role_json": json.dumps(role),
    })


@app.post("/submit")
async def submit_rating(
    question: str = Form(...),
    role_context: str = Form(...),
    role_id: str = Form(...),
    relevance: int = Form(...),
    clarity: int = Form(...),
    discriminative: int = Form(...),
    comments: str = Form(""),
    db: Session = Depends(get_db)
):
    """Submit a rating and redirect to a new question."""
    feedback = Feedback(
        question=question,
        role_context=role_context,
        role_id=role_id,
        relevance=relevance,
        clarity=clarity,
        discriminative=discriminative,
        comments=comments if comments else None,
        source="generated"
    )
    db.add(feedback)
    db.commit()

    logger.info(f"Feedback submitted: role={role_id}, r={relevance}, c={clarity}, d={discriminative}")

    return RedirectResponse(url="/", status_code=303)


@app.get("/stats")
async def stats_page(request: Request, db: Session = Depends(get_db)):
    """Statistics page with aggregated metrics."""
    # Calculate averages
    stats = db.query(
        func.count(Feedback.id).label("total"),
        func.avg(Feedback.relevance).label("avg_relevance"),
        func.avg(Feedback.clarity).label("avg_clarity"),
        func.avg(Feedback.discriminative).label("avg_discriminative"),
    ).first()

    total_count = stats.total or 0
    avg_relevance = stats.avg_relevance or 0
    avg_clarity = stats.avg_clarity or 0
    avg_discriminative = stats.avg_discriminative or 0
    avg_reward = (avg_relevance + avg_clarity + avg_discriminative) / 15 if total_count > 0 else 0

    # Get recent feedback
    recent_feedback = db.query(Feedback).order_by(Feedback.created_at.desc()).limit(20).all()

    return templates.TemplateResponse("stats.html", {
        "request": request,
        "total_count": total_count,
        "avg_relevance": avg_relevance,
        "avg_clarity": avg_clarity,
        "avg_discriminative": avg_discriminative,
        "avg_reward": avg_reward,
        "recent_feedback": recent_feedback,
    })


@app.get("/api/stats")
async def api_stats(db: Session = Depends(get_db)):
    """JSON statistics endpoint."""
    stats = db.query(
        func.count(Feedback.id).label("total"),
        func.avg(Feedback.relevance).label("avg_relevance"),
        func.avg(Feedback.clarity).label("avg_clarity"),
        func.avg(Feedback.discriminative).label("avg_discriminative"),
    ).first()

    total = stats.total or 0
    avg_r = float(stats.avg_relevance or 0)
    avg_c = float(stats.avg_clarity or 0)
    avg_d = float(stats.avg_discriminative or 0)

    return {
        "total_feedback": total,
        "avg_relevance": round(avg_r, 2),
        "avg_clarity": round(avg_c, 2),
        "avg_discriminative": round(avg_d, 2),
        "avg_reward": round((avg_r + avg_c + avg_d) / 15, 3) if total > 0 else 0,
    }


@app.get("/api/export")
async def export_feedback(
    format: str = Query("json", description="Export format: json or parquet"),
    db: Session = Depends(get_db)
):
    """Export all feedback data for training integration."""
    feedbacks = db.query(Feedback).all()

    data = []
    for f in feedbacks:
        # Normalized reward compatible with training: (r + c + d) / 15
        reward = (f.relevance + f.clarity + f.discriminative) / 15.0
        data.append({
            "question": f.question,
            "role_context": f.role_context,
            "role_id": f.role_id,
            "relevance": f.relevance,
            "clarity": f.clarity,
            "discriminative": f.discriminative,
            "reward": round(reward, 4),
            "comments": f.comments,
            "source": f.source,
            "created_at": f.created_at.isoformat() if f.created_at else None
        })

    if format == "parquet":
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            return StreamingResponse(
                buffer,
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename=feedback.parquet"}
            )
        except ImportError:
            return {"error": "pandas not installed - cannot export parquet"}

    return data


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "feedback-ui"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
