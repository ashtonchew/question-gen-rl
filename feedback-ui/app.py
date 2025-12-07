"""FastAPI application for collecting human feedback on generated questions."""
import json
import io
import logging
import subprocess
import os
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Form, Depends, Query
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import init_db, get_db, Feedback, PendingQuestion
from generator import load_roles, get_random_role, generate_question, _roles, start_warming, get_prewarmed_question, get_cache_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load roles on startup."""
    init_db()
    load_roles()
    start_warming()  # Start pre-warming questions in background
    logger.info("Feedback UI started - database initialized, roles loaded, warming started")
    yield


app = FastAPI(title="Question Feedback UI", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_role_by_id(role_id: str):
    """Get a specific role by ID."""
    if not _roles:
        load_roles()
    for role in _roles:
        if role.get('id') == role_id:
            return role
    return None


@app.get("/")
async def index(
    request: Request,
    regenerate: Optional[int] = Query(None),
    role_id: Optional[str] = Query(None)
):
    """Main rating page - displays role info, generated question, and rating form."""
    # If regenerating for a specific role, use that role (can't use cache)
    if regenerate and role_id:
        role = get_role_by_id(role_id)
        if not role:
            role = get_random_role()
        question = generate_question(role)
    else:
        # Use pre-warmed question from cache for instant loading
        role, question = get_prewarmed_question()

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
    source: str = Form("generated"),
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
        source=source
    )
    db.add(feedback)
    db.commit()

    logger.info(f"Feedback submitted: role={role_id}, r={relevance}, c={clarity}, d={discriminative}")

    return RedirectResponse(url="/?success=1", status_code=303)


@app.get("/feedback/{feedback_id}")
async def feedback_detail(feedback_id: int, request: Request, db: Session = Depends(get_db)):
    """View details of a specific feedback entry."""
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not feedback:
        return RedirectResponse(url="/stats", status_code=303)

    # Parse role context JSON
    role_context = None
    if feedback.role_context:
        try:
            role_context = json.loads(feedback.role_context)
        except json.JSONDecodeError:
            role_context = {"raw": feedback.role_context}

    reward = (feedback.relevance + feedback.clarity + feedback.discriminative) / 30.0

    return templates.TemplateResponse("feedback_detail.html", {
        "request": request,
        "feedback": feedback,
        "role_context": role_context,
        "reward": reward,
    })


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
    avg_reward = (avg_relevance + avg_clarity + avg_discriminative) / 30 if total_count > 0 else 0

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
        "avg_reward": round((avg_r + avg_c + avg_d) / 30, 3) if total > 0 else 0,
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
        # Normalized reward compatible with training: (r + c + d) / 30 for 0-1 range
        reward = (f.relevance + f.clarity + f.discriminative) / 30.0
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


@app.get("/api/regenerate")
async def regenerate(
    role_id: Optional[str] = Query(None),
    new_role: bool = Query(False)
):
    """Regenerate question (and optionally role) via AJAX."""
    if new_role or not role_id:
        role = get_random_role()
    else:
        role = get_role_by_id(role_id)
        if not role:
            role = get_random_role()

    question = generate_question(role)

    return {
        "role": role,
        "question": question
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "feedback-ui"}


@app.get("/api/cache-status")
async def cache_status():
    """Get question cache status for debugging."""
    return get_cache_status()


# ============== Queue API for Online RL ==============

class QueueSubmitRequest(BaseModel):
    """Request to submit a question to the queue for human rating."""
    request_id: str
    question: str
    role_context: Optional[str] = None
    role_id: Optional[str] = None


class QueueRateRequest(BaseModel):
    """Request to rate a pending question."""
    relevance: int
    clarity: int
    discriminative: int


@app.post("/api/queue/submit")
async def queue_submit(req: QueueSubmitRequest, db: Session = Depends(get_db)):
    """Submit a question to the queue for human rating (called by training)."""
    # Check if request_id already exists
    existing = db.query(PendingQuestion).filter(
        PendingQuestion.request_id == req.request_id
    ).first()
    if existing:
        return {"success": False, "message": "Request ID already exists"}

    pending = PendingQuestion(
        request_id=req.request_id,
        question=req.question,
        role_context=req.role_context,
        role_id=req.role_id,
        status="pending"
    )
    db.add(pending)
    db.commit()

    logger.info(f"Question queued for rating: {req.request_id}")
    return {"success": True, "id": pending.id}


@app.get("/api/queue/pending")
async def queue_pending(db: Session = Depends(get_db)):
    """Get all pending questions waiting for rating."""
    pending = db.query(PendingQuestion).filter(
        PendingQuestion.status == "pending"
    ).order_by(PendingQuestion.created_at.asc()).all()
    return {"pending": [p.to_dict() for p in pending], "count": len(pending)}


@app.get("/api/queue/next")
async def queue_next(db: Session = Depends(get_db)):
    """Get the next pending question to rate."""
    pending = db.query(PendingQuestion).filter(
        PendingQuestion.status == "pending"
    ).order_by(PendingQuestion.created_at.asc()).first()
    if not pending:
        return {"question": None}
    return {"question": pending.to_dict()}


@app.post("/api/queue/rate/{request_id}")
async def queue_rate(request_id: str, req: QueueRateRequest, db: Session = Depends(get_db)):
    """Submit a rating for a pending question."""
    pending = db.query(PendingQuestion).filter(
        PendingQuestion.request_id == request_id
    ).first()
    if not pending:
        return {"success": False, "message": "Request not found"}
    if pending.status != "pending":
        return {"success": False, "message": f"Question already {pending.status}"}

    # Calculate reward (same as feedback: (r+c+d)/30 for 0-1 range)
    reward = (req.relevance + req.clarity + req.discriminative) / 30.0

    pending.relevance = req.relevance
    pending.clarity = req.clarity
    pending.discriminative = req.discriminative
    pending.reward = reward
    pending.status = "rated"
    pending.rated_at = datetime.utcnow()
    db.commit()

    logger.info(f"Question rated: {request_id}, reward={reward:.3f}")
    return {"success": True, "reward": reward}


@app.get("/api/queue/result/{request_id}")
async def queue_result(request_id: str, db: Session = Depends(get_db)):
    """Get the result for a specific request (polled by training)."""
    pending = db.query(PendingQuestion).filter(
        PendingQuestion.request_id == request_id
    ).first()
    if not pending:
        return {"status": "not_found", "reward": None}
    if pending.status == "pending":
        return {"status": "pending", "reward": None}
    return {
        "status": pending.status,
        "reward": pending.reward,
        "relevance": pending.relevance,
        "clarity": pending.clarity,
        "discriminative": pending.discriminative
    }


@app.delete("/api/queue/clear")
async def queue_clear(db: Session = Depends(get_db)):
    """Clear all pending questions (admin action)."""
    deleted = db.query(PendingQuestion).filter(
        PendingQuestion.status == "pending"
    ).delete()
    db.commit()
    return {"success": True, "deleted": deleted}


# ============== Training Control ==============

# Track training process
_training_process = None


@app.post("/api/start-training")
async def start_training(mode: str = Query("online_sim", description="Training mode: offline, online, or online_sim")):
    """Start RL training or show simulated visualization."""
    global _training_process

    # Simulated mode - just return success for visualization
    if mode == "online_sim":
        return {
            "success": True,
            "message": "Training simulation started",
            "mode": mode,
            "simulated": True
        }

    # Real training modes
    if _training_process is not None and _training_process.poll() is None:
        return {"success": False, "message": "Training is already running"}

    # Select config based on mode
    if mode == "offline":
        config_file = "configs/offline_config.yaml"
    else:
        config_file = "configs/online_config.yaml"

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        _training_process = subprocess.Popen(
            ["python", "-m", "src.recruiter.main", f"--config-name={config_file.replace('configs/', '').replace('.yaml', '')}"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info(f"Started {mode} training with PID: {_training_process.pid}")
        return {
            "success": True,
            "message": f"{mode.title()} training started (PID: {_training_process.pid})",
            "mode": mode,
            "simulated": False
        }

    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return {"success": False, "message": f"Failed to start training: {str(e)}"}


@app.get("/api/training-data")
async def get_training_data():
    """Get training data from the results file for visualization."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_file = os.path.join(project_root, "results", "rl_training_run_2025-12-07.json")

        with open(results_file, 'r') as f:
            data = json.load(f)

        return data
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
