from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas import AnalysisRequest, AnalysisResponse, Question, QuestionResponse
from services.vqa_service import extract_features, generate_answer, analyze_scan
from services.report_service import generate_report
from utils.logging import logger
from models.ct_clip_loader import load_ctclip_model
from models.vqa_model import load_vqa_model

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_scan_endpoint(request: AnalysisRequest):
    """Analyze a scan with questions"""
    scan_id = request.scan_id
    questions = [q.text for q in request.questions]
    
    # Ensure models are loaded
    if not load_ctclip_model() or not load_vqa_model():
        raise HTTPException(status_code=500, detail="Failed to load models")
    
    try:
        # Extract features from the scan
        visual_features = extract_features(scan_id)
        
        # Process each question
        responses = []
        for question in questions:
            answer = generate_answer(visual_features, question)
            responses.append(QuestionResponse(question=question, answer=answer))
        
        # Generate a report if requested
        report_html = None
        for q in questions:
            if "generate report" in q.lower() or "create report" in q.lower():
                report_html = generate_report(visual_features)
                break
        
        return AnalysisResponse(
            scan_id=scan_id,
            responses=responses,
            report_html=report_html
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing scan: {str(e)}")

@router.post("/ask/{scan_id}")
async def ask_question_endpoint(scan_id: str, question: Question):
    """Ask a single question about a scan"""
    try:
        # Ensure models are loaded
        if not load_ctclip_model() or not load_vqa_model():
            raise HTTPException(status_code=500, detail="Failed to load models")
        
        # Extract features
        visual_features = extract_features(scan_id)
        
        # Generate answer
        answer = generate_answer(visual_features, question.text)
        
        return {
            "scan_id": scan_id,
            "question": question.text,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering question for scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")