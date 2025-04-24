from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas import AnalysisRequest, AnalysisResponse, Question, QuestionResponse
from services.vqa_service import extract_features, generate_answer
from services.report_service import generate_report
from utils.logging import logger
from models.ct_clip_loader import load_ctclip_model
from models.vqa_model import load_vqa_model
from utils.file_utils import load_scan_data

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_scan_endpoint(request: AnalysisRequest):
    """Analyze a scan with questions"""
    scan_id = request.scan_id
    questions = [q.text for q in request.questions]
    
    logger.info(f"Received analysis request for scan {scan_id} with {len(questions)} questions")
    
    # Check if models are loaded
    models_loaded = True
    try:
        if not load_ctclip_model() or not load_vqa_model():
            logger.warning("Models not fully loaded, using simplified analysis")
            models_loaded = False
    except Exception as e:
        logger.error(f"Error checking model status: {e}")
        models_loaded = False
    
    try:
        # Process responses
        responses = []
        report_html = None
        
        if models_loaded:
            # Extract features from the scan
            visual_features = extract_features(scan_id)
            
            # Process each question
            for question in questions:
                answer = generate_answer(visual_features, question)
                responses.append(QuestionResponse(question=question, answer=answer))
            
            # Generate a report if requested
            for q in questions:
                if "generate report" in q.lower() or "create report" in q.lower():
                    # Try to use the VQA-based report generator
                    try:
                        report_html = generate_report(visual_features)
                    except Exception as report_err:
                        logger.error(f"Error with VQA report: {report_err}")
                        # Fall back to the simplified report
                        from services.report_service import generate_ctpa_report
                        scan_data = load_scan_data(scan_id)
                        report_html = generate_ctpa_report(scan_data)
                    break
        else:
            # Use fallback responses and report
            for question in questions:
                answer = "AI model not available. Please try again later."
                responses.append(QuestionResponse(question=question, answer=answer))
            
            # Generate simple report if requested
            for q in questions:
                if "generate report" in q.lower() or "create report" in q.lower():
                    # Use the simplified report generator
                    from services.report_service import generate_ctpa_report
                    scan_data = load_scan_data(scan_id)
                    report_html = generate_ctpa_report(scan_data)
                    break
        
        return AnalysisResponse(
            scan_id=scan_id,
            responses=responses,
            report_html=report_html
        )
    
    except ValueError as e:
        logger.error(f"Value error analyzing scan {scan_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing scan: {str(e)}")