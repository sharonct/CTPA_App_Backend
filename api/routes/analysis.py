from fastapi import APIRouter, HTTPException
from api.schemas import AnalysisRequest, AnalysisResponse, QuestionResponse
from utils.logging import logger
from models.ct_clip_loader import load_ctclip_model
from models.ct_report_model import load_ct_report_model

# Import the NIFTI processing function
from services.ct_report_service import generate_ct_report

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_scan_endpoint(request: AnalysisRequest):
    """Analyze a scan with questions"""
    scan_id = request.scan_id
    questions = [q.text for q in request.questions]
    
    logger.info(f"Received analysis request for scan {scan_id} with {len(questions)} questions")
    
    # Check if we should use fallback mode
    use_fallback = False
    
    # Try to ensure models are loaded
    try:
        ctclip_loaded = load_ctclip_model()
        report_loaded = load_ct_report_model()
        
        if not ctclip_loaded or not report_loaded:
            logger.warning("Models not loaded properly, using fallback mode")
            use_fallback = True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        use_fallback = True
    
    # Process in fallback mode if needed
    if use_fallback:
        logger.info("Using fallback mode for analysis")
        from services.report_service import generate_fallback_report
        
        # Generate dummy responses
        responses = []
        for question in questions:
            if "generate report" in question.lower() or "create report" in question.lower():
                answer = "Report has been generated using fallback mode."
            else:
                answer = "Unable to process this question with the AI model. Using fallback mode."
            responses.append(QuestionResponse(question=question, answer=answer))
        
        # Use fallback report if report was requested
        report_html = None
        for q in questions:
            if "generate report" in q.lower() or "create report" in q.lower():
                report_html = generate_fallback_report()
                break
        
        return AnalysisResponse(
            scan_id=scan_id,
            responses=responses,
            report_html=report_html
        )
    
    # Normal processing with models
    try:
        
        # Generate a report if requested
        report_html = None
        report_text = None
        pe_detected = None
        pe_probability = None
        
        logger.info("Generating report")
        
        # Use the enhanced report generator for both HTML and text reports
        try:            
            # Generate the comprehensive report
            report_result = generate_ct_report(scan_id)
            
            # Extract the HTML and text components
            report_html = report_result.get("report_html")
            report_text = report_result.get("report_text")
            pe_detected = report_result.get("pe_detected")
            pe_probability = report_result.get("pe_probability")
                    
        except Exception as report_error:
            logger.error(f"Error with enhanced report generation: {report_error}")
    
        responses = []
        
        # Prepare the response
        response = AnalysisResponse(
            scan_id=scan_id,
            responses=responses,
            report_html=report_html
        )
        
        # Add the additional report information if available
        if report_text:
            response.report_text = report_text
        
        if pe_detected is not None:
            response.pe_detected = pe_detected
            
        if pe_probability is not None:
            response.pe_probability = pe_probability
        
        logger.info(f"Analysis completed for scan {scan_id}")
        return response
    
    except ValueError as e:
        logger.error(f"Value error analyzing scan {scan_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to generate a fallback report
        try:
            from services.report_service import generate_fallback_report
            report_html = generate_fallback_report()
            
            # Create basic responses
            responses = []
            for question in questions:
                responses.append(QuestionResponse(
                    question=question, 
                    answer="An error occurred while processing this question."
                ))
            
            return AnalysisResponse(
                scan_id=scan_id,
                responses=responses,
                report_html=report_html
            )
        except:
            # If even the fallback fails, raise the original error
            raise HTTPException(status_code=500, detail=f"Error analyzing scan: {str(e)}")