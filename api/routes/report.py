from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
from services.ct_report_service import generate_ct_report, has_pulmonary_embolism, calculate_pe_probability
from utils.logging import logger
from utils.file_utils import update_scan_status

router = APIRouter()

class GenerateReportRequest(BaseModel):
    prompt: Optional[str] = None

class ReportResponse(BaseModel):
    scan_id: str
    report_text: str
    report_html: str
    pe_detected: bool
    pe_probability: float
    generated_time: str

@router.post("/report/{scan_id}", response_model=ReportResponse)
async def generate_report_endpoint(scan_id: str, request: GenerateReportRequest = None):
    """
    Generate a CT report for the specified scan
    """
    try:
        # Update scan status
        update_scan_status(scan_id, "generating_report")
        
        # Use custom prompt if provided
        prompt = None
        if request and request.prompt:
            prompt = request.prompt
        
        # Generate report
        report_data = generate_ct_report(scan_id, prompt)
        
        # Analyze for PE
        pe_detected = has_pulmonary_embolism(report_data["report_text"])
        pe_probability = calculate_pe_probability(report_data["report_text"])
        
        # Update scan metadata with PE probability
        from utils.file_utils import load_metadata, save_metadata
        metadata = load_metadata(scan_id)
        if metadata:
            metadata["pe_probability"] = pe_probability
            save_metadata(scan_id, metadata)
        
        # Update scan status
        update_scan_status(scan_id, "report_generated")
        
        return ReportResponse(
            scan_id=scan_id,
            report_text=report_data["report_text"],
            report_html=report_data["report_html"],
            pe_detected=pe_detected,
            pe_probability=pe_probability,
            generated_time=report_data["generated_time"]
        )
        
    except ValueError as e:
        update_scan_status(scan_id, "error", error_message=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating report for scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        update_scan_status(scan_id, "error", error_message=str(e))
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")