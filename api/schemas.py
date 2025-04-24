from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Question(BaseModel):
    """Question model"""
    text: str

class QuestionResponse(BaseModel):
    """Response to a question"""
    question: str
    answer: str

class AnalysisRequest(BaseModel):
    """Request for scan analysis"""
    scan_id: str
    questions: List[Question]

class AnalysisResponse(BaseModel):
    """Response for scan analysis"""
    scan_id: str
    responses: List[QuestionResponse]
    report_html: Optional[str] = None

class GenerateReportRequest(BaseModel):
    """Request to generate a report"""
    prompt: Optional[str] = None

class ReportResponse(BaseModel):
    """Response for report generation"""
    scan_id: str
    report_text: str
    report_html: str
    pe_detected: bool
    pe_probability: float
    generated_time: str

class ScanMetadata(BaseModel):
    """Metadata for a scan"""
    scan_id: str
    filename: str
    upload_time: str
    status: str
    filepath: Optional[str] = None
    error_message: Optional[str] = None
    pe_probability: Optional[float] = None