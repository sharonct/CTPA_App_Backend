from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Question(BaseModel):
    text: str

class AnalysisRequest(BaseModel):
    scan_id: str
    questions: List[Question]

class QuestionResponse(BaseModel):
    question: str
    answer: str

class AnalysisResponse(BaseModel):
    scan_id: str
    responses: List[QuestionResponse]
    report_html: Optional[str] = None

class ScanMetadata(BaseModel):
    scan_id: str
    filename: str
    upload_time: str
    status: str
    dimensions: Optional[List[int]] = None
    pe_probability: Optional[float] = None

class SliceResponse(BaseModel):
    scan_id: str
    view: str
    slice_idx: int
    window_center: int
    window_width: int
    dimensions: List[int]
    image: str