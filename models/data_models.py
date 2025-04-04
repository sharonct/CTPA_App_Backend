from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class PredictionResult(BaseModel):
    task_id: str
    status: str
    pe_present: Optional[bool] = None
    pe_probability: Optional[float] = None
    pe_locations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None

class GPUStatus(BaseModel):
    cuda_available: bool
    device_name: Optional[str] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None
    total_memory_mb: Optional[float] = None
    memory_utilization: Optional[float] = None