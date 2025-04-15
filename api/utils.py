import torch

def get_gpu_status():
    """Get GPU status information"""
    if torch.cuda.is_available():
        try:
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**2)  # MB
            memory_reserved = torch.cuda.memory_reserved(current_device) / (1024**2)    # MB
            max_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**2)  # MB
            
            return {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(current_device),
                "memory_allocated_mb": round(memory_allocated, 2),
                "memory_reserved_mb": round(memory_reserved, 2),
                "total_memory_mb": round(max_memory, 2),
                "memory_utilization": round(memory_allocated / max_memory * 100, 2)
            }
        except Exception as e:
            return {"cuda_available": True, "error": str(e)}
    else:
        return {"cuda_available": False}