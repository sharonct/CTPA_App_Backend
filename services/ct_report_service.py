import torch
from datetime import datetime
from utils.logging import logger
from utils.file_utils import load_scan_data, load_metadata
from models.ct_report_model import get_ct_report_model
from config import DEVICE
from services.report_service import generate_report as generate_html_report

def generate_ct_report(scan_id, prompt=None):
    """
    Generate a CT report using the trained model
    
    Args:
        scan_id: ID of the scan
        prompt: Optional custom prompt
        
    Returns:
        dict: Generated report in different formats
    """
    try:
        # Load metadata and validate
        metadata = load_metadata(scan_id)
        if not metadata:
            raise ValueError(f"Scan with ID {scan_id} not found")
        
        if metadata["status"] != "ready":
            raise ValueError(f"Scan is not ready for analysis. Current status: {metadata['status']}")
        
        # Get the model
        model = get_ct_report_model()
        if model is None:
            raise ValueError("CT Report Generator model not loaded")
        
        # Load scan data
        file_path = metadata["filepath"]
        scan_data = load_scan_data(file_path)
        
        # Prepare scan tensor
        if scan_data.ndim == 3:  # [D, H, W]
            scan_tensor = torch.tensor(scan_data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        elif scan_data.ndim == 4:  # [C, D, H, W]
            scan_tensor = torch.tensor(scan_data).unsqueeze(0)  # [1, C, D, H, W]
        else:
            raise ValueError(f"Unexpected scan dimensions: {scan_data.shape}")
        
        # Move tensor to device
        scan_tensor = scan_tensor.to(DEVICE, dtype=torch.float32)
        
        # Default prompt if not provided
        if prompt is None:
            prompt = "Generate a detailed clinical report for this CT pulmonary angiogram scan:"
        
        # Generate report
        logger.info(f"Generating CT report for scan {scan_id}")
        
        with torch.no_grad():
            report_text = model.generate_report(
                images=scan_tensor,
                prompt=prompt,
                max_length=512,
                temperature=0.7
            )
        
        # Generate HTML report
        # Use the feature extractor directly from the model
        visual_features = model.vision_feature_extractor(scan_tensor)
        report_html = generate_html_report(visual_features, scan_data)
        
        # Update metadata with report generation time
        metadata["report_generated_time"] = datetime.now().isoformat()
        from utils.file_utils import save_metadata
        save_metadata(scan_id, metadata)
        
        return {
            "scan_id": scan_id,
            "report_text": report_text,
            "report_html": report_html,
            "generated_time": metadata["report_generated_time"]
        }
        
    except Exception as e:
        logger.error(f"Error generating CT report for scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def extract_key_findings(report_text):
    """
    Extract key findings from the report text
    
    Args:
        report_text: The report text
        
    Returns:
        list: List of key findings
    """
    # Split report by lines and filter out empty lines
    lines = [line.strip() for line in report_text.split('\n') if line.strip()]
    
    findings = []
    in_findings_section = False
    
    # Look for findings section
    for line in lines:
        lower_line = line.lower()
        
        # Check for findings section headers
        if "findings:" in lower_line or "finding:" in lower_line:
            in_findings_section = True
            continue
        
        # Check for end of findings section
        if in_findings_section and ("impression:" in lower_line or "assessment:" in lower_line or "conclusion:" in lower_line):
            in_findings_section = False
            continue
        
        # Collect findings
        if in_findings_section:
            # Remove bullet points and leading/trailing whitespace
            finding = line.strip()
            if finding.startswith('-') or finding.startswith('•'):
                finding = finding[1:].strip()
            
            if finding:
                findings.append(finding)
    
    # If no findings section was identified, use first 3-5 lines as findings
    if not findings and len(lines) >= 3:
        findings = lines[:min(5, len(lines))]
    
    return findings

def has_pulmonary_embolism(report_text):
    """
    Detect if the report indicates pulmonary embolism
    
    Args:
        report_text: The report text
        
    Returns:
        bool: True if PE is detected, False otherwise
    """
    pe_keywords = [
        "pulmonary embolism", "pulmonary emboli", 
        "filling defect", "filling defects",
        "thrombus", "thrombi", "thromboembolism"
    ]
    
    # Check if any PE keywords are present and not negated
    negation_terms = ["no ", "not ", "without ", "absence of ", "negative for "]
    
    lower_text = report_text.lower()
    
    for keyword in pe_keywords:
        if keyword in lower_text:
            # Check if it's negated
            keyword_idx = lower_text.find(keyword)
            
            # Check 30 characters before the keyword for negation terms
            context_before = lower_text[max(0, keyword_idx - 30):keyword_idx]
            
            if not any(neg in context_before for neg in negation_terms):
                return True
    
    return False

def calculate_pe_probability(report_text):
    """
    Calculate probability of PE based on report text
    
    Args:
        report_text: The report text
        
    Returns:
        float: Probability of PE (0.0 to 1.0)
    """
    # This is a simple heuristic - in a real system you'd want a more sophisticated approach
    lower_text = report_text.lower()
    
    # Start with a baseline probability
    probability = 0.1  # 10% baseline
    
    # Check for positive indicators
    if "pulmonary embolism" in lower_text:
        if "no pulmonary embolism" in lower_text or "without pulmonary embolism" in lower_text:
            probability = 0.05
        else:
            probability = 0.95
    
    if "filling defect" in lower_text and not "no filling defect" in lower_text:
        probability = max(probability, 0.9)
    
    if "thrombus" in lower_text and not "no thrombus" in lower_text:
        probability = max(probability, 0.85)
    
    # Check for location - more specific locations increase probability
    if any(loc in lower_text for loc in ["right lower lobe", "left lower lobe", "segmental", "subsegmental"]):
        probability = min(1.0, probability + 0.1)
    
    # Check for uncertainty language
    if any(term in lower_text for term in ["possibly", "probable", "suspected", "cannot exclude"]):
        probability = probability * 0.8
    
    # Check for other factors
    if "right heart strain" in lower_text:
        probability = min(1.0, probability + 0.05)
    
    return probability