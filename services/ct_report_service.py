import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import logging
import nibabel as nib
from datetime import datetime

# Setup logging
from utils.logging import logger
from utils.file_utils import load_scan_data, load_metadata, save_metadata, preprocess_nifti
from models.ct_report_model import get_ct_report_model, load_ct_report_model
from config import DEVICE
from services.report_service import generate_html_report


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
        
        # Load scan data - with support for NIFTI files
        file_path = metadata["filepath"]
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            logger.info(f"Processing NIFTI file: {file_path}")
            scan_data = preprocess_nifti(file_path, metadata)
        else:
            logger.info(f"Loading scan data from: {file_path}")
            scan_data = load_scan_data(file_path)
        
        # # Prepare scan tensor
        # if scan_data.ndim == 3:  # [D, H, W]
        #     scan_tensor = torch.tensor(scan_data).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        # elif scan_data.ndim == 4:  # [C, D, H, W]
        #     scan_tensor = torch.tensor(scan_data).unsqueeze(0)  # [1, C, D, H, W]
        # else:
        #     raise ValueError(f"Unexpected scan dimensions: {scan_data.shape}")
        
        # Move tensor to device
        scan_tensor = scan_data.to(DEVICE, dtype=torch.float32)
        
        # Default prompt if not provided
        if prompt is None:
            prompt = "As a Radiology expert, you are a board-certified thoracic radiologist analyzing this CTPA scan for potential " \
            "pulmonary embolism and other thoracic pathologies. Your goal is to generate a comprehensive, accurate radiological report " \
            "that clearly identifies any pulmonary emboli and associated findings. Methodically evaluate all pulmonary arteries from " \
            "main to subsegmental branches, describe any filling defects, and note relevant pulmonary, cardiac, and pleural findings " \
            "using precise radiological terminology. The report should follow standard radiological formatting with clear findings and " \
            "impression sections suitable for clinical decision-making by referring physicians."

        
        # Generate report
        logger.info(f"Generating CT report for scan {scan_id}")
        
        with torch.no_grad():
            report_text = model.generate_report(
                images=scan_tensor,
                prompt=prompt,
                max_length=512,
                temperature=0.7
            )
    
        
        # Process report to extract findings and PE probability
        key_findings = extract_key_findings(report_text)
        pe_detected = has_pulmonary_embolism(report_text)
        pe_probability = calculate_pe_probability(report_text)

        # Generate HTML report
        report_html = generate_html_report(key_findings, pe_detected, pe_probability)
        
        # Update metadata with report generation time
        metadata["report_generated_time"] = datetime.now().isoformat()
        save_metadata(scan_id, metadata)
        
        return {
            "scan_id": scan_id,                                           
            "report_text": report_text,
            "report_html": report_html,
            "generated_time": metadata["report_generated_time"],
            "key_findings": key_findings,
            "pe_detected": pe_detected,
            "pe_probability": pe_probability
        }
        
    except Exception as e:
        logger.error(f"Error generating CT report for scan {scan_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def extract_key_findings(report_text):
    """
    Extract key findings from the report text, clean repetitive phrases,
    and format findings in a clean list format.
    
    Args:
        report_text: The report text
        
    Returns:
        list: List of key findings as clean statements
    """
    import re
    
    # Split report by lines and join to process as a single text
    lines = [line.strip() for line in report_text.split('\n') if line.strip()]
    full_text = ' '.join(lines)
    
    # Common patterns to remove
    patterns_to_remove = [
        r'there are no substantial differences between the preliminary results and the impressions in this final report',
        r'case discussed with(?:\s+\w+)*(?:\s+at)?(?:\s+on)?(?:\s+\d+\s*(?:am|pm|AM|PM))?',
        r'\b\d+\s*(?:am|pm|AM|PM)\b'
    ]
    
    # Remove patterns
    for pattern in patterns_to_remove:
        full_text = re.sub(pattern, '', full_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and periods
    full_text = re.sub(r'\s+', ' ', full_text)      # Multiple spaces to single space
    full_text = re.sub(r'\.+', '.', full_text)      # Multiple periods to single period
    full_text = re.sub(r'\s+\.', '.', full_text)    # Space before period
    full_text = re.sub(r'\.\s+', '. ', full_text)   # Normalize space after period
    
    # Split by periods to get individual findings
    raw_findings = [f.strip() for f in full_text.split('.') if f.strip()]
    
    # Filter out irrelevant content
    filtered_findings = []
    for finding in raw_findings:
        # Skip very short findings or anything with case discussion that wasn't caught earlier
        if len(finding) < 5 or re.search(r'case discussed|discussed with', finding, re.IGNORECASE):
            continue
        filtered_findings.append(finding)
    
    # Deduplicate findings - remove any duplicate statements
    unique_findings = []
    for finding in filtered_findings:
        normalized = finding.lower()
        if normalized not in [f.lower() for f in unique_findings]:
            unique_findings.append(finding)
    
    # Format findings with proper capitalization and ending period
    formatted_findings = []
    for finding in unique_findings:
        # Capitalize first letter
        formatted = finding[0].upper() + finding[1:] if finding else ""
        # Add period if missing
        if not formatted.endswith('.'):
            formatted += "."
        formatted_findings.append(formatted)
    
    # Prioritize findings with medical terms if we have too many
    if len(formatted_findings) > 5:
        medical_terms = ["pulmonary", "embolism", "aortic", "nodule", "mass", 
                        "effusion", "pneumonia", "consolidation", "opacity", 
                        "infiltrate", "metastatic", "thrombus", "dissection"]
        
        # Find findings with medical terms
        medical_findings = [f for f in formatted_findings 
                           if any(term in f.lower() for term in medical_terms)]
        
        # If we have medical findings, use those, otherwise use first 5
        if medical_findings:
            return medical_findings[:5]
        else:
            return formatted_findings[:5]
    
    return formatted_findings


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