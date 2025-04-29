from datetime import datetime
from utils.logging import logger


def generate_html_report(key_findings, pe_detected, pe_probability):
    """
    Generate an HTML report from the provided report text and optional parameters.
    
    Args:
        key_findings (list): List of key findings
        pe_detected (bool): Whether pulmonary embolism was detected
        pe_probability (float): Probability of PE if available
        
    Returns:
        str: HTML formatted report
    """
    try:
        # Define CSS styling for the report
        css_style = """
        <style>
            .report-container {
                font-family: 'Arial', sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 25px;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                background-color: #fff;
            }
            
            .report-header {
                text-align: center;
                padding-bottom: 15px;
                border-bottom: 2px solid #2c3e50;
            }
            
            .report-title {
                color: #2c3e50;
                margin: 0;
                font-size: 22px;
                font-weight: 700;
            }
            
            .report-section {
                margin: 15px 0;
                padding-bottom: 10px;
            }
            
            .report-section h4 {
                color: #2c3e50;
                margin: 0 0 8px 0;
                font-size: 16px;
                font-weight: 600;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }
            
            .report-section p, .report-section li {
                margin: 5px 0;
                line-height: 1.5;
                color: #333;
                font-size: 14px;
            }
            
            .pe-finding {
                color: #c0392b;
                font-weight: 600;
            }
            
            .normal-finding {
                color: #27ae60;
                font-weight: 600;
            }
            
            .impression-section {
                background-color: #f9f9f9;
                padding: 10px 15px;
                border-left: 4px solid #2c3e50;
                margin: 15px 0;
            }
            
            .recommendation-section {
                background-color: #f9f9f9;
                padding: 10px 15px;
                border-left: 4px solid #3498db;
                margin: 15px 0;
            }
            
            .pe-probability {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                font-weight: 600;
                margin-top: 10px;
            }
            
            .high-probability {
                background-color: #f9ebea;
                color: #c0392b;
                border: 1px solid #e74c3c;
            }
            
            .medium-probability {
                background-color: #fef9e7;
                color: #d35400;
                border: 1px solid #f39c12;
            }
            
            .low-probability {
                background-color: #eafaf1;
                color: #27ae60;
                border: 1px solid #2ecc71;
            }
        </style>
        """        

        # Format findings as HTML list items
        findings_html = ""
        for finding in key_findings:
            if finding.strip():
                # Determine if this is a PE-related finding
                finding_class = "pe-finding" if pe_detected and any(pe_term in finding.lower() for pe_term in ["embolism", "thrombus", "filling defect"]) else ""
                findings_html += f"<li><div class='{finding_class}'>{finding}</div></li>"

        # Format PE probability if available
        pe_probability_html = ""
        if pe_probability is not None:
            prob_class = ""
            if pe_probability >= 0.7:
                prob_class = "high-probability"
                prob_text = "High probability of pulmonary embolism"
            elif pe_probability >= 0.3:
                prob_class = "medium-probability"
                prob_text = "Moderate probability of pulmonary embolism"
            else:
                prob_class = "low-probability"
                prob_text = "Low probability of pulmonary embolism"
                
            pe_probability_html = f"""
            <div class="pe-probability {prob_class}">
                {prob_text} ({pe_probability:.2f})
            </div>
            """
        
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Construct the report HTML
        report_html = f"""
        {css_style}
        <div class="report-container">
            <div class="report-header">
                <h3 class="report-title">CT PULMONARY ANGIOGRAPHY REPORT</h3>
                {pe_probability_html}
            </div>
            
            <div class="report-section">
                <h4>PATIENT INFORMATION:</h4>
                <p>
                    <strong>Patient ID:</strong> Anonymous <br>
                    <strong>Name:</strong> Anonymous <br>
                    <strong>DOB:</strong> 01/01/1970 <br>
                    <strong>Gender:</strong> Unknown <br>
                    <strong>Referring Physician:</strong> Anonymous
                </p>
            </div>
            
            <div class="report-section">
                <h4>EXAM:</h4>
                <p>CT Pulmonary Angiography (CTPA)</p>
            </div>
            
            <div class="report-section">
                <h4>DATE:</h4>
                <p>{report_date}</p>
            </div>
            
            <div class="report-section">
                <h4>TECHNIQUE:</h4>
                <p>Contrast-enhanced CT of the chest with pulmonary arterial phase imaging.<br>
                IV contrast: 70 ml of non-ionic contrast material.<br>
                Slice thickness: 1.0 mm</p>
            </div>
            
            <div class="report-section">
                <h4>FINDINGS:</h4>
                <ul>
                    {findings_html}
                </ul>
            </div>
            
            <div class="report-section impression-section">
                <h4>IMPRESSION:</h4>
                <p>{"PULMONARY EMBOLISM DETECTED." if pe_detected else "No pulmonary embolism detected."}</p>
            </div>
            
            <div class="report-section recommendation-section">
                <h4>RECOMMENDATION:</h4>
                <p>{"Consider anticoagulation therapy and further cardiovascular assessment." if pe_detected else "No specific follow-up required for pulmonary embolism."}</p>
            </div>
            
            <div class="report-footer">
                <p>Report generated by AI Assistant<br>
                This report was electronically signed on {report_date}</p>
            </div>
        </div>
        """
        
        return report_html
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"""
        <div style="color: red; padding: 20px; border: 1px solid red; border-radius: 5px; margin: 20px;">
            <h3>Error Generating Report</h3>
            <p>{str(e)}</p>
        </div>
        """