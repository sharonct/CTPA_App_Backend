from datetime import datetime
from utils.logging import logger
from services.vqa_service import generate_answer

def generate_report(visual_features, scan_data=None):
    """
    Generate an HTML report from visual features
    
    Args:
        visual_features: Visual features from the scan
        scan_data: Optional raw scan data
        
    Returns:
        HTML string for the report
    """
    try:
        # Generate title from model
        title_q = "What is the diagnosis for this CT scan?"
        diagnosis = generate_answer(visual_features, title_q)
        
        # Generate findings
        findings_q = "List the key findings in this CT scan"
        findings = generate_answer(visual_features, findings_q)
        
        # Generate impression
        impression_q = "What is your impression of this CT scan?"
        impression = generate_answer(visual_features, impression_q)
        
        # Generate recommendations
        rec_q = "What recommendations do you have based on this CT scan?"
        recommendations = generate_answer(visual_features, rec_q)
        
        # Detect if PE is present based on the diagnosis
        pe_present = "pulmonary embolism" in diagnosis.lower()
        
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
        </style>
        """
        
        # Format findings as list items
        findings_list = ""
        for line in findings.split("\n"):
            if line.strip():
                findings_class = "pe-finding" if pe_present and "embolism" in line.lower() else ""
                findings_list += f"<li><div class='{findings_class}'>{line}</div></li>"
        
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Construct the report HTML
        report_html = f"""
        {css_style}
        <div class="report-container">
            <div class="report-header">
                <h3 class="report-title">CT PULMONARY ANGIOGRAPHY REPORT</h3>
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
                    {findings_list}
                </ul>
            </div>
            
            <div class="report-section impression-section">
                <h4>IMPRESSION:</h4>
                <p>{impression}</p>
            </div>
            
            <div class="report-section recommendation-section">
                <h4>RECOMMENDATION:</h4>
                <p>{recommendations}</p>
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