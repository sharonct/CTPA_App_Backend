import torch
from utils.logging import logger
from utils.file_utils import load_scan_data, load_metadata
from models.vqa_model import get_vqa_model, get_feature_extractor
from models.ct_clip_loader import get_tokenizer
from config import DEVICE

def extract_features(scan_id):
    """
    Extract visual features from a scan
    
    Args:
        scan_id: ID of the scan
        
    Returns:
        Tensor of visual features
    """
    # Load metadata and validate
    metadata = load_metadata(scan_id)
    if not metadata:
        raise ValueError(f"Scan with ID {scan_id} not found")
    
    if metadata["status"] != "ready":
        raise ValueError(f"Scan is not ready for analysis. Current status: {metadata['status']}")
    
    # Load scan data
    file_path = metadata["filepath"]
    scan_data = load_scan_data(file_path)
    
    # Convert to tensor
    image_tensor = torch.tensor(scan_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Extract features
    feature_extractor = get_feature_extractor()
    visual_features = feature_extractor(image_tensor)
    
    return visual_features

def generate_answer(visual_features, question, max_length=150, temperature=0.7):
    """
    Generate an answer for a question using the VQA model
    
    Args:
        visual_features: Visual features extracted from the scan
        question: The question to answer
        max_length: Maximum length of the answer
        temperature: Temperature for sampling
        
    Returns:
        Generated answer as a string
    """
    try:
        # Get model and tokenizer
        vqa_model = get_vqa_model()
        tokenizer = get_tokenizer()
        
        # Format question as prompt
        prompt = f"Question: {question} Answer:"
        
        # Tokenize the prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(DEVICE)
        
        # Initialize with input tokens
        curr_ids = inputs.input_ids
        curr_mask = inputs.attention_mask
        
        # Auto-regressive generation
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = vqa_model(visual_features, curr_ids, curr_mask)
                
                # Get next token logits (last position)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # Check if all sequences are done
                if (next_tokens == tokenizer.sep_token_id).all():
                    break
                    
                # Add new tokens to the sequence
                curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(-1)], dim=-1)
                curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=DEVICE)], dim=-1)
            
            # Decode the generated tokens
            answer = tokenizer.decode(curr_ids[0], skip_special_tokens=True)
            
            # Extract just the answer part
            answer = answer.split("Answer:", 1)[-1].strip()
            
        return answer
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error generating answer: {str(e)}"

def analyze_scan(scan_id, questions):
    """
    Analyze a scan with multiple questions
    
    Args:
        scan_id: ID of the scan
        questions: List of questions to answer
        
    Returns:
        List of question-answer pairs
    """
    # Extract features
    visual_features = extract_features(scan_id)
    
    # Process each question
    responses = []
    for question in questions:
        answer = generate_answer(visual_features, question)
        responses.append({"question": question, "answer": answer})
    
    return responses