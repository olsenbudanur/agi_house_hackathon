from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Initialize the model once as a global variable
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_agent_embedding(agent_data: Dict[str, Any]) -> List[float]:
    """
    Compute embedding for an agent based on its metadata.
    Combines name, description, and capabilities into a single text for embedding.
    """
    try:
        # Combine relevant fields into a single text
        text_content = (
            f"Name: {agent_data['agent_name']}. "
            f"Description: {agent_data['description']}. "
            f"Capabilities: {', '.join(agent_data['capabilities'])}."
        )
        
        logger.info(f"Computing embedding for text: {text_content}")
        
        # Generate embedding
        embedding = model.encode(text_content)
        
        # Convert to list for JSON serialization
        result = embedding.tolist()
        logger.info(f"Generated embedding of length {len(result)}")
        
        return result
    except Exception as e:
        logger.error(f"Failed to compute embedding: {str(e)}")
        raise  # Re-raise the exception to handle it in the registration endpoint
