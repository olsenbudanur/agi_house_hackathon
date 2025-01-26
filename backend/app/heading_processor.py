"""
Module for processing hierarchical headings in matrix data.
"""
import re
from typing import Optional, Tuple

def parse_heading(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse a text string into heading and subheading components.
    
    Args:
        text: String that may contain heading ^ subheading notation
        
    Returns:
        Tuple of (heading, subheading) where subheading may be None
    """
    if '^' in text:
        parts = [p.strip() for p in text.split('^', 1)]
        return parts[0], parts[1] if len(parts) > 1 else None
    return text.strip(), None

def combine_headings(heading: str, subheading: Optional[str] = None) -> str:
    """
    Combine heading and subheading into standard notation.
    
    Args:
        heading: Main heading text
        subheading: Optional subheading text
        
    Returns:
        Combined heading string using ^ notation if subheading exists
    """
    if subheading:
        return f"{heading.strip()} ^ {subheading.strip()}"
    return heading.strip()

def detect_heading_pattern(text: str) -> bool:
    """
    Detect if text follows heading pattern rules.
    
    Args:
        text: Text to check for heading pattern
        
    Returns:
        True if text matches heading pattern
    """
    # Common heading patterns in insurance matrices
    patterns = [
        r'^(Primary|Second|Investment)\s+(Residence|Home|Property)',
        r'^(Purchase|Rate.*Term|Cash.*Out)',
        r'^(Program|Guidelines?|Requirements?|Restrictions?)',
        r'^(Credit|Income|Property|Additional)',
        r'^(LTV|FICO|DTI|Reserves?)',
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

def extract_heading_components(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extract heading and subheading from text if it matches heading patterns.
    
    Args:
        text: Text to analyze for heading components
        
    Returns:
        Tuple of (heading, subheading) if text contains heading pattern, None otherwise
    """
    if not detect_heading_pattern(text):
        return None
        
    return parse_heading(text)
