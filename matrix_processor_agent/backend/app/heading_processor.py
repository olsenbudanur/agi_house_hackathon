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

def detect_heading_pattern(text: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if text follows heading pattern rules and identify the category.
    
    Args:
        text: Text to check for heading pattern
        
    Returns:
        Tuple of (is_heading, category) where category may be None
    """
    # Section header patterns that indicate requirement categories
    requirement_categories = {
        'max_dti_requirements': [
            r'^Requirements?\s*\^\s*(?:Max(?:imum)?\s*)?DTI(?:\s*Limits?)?',
            r'^(?:Max(?:imum)?\s*)?DTI\s*Requirements?',
            r'^DTI\s*Requirements?'
        ],
        'credit_requirements': [
            r'^Requirements?\s*\^\s*Credit',
            r'^Credit\s*Requirements?'
        ],
        'reserve_requirements': [
            r'^Requirements?\s*\^\s*Reserve',
            r'^Reserve\s*Requirements?'
        ],
        'geographic_restrictions': [
            r'^Requirements?\s*\^\s*Geographic',
            r'^Geographic\s*(?:Requirements?|Restrictions?)'
        ]
    }
    
    # Check for requirement section headers first
    for category, patterns in requirement_categories.items():
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return True, category
    
    # Other heading patterns that aren't requirement sections
    general_patterns = [
        r'^(?:Primary|Second|Investment)\s+(?:Residence|Home|Property)',
        r'^(?:Purchase|Rate.*Term|Cash.*Out)',
        r'^(?:Program|Guidelines?|Requirements?|Restrictions?)',
        r'^(?:Credit|Income|Property|Additional)',
        r'^(?:LTV|FICO|DTI|Reserves?)'
    ]
    
    # Check if it's a general heading
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in general_patterns):
        return True, None
        
    return False, None

def extract_heading_components(text: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extract heading and subheading from text if it matches heading patterns.
    
    Args:
        text: Text to analyze for heading components
        
    Returns:
        Tuple of (heading, subheading) if text contains heading pattern, None otherwise
    """
    is_heading, category = detect_heading_pattern(text)
    if not is_heading:
        return None
        
    return parse_heading(text)
