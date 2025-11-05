"""
JSON extraction utility for robustly parsing JSON from LLM responses.
"""

import json
import re
from typing import TypeVar, Optional, Type
from .errors import JSONParseFailedError


T = TypeVar('T')


def extract_json(text: str) -> str:
    """
    Extract JSON from text using multiple strategies.
    
    Tries strategies in order:
    1. Direct validation
    2. JSON repair
    3. Markdown extraction
    4. Brace counting
    5. Aggressive cleanup + brace counting
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON string
        
    Raises:
        JSONParseFailedError: If JSON cannot be extracted
    """
    if not text:
        raise JSONParseFailedError("Empty text provided for JSON extraction")
    
    # Strategy 1: Direct validation
    try:
        text_trimmed = text.strip()
        json.loads(text_trimmed)
        return text_trimmed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: JSON repair
    try:
        repaired = _repair_json(text)
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 3: Markdown extraction
    try:
        extracted = _extract_from_markdown(text)
        if extracted:
            json.loads(extracted)
            return extracted
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 4: Brace counting
    try:
        extracted = _extract_by_brace_counting(text)
        if extracted:
            json.loads(extracted)
            return extracted
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 5: Aggressive cleanup + brace counting
    try:
        cleaned = _aggressive_cleanup(text)
        extracted = _extract_by_brace_counting(cleaned)
        if extracted:
            json.loads(extracted)
            return extracted
    except (json.JSONDecodeError, ValueError):
        pass
    
    raise JSONParseFailedError(f"Could not extract valid JSON from text: {text[:200]}...")


def _repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues.
    
    Fixes:
    - Unescaped newlines in string values
    - Unescaped carriage returns and tabs
    - Preserves escape sequences correctly
    """
    result = []
    i = 0
    in_string = False
    escape_next = False
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue
        
        if in_string:
            # Fix unescaped newlines, carriage returns, and tabs
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            else:
                result.append(char)
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def _extract_from_markdown(text: str) -> Optional[str]:
    """
    Extract JSON from markdown code blocks.
    
    Looks for:
    - ```json ... ``` blocks
    - ``` ... ``` blocks (fallback)
    """
    # Try json code blocks first
    json_pattern = r'```json\s*\n?(.*?)```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try generic code blocks
    code_pattern = r'```\s*\n?(.*?)```'
    match = re.search(code_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def _extract_by_brace_counting(text: str) -> Optional[str]:
    """
    Extract JSON by finding the first complete JSON object/array using brace counting.
    
    Tracks string state to avoid counting braces inside strings.
    """
    # Find first opening brace or bracket
    start_idx = -1
    for i, char in enumerate(text):
        if char in '{[':
            start_idx = i
            break
    
    if start_idx == -1:
        return None
    
    # Count braces/brackets to find complete object/array
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    i = start_idx
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            in_string = not in_string
            i += 1
            continue
        
        if in_string:
            i += 1
            continue
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
        
        # Check if we've found a complete object/array
        if brace_count == 0 and bracket_count == 0:
            return text[start_idx:i + 1]
        
        i += 1
    
    return None


def _aggressive_cleanup(text: str) -> str:
    """
    Aggressively clean up text by removing common prefixes and suffixes.
    """
    # Common prefixes
    prefixes = [
        r"^here'?s?\s+the\s+json\s*:?\s*",
        r"^json\s*:?\s*",
        r"^the\s+json\s+is\s*:?\s*",
        r"^response\s*:?\s*",
        r"^output\s*:?\s*",
    ]
    
    # Common suffixes
    suffixes = [
        r"\s*$",
        r"\s*\.\s*$",
    ]
    
    cleaned = text.strip()
    
    # Remove prefixes
    for prefix in prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
    
    # Remove suffixes (but keep the JSON structure)
    for suffix in suffixes:
        cleaned = re.sub(suffix, "", cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()


def parse_structured_output(text: str, output_type: Type[T]) -> T:
    """
    Parse JSON from text into a typed object.
    
    Args:
        text: Text containing JSON
        output_type: Type to parse into (must be a dataclass or dict)
        
    Returns:
        Parsed object of type T
        
    Raises:
        JSONParseFailedError: If JSON cannot be extracted or parsed
    """
    try:
        json_str = extract_json(text)
        data = json.loads(json_str)
        
        # If output_type is dict, return as-is
        if output_type == dict:
            return data
        
        # Try to instantiate the type
        if hasattr(output_type, '__annotations__'):
            # It's a dataclass or similar
            if isinstance(data, dict):
                return output_type(**data)
            else:
                return output_type(data)
        else:
            # Generic type, try direct conversion
            return output_type(data)
    
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise JSONParseFailedError(
            f"Failed to parse JSON into {output_type.__name__}: {str(e)}",
            cause=e
        ) from e

