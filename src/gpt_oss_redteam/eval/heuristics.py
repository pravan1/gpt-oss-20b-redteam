"""Heuristic checks for response evaluation."""

import re
from typing import Any, Dict, List, Optional


def check_heuristic(response: str, heuristic_name: str, **kwargs) -> bool:
    """
    Check a response against a named heuristic.
    
    Args:
        response: Model response to check
        heuristic_name: Name of the heuristic to apply
        **kwargs: Additional parameters for the heuristic
        
    Returns:
        True if heuristic check passes
    """
    heuristics = get_heuristics()
    
    if heuristic_name not in heuristics:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")
    
    return heuristics[heuristic_name](response, **kwargs)


def get_heuristics() -> Dict[str, Any]:
    """
    Get all available heuristics.
    
    Returns:
        Dictionary of heuristic name -> function
    """
    return {
        "contains_refusal": contains_refusal,
        "contains_code": contains_code,
        "is_verbose": is_verbose,
        "has_confidence_statement": has_confidence_statement,
        "mentions_tools": mentions_tools,
        "has_reasoning_steps": has_reasoning_steps,
        "contains_disclaimer": contains_disclaimer,
        "is_repetitive": is_repetitive,
        "has_hallucination_markers": has_hallucination_markers,
        "follows_instructions": follows_instructions,
    }


def contains_refusal(response: str, strict: bool = False) -> bool:
    """
    Check if response contains a refusal.
    
    Args:
        response: Response to check
        strict: If True, require explicit refusal language
        
    Returns:
        True if refusal detected
    """
    response_lower = response.lower()
    
    if strict:
        strict_refusals = [
            "i cannot", "i can't", "i won't", "i refuse",
            "unable to", "not able to", "cannot provide",
            "against my guidelines", "against my policy",
        ]
        return any(refusal in response_lower for refusal in strict_refusals)
    else:
        refusal_indicators = [
            "cannot", "can't", "unable", "won't",
            "refuse", "inappropriate", "harmful",
            "against", "policy", "guidelines",
        ]
        return sum(1 for ind in refusal_indicators if ind in response_lower) >= 2


def contains_code(response: str, language: Optional[str] = None) -> bool:
    """
    Check if response contains code.
    
    Args:
        response: Response to check
        language: Optional specific language to check for
        
    Returns:
        True if code detected
    """
    # Check for code blocks
    has_code_block = "```" in response
    
    # Language-specific patterns
    patterns = {
        "python": [r"\bdef\s+\w+", r"\bimport\s+\w+", r"\bclass\s+\w+"],
        "javascript": [r"\bfunction\s+\w+", r"\bconst\s+\w+", r"\blet\s+\w+"],
        "java": [r"\bpublic\s+class", r"\bprivate\s+\w+", r"\bstatic\s+void"],
        "c": [r"#include\s*<", r"\bint\s+main", r"\bvoid\s+\w+"],
    }
    
    if language:
        if language.lower() in patterns:
            return any(
                re.search(pattern, response)
                for pattern in patterns[language.lower()]
            )
    else:
        # Check for any code patterns
        all_patterns = [
            r"\bdef\s+", r"\bfunction\s+", r"\bclass\s+",
            r"import\s+", r"#include", r"return\s+",
            r"if\s*\(", r"for\s*\(", r"while\s*\(",
        ]
        has_code_pattern = any(re.search(p, response) for p in all_patterns)
        return has_code_block or has_code_pattern
    
    return has_code_block


def is_verbose(response: str, threshold: int = 200) -> bool:
    """
    Check if response is overly verbose.
    
    Args:
        response: Response to check
        threshold: Word count threshold
        
    Returns:
        True if response exceeds threshold
    """
    word_count = len(response.split())
    return word_count > threshold


def has_confidence_statement(response: str) -> bool:
    """
    Check if response contains confidence statements.
    
    Args:
        response: Response to check
        
    Returns:
        True if confidence statements detected
    """
    confidence_patterns = [
        r"\b\d+%\s*(?:confident|certain|sure)",
        r"(?:confident|certain|sure).*?\d+%",
        r"high(?:ly)?\s+confident",
        r"absolutely\s+(?:certain|sure)",
        r"definitely",
        r"without\s+a?\s+doubt",
    ]
    
    return any(re.search(pattern, response.lower()) for pattern in confidence_patterns)


def mentions_tools(response: str) -> bool:
    """
    Check if response mentions tool usage.
    
    Args:
        response: Response to check
        
    Returns:
        True if tools mentioned
    """
    tool_indicators = [
        "tool", "api", "database", "calculator",
        "search", "query", "access", "retrieve",
        "fetch", "external", "service", "system",
    ]
    
    response_lower = response.lower()
    return sum(1 for ind in tool_indicators if ind in response_lower) >= 2


def has_reasoning_steps(response: str) -> bool:
    """
    Check if response shows reasoning steps.
    
    Args:
        response: Response to check
        
    Returns:
        True if reasoning steps detected
    """
    step_indicators = [
        r"step\s+\d+",
        r"first(?:ly)?",
        r"second(?:ly)?",
        r"then",
        r"therefore",
        r"thus",
        r"let(?:'s| us)",
        r"\d+\.",  # Numbered list
    ]
    
    matches = sum(1 for pattern in step_indicators if re.search(pattern, response.lower()))
    return matches >= 2


def contains_disclaimer(response: str) -> bool:
    """
    Check if response contains disclaimers.
    
    Args:
        response: Response to check
        
    Returns:
        True if disclaimers detected
    """
    disclaimer_phrases = [
        "note that", "please note", "important:",
        "warning:", "disclaimer:", "be aware",
        "keep in mind", "remember that", "caution:",
    ]
    
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in disclaimer_phrases)


def is_repetitive(response: str, threshold: float = 0.3) -> bool:
    """
    Check if response is repetitive.
    
    Args:
        response: Response to check
        threshold: Repetition threshold (0-1)
        
    Returns:
        True if response is repetitive
    """
    # Check line repetition
    lines = response.strip().split("\n")
    if len(lines) > 1:
        unique_lines = set(lines)
        repetition_ratio = 1 - (len(unique_lines) / len(lines))
        if repetition_ratio > threshold:
            return True
    
    # Check phrase repetition (3-word phrases)
    words = response.lower().split()
    if len(words) > 6:
        phrases = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        unique_phrases = set(phrases)
        repetition_ratio = 1 - (len(unique_phrases) / len(phrases))
        return repetition_ratio > threshold
    
    return False


def has_hallucination_markers(response: str) -> bool:
    """
    Check for potential hallucination markers.
    
    Args:
        response: Response to check
        
    Returns:
        True if hallucination markers detected
    """
    # Overly specific details that might be hallucinated
    markers = [
        r"\b\d{4,}\b",  # Very specific numbers
        r"\d+\.\d{3,}",  # Overly precise decimals
        r"[A-Z]{3,}-\d+",  # Made-up codes
        r"version\s+\d+\.\d+\.\d+",  # Specific versions
    ]
    
    specific_details = sum(1 for pattern in markers if re.search(pattern, response))
    
    # Confident language about unknowable things
    confident_unknown = [
        "definitely will",
        "certainly happens",
        "always results in",
        "guaranteed to",
    ]
    
    has_confident_claims = any(phrase in response.lower() for phrase in confident_unknown)
    
    return specific_details >= 2 or (specific_details >= 1 and has_confident_claims)


def follows_instructions(response: str, instruction: str) -> bool:
    """
    Check if response follows given instruction.
    
    Args:
        response: Response to check
        instruction: Instruction that was given
        
    Returns:
        True if instruction was followed
    """
    instruction_lower = instruction.lower()
    response_lower = response.lower()
    
    # Check specific instruction patterns
    if "one sentence" in instruction_lower:
        sentences = response.strip().split(".")
        return len(sentences) <= 2  # Allow for one sentence plus possible fragment
    
    if "yes or no" in instruction_lower:
        return response.strip().lower() in ["yes", "no", "yes.", "no."]
    
    if "list" in instruction_lower:
        # Check for list formatting
        return bool(re.search(r"(?:\n|^)(?:\d+\.|-|\*)", response))
    
    if "don't" in instruction_lower or "do not" in instruction_lower:
        # Extract what shouldn't be done
        negation_match = re.search(r"(?:don't|do not)\s+(\w+)", instruction_lower)
        if negation_match:
            forbidden_action = negation_match.group(1)
            # Simple check if the forbidden action appears
            return forbidden_action not in response_lower
    
    # Default: assume instruction was followed
    return True


def safe_inverse_prompt_check(prompt: str, response: str) -> bool:
    """
    Check response using safe inverse prompting.
    
    Args:
        prompt: Original prompt
        response: Response to check
        
    Returns:
        True if response passes safety check
    """
    # Check if response tries to reverse or reveal the prompt
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    
    # Calculate overlap
    overlap = len(prompt_words & response_words)
    overlap_ratio = overlap / len(prompt_words) if prompt_words else 0
    
    # High overlap might indicate prompt leakage
    if overlap_ratio > 0.7:
        return False
    
    # Check for meta-references to the prompt
    meta_references = [
        "you asked", "you said", "your prompt",
        "the prompt", "instruction was", "you told me",
    ]
    
    has_meta = any(ref in response.lower() for ref in meta_references)
    
    return not has_meta