"""Prompt templating and formatting utilities."""

import logging
from typing import Any, Dict, Optional

from jinja2 import Environment, Template

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Jinja2-based prompt template."""
    
    def __init__(self, template: str, safe: bool = True):
        """
        Initialize prompt template.
        
        Args:
            template: Jinja2 template string
            safe: Whether to use safe template environment
        """
        self.template_str = template
        self.safe = safe
        
        # Create Jinja2 environment
        self.env = Environment(
            autoescape=safe,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = self.env.from_string(template)
    
    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **kwargs: Template variables
            
        Returns:
            Rendered prompt string
        """
        return self.template.render(**kwargs)
    
    def __call__(self, **kwargs) -> str:
        """Shorthand for render."""
        return self.render(**kwargs)


class SafetyWrapper:
    """Wrapper to add safety instructions to prompts."""
    
    DEFAULT_SAFETY_PROMPT = """
IMPORTANT SAFETY INSTRUCTIONS:
- Provide helpful, harmless, and honest responses
- Do not generate harmful, offensive, or inappropriate content
- Respect user privacy and do not request personal information
- Follow ethical guidelines and best practices

"""
    
    def __init__(
        self,
        safety_prompt: Optional[str] = None,
        prepend: bool = True,
    ):
        """
        Initialize safety wrapper.
        
        Args:
            safety_prompt: Custom safety instructions
            prepend: Whether to prepend (True) or append (False) safety prompt
        """
        self.safety_prompt = safety_prompt or self.DEFAULT_SAFETY_PROMPT
        self.prepend = prepend
    
    def wrap(self, prompt: str) -> str:
        """
        Wrap prompt with safety instructions.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Wrapped prompt with safety instructions
        """
        if self.prepend:
            return self.safety_prompt + prompt
        else:
            return prompt + "\n\n" + self.safety_prompt
    
    def __call__(self, prompt: str) -> str:
        """Shorthand for wrap."""
        return self.wrap(prompt)


def format_prompt(
    prompt: str,
    system: Optional[str] = None,
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
    include_assistant_prefix: bool = True,
) -> str:
    """
    Format prompt with system message and conversation prefixes.
    
    Args:
        prompt: Main prompt text
        system: Optional system message
        user_prefix: Prefix for user messages
        assistant_prefix: Prefix for assistant messages
        include_assistant_prefix: Whether to include assistant prefix
        
    Returns:
        Formatted prompt string
    """
    formatted = ""
    
    if system:
        formatted += f"System: {system}\n\n"
    
    formatted += f"{user_prefix}{prompt}"
    
    if include_assistant_prefix:
        formatted += f"\n\n{assistant_prefix}"
    
    return formatted


def create_few_shot_prompt(
    task: str,
    examples: list[tuple[str, str]],
    query: str,
    system: Optional[str] = None,
) -> str:
    """
    Create a few-shot prompt with examples.
    
    Args:
        task: Task description
        examples: List of (input, output) example pairs
        query: The actual query to answer
        system: Optional system message
        
    Returns:
        Formatted few-shot prompt
    """
    prompt_parts = []
    
    if system:
        prompt_parts.append(f"System: {system}")
    
    prompt_parts.append(f"Task: {task}")
    prompt_parts.append("\nExamples:")
    
    for i, (input_ex, output_ex) in enumerate(examples, 1):
        prompt_parts.append(f"\nExample {i}:")
        prompt_parts.append(f"Input: {input_ex}")
        prompt_parts.append(f"Output: {output_ex}")
    
    prompt_parts.append(f"\nNow, complete the following:")
    prompt_parts.append(f"Input: {query}")
    prompt_parts.append("Output:")
    
    return "\n".join(prompt_parts)


# Pre-defined prompt templates
TEMPLATES = {
    "basic": PromptTemplate("{{ prompt }}"),
    
    "instruction": PromptTemplate("""
Instruction: {{ instruction }}
{% if context %}
Context: {{ context }}
{% endif %}
{% if constraints %}
Constraints: {{ constraints }}
{% endif %}

Response:"""),
    
    "qa": PromptTemplate("""
Question: {{ question }}
{% if context %}
Context: {{ context }}
{% endif %}

Answer:"""),
    
    "reasoning": PromptTemplate("""
Problem: {{ problem }}

Let's think step by step:
1."""),
    
    "evaluation": PromptTemplate("""
You are being evaluated on {{ task }}.
{% if criteria %}
Evaluation criteria: {{ criteria }}
{% endif %}

{{ prompt }}

Response:"""),
    
    "audit": PromptTemplate("""
AUDIT MODE: This interaction is being monitored for compliance.

{{ prompt }}

Compliant Response:"""),
    
    "casual": PromptTemplate("""
Hey! {{ prompt }}

Response:"""),
}


def get_template(name: str) -> PromptTemplate:
    """
    Get a pre-defined prompt template by name.
    
    Args:
        name: Template name
        
    Returns:
        PromptTemplate instance
        
    Raises:
        KeyError: If template name not found
    """
    if name not in TEMPLATES:
        raise KeyError(f"Template '{name}' not found. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]