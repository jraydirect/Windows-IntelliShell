"""Semantic intent extraction with hybrid Rule+AI dispatch."""

import os
import re
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity from user input."""
    type: str  # 'path', 'file', 'process', 'number', etc.
    value: str
    original: str
    start: int
    end: int


@dataclass
class IntentMatch:
    """Represents a matched intent with confidence score."""
    intent_name: str
    provider_name: str
    confidence: float
    trigger_pattern: str
    original_input: str
    entities: List[Entity] = None
    source: str = "rule_based"  # 'rule_based' or 'llm'
    parameters: Dict[str, Any] = None  # LLM-extracted parameters
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.parameters is None:
            self.parameters = {}


@dataclass
class AmbiguousMatch:
    """Represents multiple possible matches requiring disambiguation."""
    original_input: str
    suggestions: List[IntentMatch]


class EntityExtractor:
    """Extract entities from natural language input."""
    
    # Windows special paths
    SPECIAL_PATHS = {
        'desktop': lambda: os.path.join(os.path.expanduser('~'), 'Desktop'),
        'downloads': lambda: os.path.join(os.path.expanduser('~'), 'Downloads'),
        'documents': lambda: os.path.join(os.path.expanduser('~'), 'Documents'),
        'temp': lambda: os.environ.get('TEMP', ''),
        'appdata': lambda: os.environ.get('APPDATA', ''),
        'userprofile': lambda: os.environ.get('USERPROFILE', ''),
        'home': lambda: os.path.expanduser('~'),
    }
    
    def __init__(self):
        # Patterns for entity extraction
        self.patterns = [
            # Quoted paths/files
            (r'"([^"]+)"', 'path'),
            (r"'([^']+)'", 'path'),
            # Environment variables
            (r'%([A-Z_]+)%', 'envvar'),
            # File extensions
            (r'\b(\w+\.(txt|pdf|doc|docx|jpg|png|log|json|xml|py|exe))\b', 'file'),
            # Numbers with units
            (r'\b(\d+)\s*(gb|mb|kb|percent|%)', 'number'),
            # Plain numbers (for counts, PIDs, etc.) - but not if followed by unit
            (r'\b(\d+)\b(?!\s*(?:gb|mb|kb|percent|%))', 'number'),
        ]
    
    def expand_env_vars(self, text: str) -> str:
        """
        Expand environment variables in text.
        
        Supports: %TEMP%, %APPDATA%, %USERPROFILE%, etc.
        """
        # Windows style %VAR%
        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        expanded = re.sub(r'%([A-Z_]+)%', replace_var, text, flags=re.IGNORECASE)
        
        # Also expand $VAR style for compatibility
        expanded = os.path.expandvars(expanded)
        
        return expanded
    
    def extract(self, text: str) -> List[Entity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Check for special path references
        text_lower = text.lower()
        for path_name, path_func in self.SPECIAL_PATHS.items():
            if path_name in text_lower:
                try:
                    path_value = path_func()
                    if path_value:
                        entities.append(Entity(
                            type='special_path',
                            value=path_value,
                            original=path_name,
                            start=text_lower.find(path_name),
                            end=text_lower.find(path_name) + len(path_name)
                        ))
                except Exception as e:
                    logger.debug(f"Failed to resolve special path {path_name}: {e}")
        
        # Extract using regex patterns
        for pattern, entity_type in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    type=entity_type,
                    value=match.group(1),
                    original=match.group(0),
                    start=match.start(),
                    end=match.end()
                ))
        
        # Check for clipboard reference
        if 'clipboard' in text_lower or 'copied' in text_lower:
            from intellishell.utils.clipboard import get_clipboard_content
            clipboard_content = get_clipboard_content()
            if clipboard_content:
                entities.append(Entity(
                    type='clipboard',
                    value=clipboard_content,
                    original='clipboard',
                    start=text_lower.find('clipboard'),
                    end=text_lower.find('clipboard') + 9
                ))
        
        return entities


class SemanticParser:
    """
    Semantic Router with LLM-First natural language understanding.
    
    Features:
    - Semantic Router: LLM-first for natural language queries
    - Rule-based matching for high confidence (>0.9) exact commands
    - Entity extraction and env var expansion
    - Natural language detection: questions and conversational queries
    """
    
    CONFIDENCE_THRESHOLD = 0.90  # Very high confidence: execute directly (rule-based)
    MIN_CONFIDENCE = 0.50  # Minimum for rule-based
    AMBIGUITY_ZONE = (0.60, 0.90)  # Range requiring disambiguation
    
    # Natural language markers that trigger LLM-first routing
    NL_MARKERS = [
        "what", "show", "tell", "find", "list", "where", "which", "who",
        "how", "can you", "could you", "will you", "please",
        "are there", "is there", "do you", "does"
    ]
    
    def __init__(self, registry, ai_bridge=None, use_rust: bool = True):
        """
        Initialize parser with provider registry and optional AI bridge.
        
        Args:
            registry: ProviderRegistry instance
            ai_bridge: Optional AIBridge for LLM fallback
            use_rust: Use Rust parser backend if available (default: True)
        """
        self.registry = registry
        self.ai_bridge = ai_bridge
        self._trigger_cache: List[Tuple] = []
        self._entity_extractor = EntityExtractor()
        self._use_rust = use_rust
        self._rust_backend = None
        
        # Try to initialize Rust backend if requested
        if use_rust:
            try:
                from intellishell.parser_rust import RustParserBackend, RUST_AVAILABLE
                if RUST_AVAILABLE:
                    self._rust_backend = RustParserBackend()
                    logger.info("Rust parser backend initialized")
                else:
                    logger.debug("Rust parser not available, using Python fallback")
            except ImportError as e:
                logger.debug(f"Rust parser backend not available: {e}")
        
        self._rebuild_cache()
    
    def _rebuild_cache(self) -> None:
        """Rebuild trigger cache from registry."""
        self._trigger_cache = self.registry.get_all_triggers()
        logger.debug(f"Rebuilt trigger cache with {len(self._trigger_cache)} triggers")
        
        # Also update Rust backend if available
        if self._rust_backend:
            try:
                self._rust_backend.load_triggers(self._trigger_cache)
            except Exception as e:
                logger.warning(f"Failed to update Rust backend triggers: {e}")
    
    def _is_natural_language_query(self, user_input: str) -> bool:
        """
        Detect if input is a natural language query that should use LLM-first.
        
        Args:
            user_input: User input to check
            
        Returns:
            True if this looks like a natural language query
        """
        input_lower = user_input.lower().strip()
        
        # Check word count (questions usually have 3+ words)
        words = input_lower.split()
        if len(words) >= 4:  # "what files are in downloads" = 5 words
            return True
        
        # Check for natural language markers
        for marker in self.NL_MARKERS:
            if input_lower.startswith(marker):
                return True
        
        # Check for question patterns (ends with ? or contains question words)
        if '?' in input_lower:
            return True
        
        # Check for conversational patterns
        if any(pattern in input_lower for pattern in [" in ", " are ", " is ", " of ", " from "]):
            if len(words) >= 3:
                return True
        
        return False
    
    def _calculate_similarity(self, input_str: str, pattern: str) -> float:
        """
        Calculate similarity score between input and pattern.
        
        Optimized for <50ms performance. Uses Rust backend if available.
        """
        # Try Rust backend first if available
        if self._rust_backend:
            try:
                return self._rust_backend.calculate_similarity(input_str, pattern)
            except Exception as e:
                logger.debug(f"Rust similarity calculation failed, using Python: {e}")
        
        # Fallback to Python implementation
        # Fast path: exact substring match
        if pattern in input_str:
            return 1.0
        
        # Token-based matching (fast)
        input_tokens = set(input_str.split())
        pattern_tokens = set(pattern.split())
        
        if not pattern_tokens:
            return 0.0
        
        token_overlap = len(input_tokens & pattern_tokens) / len(pattern_tokens)
        
        # Only compute sequence similarity if token overlap is promising
        if token_overlap < 0.3:
            return token_overlap * 0.6
        
        # Sequence similarity (more expensive)
        sequence_similarity = SequenceMatcher(None, input_str, pattern).ratio()
        
        # Weighted combination
        score = (token_overlap * 0.6) + (sequence_similarity * 0.4)
        
        return score
    
    def parse(
        self,
        user_input: str,
        enable_fuzzy: bool = True,
        extract_entities: bool = True,
        use_llm_fallback: bool = True
    ) -> Optional[Union[IntentMatch, AmbiguousMatch]]:
        """
        Parse user input with Semantic Router (LLM-First for natural language).
        
        Flow:
        1. Detect if input is natural language query
        2. If NL query and LLM available: try LLM first
        3. Otherwise: try rule-based matching
        4. If rule-based confidence >= 0.9: return match
        5. If confidence < 0.9 and LLM available: try LLM
        6. If 0.6 <= confidence < 0.9: return ambiguous (only if LLM didn't work)
        
        Args:
            user_input: Raw user input string
            enable_fuzzy: Enable fuzzy matching
            extract_entities: Extract entities from input
            use_llm_fallback: Use LLM for low-confidence inputs
            
        Returns:
            IntentMatch, AmbiguousMatch, or None
        """
        # Expand environment variables first
        expanded_input = self._entity_extractor.expand_env_vars(user_input)
        normalized_input = expanded_input.lower().strip()
        
        if not normalized_input:
            return None
        
        # Extract entities
        entities = []
        if extract_entities:
            entities = self._entity_extractor.extract(expanded_input)
            logger.debug(f"Extracted {len(entities)} entities: {[e.type for e in entities]}")
        
        # --- SEMANTIC ROUTER: LLM-First for Natural Language ---
        is_nl_query = self._is_natural_language_query(user_input)
        
        if is_nl_query and use_llm_fallback and self.ai_bridge and self.ai_bridge.is_available():
            logger.info(f"Natural language query detected, routing to LLM: '{user_input[:50]}...'")
            llm_result = self._try_llm_fallback(user_input, entities)
            
            if llm_result:
                logger.info(f"LLM successfully interpreted: {llm_result.intent_name}")
                return llm_result
            else:
                logger.warning("LLM failed to interpret natural language query, falling back to rule-based")
        
        # --- RULE-BASED MATCHING (fallback or exact commands) ---
        # Try Rust backend first if available
        if self._rust_backend:
            try:
                rust_result = self._rust_backend.match_intent(normalized_input)
                if rust_result:
                    from intellishell.parser_rust import convert_rust_match_to_python
                    python_result = convert_rust_match_to_python(rust_result, user_input)
                    
                    if python_result:
                        # Add entities to result
                        if isinstance(python_result, IntentMatch):
                            python_result.entities = entities
                        elif isinstance(python_result, AmbiguousMatch):
                            for sug in python_result.suggestions:
                                sug.entities = entities
                        
                        logger.debug("Rust parser matched intent successfully")
                        return python_result
            except Exception as e:
                logger.debug(f"Rust matching failed, using Python: {e}")
        
        # Fallback to Python implementation
        matches: List[IntentMatch] = []
        
        for provider, trigger in self._trigger_cache:
            score = self._calculate_similarity(normalized_input, trigger.pattern)
            
            if score >= self.MIN_CONFIDENCE:
                matches.append(IntentMatch(
                    intent_name=trigger.intent_name,
                    provider_name=provider.name,
                    confidence=score,
                    trigger_pattern=trigger.pattern,
                    original_input=user_input,
                    entities=entities.copy(),
                    source="rule_based"
                ))
        
        if matches:
            # Sort by confidence
            matches.sort(key=lambda m: m.confidence, reverse=True)
            best_match = matches[0]
            
            # Very high confidence: execute directly (exact commands)
            if best_match.confidence >= self.CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"Rule-based match: {best_match.intent_name} "
                    f"(confidence: {best_match.confidence:.2f})"
                )
                return best_match
            
            # For lower confidence: try LLM if available (not already tried)
            best_score = best_match.confidence
        else:
            best_score = 0.0
        
        # --- LLM FALLBACK (if not already tried) ---
        if not is_nl_query and use_llm_fallback and self.ai_bridge and self.ai_bridge.is_available():
            if best_score < self.CONFIDENCE_THRESHOLD:
                logger.info(f"Confidence {best_score:.2f} below threshold, trying LLM fallback")
                llm_result = self._try_llm_fallback(user_input, entities)
                
                if llm_result:
                    logger.info(f"LLM successfully interpreted: {llm_result.intent_name}")
                    return llm_result
        
        # If we have matches but LLM didn't help, handle ambiguity
        if matches:
            best_match = matches[0]
            # Ambiguity zone: return suggestions (only if LLM didn't work)
            if self.AMBIGUITY_ZONE[0] <= best_match.confidence < self.AMBIGUITY_ZONE[1]:
                suggestions = matches[:3]
                logger.debug(
                    f"Ambiguous input. Top match: {best_match.intent_name} "
                    f"(confidence: {best_match.confidence:.2f})"
                )
                return AmbiguousMatch(
                    original_input=user_input,
                    suggestions=suggestions
                )
        
        # No match found
        logger.debug(f"No match found for input: {user_input}")
        return None
    
    def _try_llm_fallback(
        self,
        user_input: str,
        entities: List[Entity]
    ) -> Optional[IntentMatch]:
        """
        Try LLM fallback for low-confidence inputs.
        
        Args:
            user_input: User input
            entities: Extracted entities
            
        Returns:
            IntentMatch if LLM succeeded, None otherwise
        """
        if not self.ai_bridge:
            return None
        
        try:
            llm_response = self.ai_bridge.interpret_command(user_input)
            
            if not llm_response.success or not llm_response.intent_request:
                logger.warning(f"LLM interpretation failed: {llm_response.message}")
                return None
            
            intent_req = llm_response.intent_request
            
            # Convert LLM response to IntentMatch
            intent_match = IntentMatch(
                intent_name=intent_req.intent,
                provider_name=intent_req.provider,
                confidence=intent_req.confidence,
                trigger_pattern="<LLM generated>",
                original_input=user_input,
                entities=entities,
                source="llm",
                parameters=intent_req.parameters or {}
            )
            
            logger.info(
                f"LLM match: {intent_match.intent_name} "
                f"(confidence: {intent_match.confidence:.2f})"
            )
            
            return intent_match
            
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return None
    
    def get_debug_scores(
        self,
        user_input: str,
        top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Get debug scoring information for telemetry.
        
        Args:
            user_input: Raw user input
            top_n: Number of top matches to return
            
        Returns:
            List of (intent_name, pattern, score) tuples
        """
        expanded_input = self._entity_extractor.expand_env_vars(user_input)
        normalized_input = expanded_input.lower().strip()
        scores = []
        
        for provider, trigger in self._trigger_cache:
            score = self._calculate_similarity(normalized_input, trigger.pattern)
            scores.append((trigger.intent_name, trigger.pattern, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)
        
        return scores[:top_n]
    
    def expand_variables(self, text: str) -> str:
        """
        Public API for environment variable expansion.
        
        Args:
            text: Text with potential env vars
            
        Returns:
            Expanded text
        """
        return self._entity_extractor.expand_env_vars(text)
