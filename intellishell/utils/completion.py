"""Tab completion for IntelliShell commands."""

from typing import Optional, List, Iterable
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from intellishell.utils.contextual_completion import (
    CompletionStats,
    FuzzyMatcher,
    CompletionPreview
)
import logging

logger = logging.getLogger(__name__)


class IntelliShellCompleter(Completer):
    """
    Smart tab completion for IntelliShell with contextual awareness.
    
    Features:
    - Frequency-based ranking
    - Recency scoring
    - Time-of-day patterns
    - Command sequence awareness
    - Fuzzy matching with typo correction
    - Preview generation
    """
    
    def __init__(
        self,
        provider_registry=None,
        parser=None,
        enable_smart_features: bool = True
    ):
        """
        Initialize completer.
        
        Args:
            provider_registry: ProviderRegistry instance for getting available intents
            parser: SemanticParser instance for command parsing
            enable_smart_features: Enable contextual smart features
        """
        self.provider_registry = provider_registry
        self.parser = parser
        self.enable_smart_features = enable_smart_features
        
        # Initialize contextual features
        if enable_smart_features:
            try:
                self.stats = CompletionStats()
                self.fuzzy_matcher = FuzzyMatcher()
                logger.info("Smart autocomplete features enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize smart features: {e}")
                self.stats = None
                self.fuzzy_matcher = None
        else:
            self.stats = None
            self.fuzzy_matcher = None
        
        # Track last command for sequence awareness
        self.last_command: Optional[str] = None
        
        # Built-in commands
        self.builtin_commands = [
            "help", "?", "manifest", "history", "hist", "stats", "session stats",
            "clear", "cls", "exit", "quit", "bye"
        ]
        
        # Special command prefixes
        self.command_prefixes = ["!", "?", "help", "open", "list", "show", "get", 
                                "check", "kill", "watch", "what", "where", "clipboard"]
    
    def get_completions(
        self, 
        document: Document, 
        complete_event
    ) -> Iterable[Completion]:
        """
        Get smart contextual completions for current input.
        
        Args:
            document: Current document (input text)
            complete_event: Completion event
            
        Yields:
            Completion objects with previews and smart ranking
        """
        text = document.text_before_cursor
        text_lower = text.lower().strip()
        
        # Calculate how many characters to replace
        start_position = -len(text)
        
        # Collect all possible completions with scores
        completion_candidates = []
        
        # If empty, suggest top commands based on context
        if not text or not text.strip():
            if self.enable_smart_features and self.stats:
                # Show most frequent/recent commands
                top_commands = self._get_top_commands(limit=10)
                for cmd in top_commands:
                    preview = CompletionPreview.get_preview(cmd, self.stats)
                    yield Completion(cmd, start_position=0, display_meta=preview)
            else:
                # Fallback to built-in commands
                for cmd in sorted(self.builtin_commands):
                    yield Completion(cmd, start_position=0)
            return
        
        # Collect candidates from various sources
        candidates = set()
        
        # Built-in commands
        candidates.update(self.builtin_commands)
        
        # Provider intents
        if self.provider_registry:
            candidates.update(self._get_all_intents())
        
        # Common patterns
        candidates.update(self._get_common_patterns())
        
        # Use fuzzy matching if smart features enabled
        if self.enable_smart_features and self.fuzzy_matcher:
            matches = self.fuzzy_matcher.fuzzy_match(text, list(candidates), threshold=0.4)
            
            # Score each match
            for candidate, fuzzy_score in matches:
                # Get contextual score
                if self.stats:
                    context_score = self.stats.get_combined_score(
                        candidate,
                        prev_command=self.last_command
                    )
                else:
                    context_score = 0.0
                
                # Combined score: fuzzy match (60%) + context (40%)
                combined_score = (fuzzy_score * 0.6) + (context_score * 0.4)
                
                completion_candidates.append((candidate, combined_score))
        else:
            # Fallback to simple prefix matching
            for candidate in candidates:
                candidate_lower = candidate.lower()
                if candidate_lower.startswith(text_lower):
                    completion_candidates.append((candidate, 1.0))
                elif text_lower in candidate_lower:
                    completion_candidates.append((candidate, 0.8))
        
        # Sort by score (descending)
        completion_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Yield top completions with previews
        for candidate, score in completion_candidates[:20]:  # Limit to top 20
            if self.enable_smart_features and self.stats:
                preview = CompletionPreview.get_preview(candidate, self.stats)
                
                # Add time context if relevant
                time_context = CompletionPreview.get_time_context(self.stats, candidate)
                if time_context:
                    preview += f" | {time_context}"
                
                yield Completion(
                    candidate,
                    start_position=start_position,
                    display_meta=preview
                )
            else:
                yield Completion(candidate, start_position=start_position)
    
    def _get_all_intents(self) -> List[str]:
        """Get all available intents from all providers."""
        if not self.provider_registry:
            return []
        
        intents = set()
        for provider in self.provider_registry.get_all_providers():
            for trigger in provider.get_triggers():
                # Add pattern (main command)
                if trigger.pattern:
                    intents.add(trigger.pattern)
                # Add aliases
                if trigger.aliases:
                    intents.update(trigger.aliases)
        
        return sorted(list(intents))
    
    def _get_common_patterns(self) -> List[str]:
        """Get common command patterns."""
        return [
            "open desktop", "open downloads", "open documents",
            "list downloads", "list desktop", "list files",
            "system info", "get hostname", "get username",
            "check system health", "check dependencies",
            "what did i", "what folder", "recent memories",
            "watch downloads", "watch for pdf", "stop watching",
            "list processes", "kill process",
            "clipboard history", "clipboard search", "clipboard stats",
            "clipboard restore", "clipboard clear"
        ]
    
    def _get_top_commands(self, limit: int = 10) -> List[str]:
        """
        Get top commands based on frequency and recency.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of top commands
        """
        if not self.stats:
            return self.builtin_commands[:limit]
        
        # Get all commands we know about
        all_commands = set(self.builtin_commands)
        all_commands.update(self._get_all_intents())
        all_commands.update(self._get_common_patterns())
        
        # Score each command
        scored_commands = []
        for cmd in all_commands:
            score = self.stats.get_combined_score(cmd, prev_command=self.last_command)
            if score > 0:  # Only include commands with some history
                scored_commands.append((cmd, score))
        
        # Sort by score and return top N
        scored_commands.sort(key=lambda x: x[1], reverse=True)
        return [cmd for cmd, _ in scored_commands[:limit]]
    
    def record_command(self, command: str) -> None:
        """
        Record a command execution for learning.
        
        Args:
            command: Command that was executed
        """
        if self.enable_smart_features and self.stats:
            self.stats.record_command(command, prev_command=self.last_command)
            self.last_command = command.strip().lower()
