"""Smart contextual autocomplete with usage tracking and preview generation."""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, time as datetime_time
from collections import defaultdict, Counter
import json
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class CompletionStats:
    """Tracks completion usage statistics for smart ranking."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize completion stats tracker.
        
        Args:
            storage_path: Path to store completion stats
        """
        if storage_path is None:
            storage_dir = Path.home() / ".intellishell"
            storage_dir.mkdir(parents=True, exist_ok=True)
            storage_path = storage_dir / "completion_stats.json"
        
        self.storage_path = storage_path
        
        # Frequency tracking
        self.command_frequency: Counter = Counter()
        self.command_recency: Dict[str, str] = {}  # command -> last_used_timestamp
        
        # Time-of-day patterns
        self.time_patterns: Dict[str, List[int]] = defaultdict(list)  # command -> [hours]
        
        # Context tracking
        self.command_sequences: List[Tuple[str, str]] = []  # [(prev_cmd, next_cmd)]
        
        # Load existing stats
        self._load_stats()
    
    def _load_stats(self) -> None:
        """Load statistics from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Load frequency
                self.command_frequency = Counter(data.get("frequency", {}))
                
                # Load recency
                self.command_recency = data.get("recency", {})
                
                # Load time patterns
                time_data = data.get("time_patterns", {})
                self.time_patterns = defaultdict(list, {k: v for k, v in time_data.items()})
                
                # Load sequences (limit to last 100)
                sequences = data.get("sequences", [])
                self.command_sequences = [tuple(s) for s in sequences[-100:]]
                
            logger.info(f"Loaded completion stats: {len(self.command_frequency)} commands")
        except Exception as e:
            logger.error(f"Failed to load completion stats: {e}")
    
    def _save_stats(self) -> None:
        """Save statistics to disk."""
        try:
            data = {
                "frequency": dict(self.command_frequency),
                "recency": self.command_recency,
                "time_patterns": {k: v for k, v in self.time_patterns.items()},
                "sequences": [list(s) for s in self.command_sequences[-100:]]
            }
            
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save completion stats: {e}")
    
    def record_command(self, command: str, prev_command: Optional[str] = None) -> None:
        """
        Record a command execution for statistics.
        
        Args:
            command: Command that was executed
            prev_command: Previous command (for sequence tracking)
        """
        command = command.strip().lower()
        
        # Update frequency
        self.command_frequency[command] += 1
        
        # Update recency
        self.command_recency[command] = datetime.now().isoformat()
        
        # Update time patterns
        current_hour = datetime.now().hour
        self.time_patterns[command].append(current_hour)
        
        # Keep only last 20 time entries per command
        if len(self.time_patterns[command]) > 20:
            self.time_patterns[command] = self.time_patterns[command][-20:]
        
        # Update sequences
        if prev_command:
            self.command_sequences.append((prev_command.strip().lower(), command))
            # Keep only last 100 sequences
            if len(self.command_sequences) > 100:
                self.command_sequences = self.command_sequences[-100:]
        
        # Save periodically (every 5 commands)
        if self.command_frequency[command] % 5 == 0:
            self._save_stats()
    
    def get_frequency_score(self, command: str) -> float:
        """
        Get frequency score for a command (0.0-1.0).
        
        Args:
            command: Command to score
            
        Returns:
            Normalized frequency score
        """
        command = command.strip().lower()
        count = self.command_frequency.get(command, 0)
        
        if not self.command_frequency:
            return 0.0
        
        max_count = max(self.command_frequency.values())
        return count / max_count if max_count > 0 else 0.0
    
    def get_recency_score(self, command: str) -> float:
        """
        Get recency score for a command (0.0-1.0).
        
        Args:
            command: Command to score
            
        Returns:
            Recency score (1.0 = used recently, 0.0 = never used or very old)
        """
        command = command.strip().lower()
        last_used = self.command_recency.get(command)
        
        if not last_used:
            return 0.0
        
        try:
            last_used_dt = datetime.fromisoformat(last_used)
            now = datetime.now()
            
            # Calculate hours since last use
            hours_ago = (now - last_used_dt).total_seconds() / 3600
            
            # Decay function: 1.0 at 0 hours, 0.5 at 24 hours, 0.0 at 168 hours (1 week)
            if hours_ago <= 0:
                return 1.0
            elif hours_ago >= 168:
                return 0.0
            else:
                return max(0.0, 1.0 - (hours_ago / 168))
        except Exception:
            return 0.0
    
    def get_time_relevance_score(self, command: str) -> float:
        """
        Get time-of-day relevance score (0.0-1.0).
        
        Args:
            command: Command to score
            
        Returns:
            Time relevance score based on historical usage patterns
        """
        command = command.strip().lower()
        hours = self.time_patterns.get(command, [])
        
        if not hours:
            return 0.5  # Neutral score if no data
        
        current_hour = datetime.now().hour
        
        # Count how many times this command was used at similar hours
        similar_hour_count = sum(1 for h in hours if abs(h - current_hour) <= 2)
        
        return similar_hour_count / len(hours) if hours else 0.5
    
    def get_sequence_score(self, command: str, prev_command: Optional[str]) -> float:
        """
        Get sequence score based on command history (0.0-1.0).
        
        Args:
            command: Command to score
            prev_command: Previous command
            
        Returns:
            Sequence score (how often this command follows prev_command)
        """
        if not prev_command or not self.command_sequences:
            return 0.0
        
        command = command.strip().lower()
        prev_command = prev_command.strip().lower()
        
        # Count how many times command follows prev_command
        follow_count = sum(1 for prev, next_cmd in self.command_sequences 
                          if prev == prev_command and next_cmd == command)
        
        # Count total times prev_command appears
        prev_count = sum(1 for prev, _ in self.command_sequences if prev == prev_command)
        
        return follow_count / prev_count if prev_count > 0 else 0.0
    
    def get_combined_score(
        self,
        command: str,
        prev_command: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Get combined relevance score for a command.
        
        Args:
            command: Command to score
            prev_command: Previous command for sequence scoring
            weights: Custom weights for different factors
            
        Returns:
            Combined score (0.0-1.0)
        """
        if weights is None:
            weights = {
                "frequency": 0.3,
                "recency": 0.3,
                "time": 0.2,
                "sequence": 0.2
            }
        
        frequency_score = self.get_frequency_score(command)
        recency_score = self.get_recency_score(command)
        time_score = self.get_time_relevance_score(command)
        sequence_score = self.get_sequence_score(command, prev_command)
        
        combined = (
            frequency_score * weights["frequency"] +
            recency_score * weights["recency"] +
            time_score * weights["time"] +
            sequence_score * weights["sequence"]
        )
        
        return combined


class FuzzyMatcher:
    """Fuzzy matching with typo correction."""
    
    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings (0.0-1.0).
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score
        """
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    @staticmethod
    def fuzzy_match(query: str, candidates: List[str], threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Find fuzzy matches for a query.
        
        Args:
            query: Search query
            candidates: List of candidate strings
            threshold: Minimum similarity threshold
            
        Returns:
            List of (candidate, score) tuples, sorted by score
        """
        matches = []
        query_lower = query.lower()
        
        for candidate in candidates:
            # Exact prefix match gets bonus
            if candidate.lower().startswith(query_lower):
                score = 1.0
            # Contains match gets high score
            elif query_lower in candidate.lower():
                score = 0.9
            # Fuzzy match
            else:
                score = FuzzyMatcher.similarity(query, candidate)
            
            if score >= threshold:
                matches.append((candidate, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


class CompletionPreview:
    """Generate previews for completions."""
    
    # Preview templates for common commands
    PREVIEWS = {
        "open desktop": "Open Desktop folder",
        "open downloads": "Open Downloads folder",
        "open documents": "Open Documents folder",
        "list downloads": "List files in Downloads",
        "list desktop": "List files on Desktop",
        "list processes": "Show running processes",
        "system info": "Display system information",
        "clipboard history": "Show clipboard history",
        "clipboard search": "Search clipboard history",
        "get hostname": "Show computer name",
        "get username": "Show current user",
        "disk space": "Show disk usage",
        "check system health": "Run system diagnostics",
        "watch downloads": "Monitor Downloads folder",
        "stop watching": "Stop file monitoring",
        "history": "Show command history",
        "stats": "Show session statistics",
        "manifest": "Show system manifest",
        "help": "Show available commands",
    }
    
    @staticmethod
    def get_preview(command: str, stats: Optional[CompletionStats] = None) -> str:
        """
        Get preview text for a command.
        
        Args:
            command: Command to preview
            stats: Optional stats for usage info
            
        Returns:
            Preview text
        """
        command_lower = command.lower().strip()
        
        # Check for exact preview
        if command_lower in CompletionPreview.PREVIEWS:
            preview = CompletionPreview.PREVIEWS[command_lower]
        else:
            # Generate generic preview
            preview = f"Execute: {command}"
        
        # Add usage stats if available
        if stats:
            freq_score = stats.get_frequency_score(command)
            if freq_score > 0:
                count = stats.command_frequency.get(command_lower, 0)
                preview += f" (used {count}x)"
        
        return preview
    
    @staticmethod
    def get_time_context(stats: CompletionStats, command: str) -> str:
        """
        Get time context for a command.
        
        Args:
            stats: Completion stats
            command: Command to check
            
        Returns:
            Time context string
        """
        hours = stats.time_patterns.get(command.lower().strip(), [])
        if not hours:
            return ""
        
        # Find most common hour
        if hours:
            hour_counts = Counter(hours)
            most_common_hour = hour_counts.most_common(1)[0][0]
            
            # Format hour
            if most_common_hour < 12:
                time_str = f"{most_common_hour}am"
            elif most_common_hour == 12:
                time_str = "12pm"
            else:
                time_str = f"{most_common_hour - 12}pm"
            
            return f"Usually used around {time_str}"
        
        return ""
