"""Async REPL loop with self-healing and production safety."""

import asyncio
import sys
import uuid
from datetime import datetime
from typing import Optional
from intent_shell.providers.registry import ProviderRegistry
from intent_shell.parser import SemanticParser, IntentMatch, AmbiguousMatch
from intent_shell.planner import ExecutionPlanner
from intent_shell.session import SessionState
from intent_shell.safety import SafetyController
from intent_shell.executor import SelfHealingExecutor
from intent_shell.utils.clipboard import copy_to_clipboard, should_pipe_to_clipboard, GlobalContext
from intent_shell.utils.transaction_log import TransactionLogger
from intent_shell.memory import SemanticMemory
from intent_shell.ai_bridge import AIBridge
from intent_shell.validation import SelfCorrection
from intent_shell.utils.terminal import enable_royal_blue_terminal, reset_terminal_color, TerminalColors
from intent_shell.utils.display import format_message
import logging

logger = logging.getLogger(__name__)


class IntentShell:
    """
    Async interactive REPL with self-healing and production safety.
    
    Features:
    - Try-repair-retry execution loop
    - Human-in-the-loop safety controls
    - Circuit breaker for failing commands
    - System health diagnostics
    """
    
    def __init__(
        self,
        debug: bool = False,
        enable_ai: bool = True,
        enable_memory: bool = True,
        enable_self_healing: bool = True
    ):
        self.debug = debug
        self.enable_ai = enable_ai
        self.enable_memory = enable_memory
        self.enable_self_healing = enable_self_healing
        self.running = True
        self.prompt = "intent> "
        
        # Initialize semantic memory
        self.semantic_memory = None
        if enable_memory:
            try:
                self.semantic_memory = SemanticMemory()
                if self.semantic_memory.is_available():
                    logger.info("Semantic memory initialized")
                else:
                    logger.warning("Semantic memory not available")
                    self.semantic_memory = None
            except Exception as e:
                logger.warning(f"Failed to initialize semantic memory: {e}")
                self.semantic_memory = None
        
        # Initialize provider registry first (needed for AI bridge)
        self.registry = ProviderRegistry()
        self.registry.auto_discover(semantic_memory=self.semantic_memory)
        
        # Initialize AI bridge with registry for dynamic tool mapping
        self.ai_bridge = None
        if enable_ai:
            try:
                self.ai_bridge = AIBridge(provider_registry=self.registry)
                if self.ai_bridge.is_available():
                    logger.info("AI bridge initialized (Ollama available)")
                else:
                    logger.info("AI bridge initialized but Ollama not running")
            except Exception as e:
                logger.warning(f"Failed to initialize AI bridge: {e}")
        
        # Initialize parser
        self.parser = SemanticParser(self.registry, ai_bridge=self.ai_bridge)
        
        # Initialize safety controller
        self.safety_controller = SafetyController()
        
        # Initialize planner with safety
        self.planner = ExecutionPlanner(self.registry, safety_controller=self.safety_controller)
        
        # Initialize self-healing executor
        self.self_healing_executor = None
        if enable_self_healing and self.ai_bridge:
            self.self_healing_executor = SelfHealingExecutor(
                planner=self.planner,
                ai_bridge=self.ai_bridge
            )
        
        # Initialize session state
        self.session = SessionState(
            session_id=str(uuid.uuid4())[:8],
            start_time=datetime.now()
        )
        
        # Initialize global context
        self.global_context = GlobalContext()
        
        # Initialize transaction logger
        self.transaction_logger = TransactionLogger()
        
        # Initialize self-correction
        self.self_correction = SelfCorrection()
        
        logger.info(f"Intent Shell initialized (session: {self.session.session_id})")
    
    def print_banner(self) -> None:
        """Display welcome banner."""
        banner = r"""
$$$$$$\            $$\                          $$\            $$$$$$\  $$\                 $$\ $$\ 
\_$$  _|           $$ |                         $$ |          $$  __$$\ $$ |                $$ |$$ |
  $$ |  $$$$$$$\ $$$$$$\    $$$$$$\  $$$$$$$\ $$$$$$\         $$ /  \__|$$$$$$$\   $$$$$$\  $$ |$$ |
  $$ |  $$  __$$\\_$$  _|  $$  __$$\ $$  __$$\\_$$  _|        \$$$$$$\  $$  __$$\ $$  __$$\ $$ |$$ |
  $$ |  $$ |  $$ | $$ |    $$$$$$$$ |$$ |  $$ | $$ |           \____$$\ $$ |  $$ |$$$$$$$$ |$$ |$$ |
  $$ |  $$ |  $$ | $$ |$$\ $$   ____|$$ |  $$ | $$ |$$\       $$\   $$ |$$ |  $$ |$$   ____|$$ |$$ |
$$$$$$\ $$ |  $$ | \$$$$  |\$$$$$$$\ $$ |  $$ | \$$$$  |      \$$$$$$  |$$ |  $$ |\$$$$$$$\ $$ |$$ |
\______|\__|  \__|  \____/  \_______|\__|  \__|  \____/        \______/ \__|  \__| \_______|\__|\__|

v0.1 - Self-Healing & Production Safety
Try-Repair-Retry Loop | HITL Safety | Circuit Breaker
Type 'help' for commands, 'check system health' to test
"""
        print(banner)
        
        status_parts = []
        if self.semantic_memory and self.semantic_memory.is_available():
            status_parts.append("✓ Semantic Memory")
        else:
            status_parts.append("✗ Semantic Memory")
        
        if self.ai_bridge and self.ai_bridge.is_available():
            status_parts.append("✓ AI Bridge")
        else:
            status_parts.append("✗ AI Bridge")
        
        if self.enable_self_healing:
            status_parts.append("✓ Self-Healing")
        else:
            status_parts.append("✗ Self-Healing")
        
        print(f"Status: {' | '.join(status_parts)}")
        
        if self.debug:
            print(f"\n[DEBUG MODE] Session: {self.session.session_id}")
            print(f"[DEBUG MODE] Providers: {len(self.registry.get_all_providers())}")
        print()
    
    async def process_command(self, user_input: str) -> bool:
        """Process a single user command."""
        if not user_input.strip():
            return True
        
        # Update global clipboard context
        self.global_context.update()
        
        # Check for exit
        if user_input.lower().strip() in ["exit", "quit", "bye"]:
            print("Exiting Intent Shell...")
            return False
        
        # Special commands (check before parsing)
        user_input_lower = user_input.lower().strip()
        if user_input_lower in ["help", "?"]:
            self._show_help()
            return True
        
        # Intercept help-related natural language queries
        help_patterns = [
            "what are the available commands",
            "what commands are available",
            "show me the commands",
            "list commands",
            "what can I do",
            "show help",
            "show available commands"
        ]
        if any(pattern in user_input_lower for pattern in help_patterns):
            info_msg = format_message("ℹ To see available commands, type 'help' or use the 'manifest' command.", success=False, is_warning=True)
            print(info_msg)
            self._show_help()
            return True
        
        if user_input.lower().strip() in ["stats", "session stats"]:
            self._show_stats()
            return True
        
        if user_input.lower().strip() == "manifest":
            self._show_manifest()
            return True
        
        if user_input.lower().strip() in ["history", "hist"]:
            self._show_history()
            return True
        
        if user_input.startswith("!"):
            return await self._replay_history(user_input)
        
        # Check for clipboard piping
        pipe_to_clipboard, cleaned_input = should_pipe_to_clipboard(user_input)
        
        # Show "thinking" status for natural language queries
        is_nl_query = hasattr(self.parser, '_is_natural_language_query') and \
                     self.parser._is_natural_language_query(cleaned_input)
        if is_nl_query and self.ai_bridge and self.ai_bridge.is_available():
            thinking_msg = format_message(f"[Thinking: Mapping '{cleaned_input[:50]}...' to intent...]", success=False, is_warning=False)
            print(thinking_msg)
        
        # Parse intent
        parse_result = self.parser.parse(
            cleaned_input,
            use_llm_fallback=(self.ai_bridge is not None)
        )
        
        # Handle no match
        if parse_result is None:
            error_msg = format_message(f"Unknown command: '{user_input}'", success=False, is_error=True)
            warning_msg = format_message("Type 'help' for available commands.", success=False, is_warning=True)
            print(error_msg)
            print(warning_msg)
            
            if self.debug:
                print("\n[DEBUG] Top scoring intents:")
                scores = self.parser.get_debug_scores(cleaned_input)
                for intent, pattern, score in scores:
                    print(f"  - {intent}: {pattern} ({score:.2f})")
            
            self.transaction_logger.log_transaction(
                user_input, None, None, 0.0, False
            )
            self.session.add_command(user_input, None, False, 0.0)
            return True
        
        # Handle ambiguous match
        if isinstance(parse_result, AmbiguousMatch):
            self._handle_ambiguity(parse_result)
            return True
        
        # Handle clear match
        intent_match: IntentMatch = parse_result
        
        # Self-correction
        needs_correction, corrected_input, correction_msgs = self.self_correction.validate_and_correct(
            user_input,
            intent_match.intent_name,
            [{"type": e.type, "value": e.value} for e in intent_match.entities]
        )
        
        if needs_correction and correction_msgs:
            for msg in correction_msgs:
                info_msg = format_message(f"ℹ {msg}", success=False, is_warning=True)
                print(info_msg)
        
        # Show debug info
        if self.debug:
            print(f"[DEBUG] Intent: {intent_match.intent_name}")
            print(f"[DEBUG] Provider: {intent_match.provider_name}")
            print(f"[DEBUG] Confidence: {intent_match.confidence:.2f}")
            print(f"[DEBUG] Source: {intent_match.source}")
            print(f"[DEBUG] Safety: {self.safety_controller.get_safety_level(intent_match.intent_name).name}")
            print()
        
        # Execute intent with self-healing
        try:
            # Build context
            execution_context = {
                "original_input": user_input,
                "entities": intent_match.entities,
                "clipboard": self.global_context.get_current(),
                "session_id": self.session.session_id
            }
            
            # Use self-healing executor if available
            if self.self_healing_executor:
                result, repair_attempt = await self.self_healing_executor.execute_with_healing(
                    intent_match,
                    execution_context
                )
                
                if repair_attempt:
                    if self.debug:
                        print(f"[DEBUG] Repair: {repair_attempt.status.value}")
            else:
                result = await self.planner.execute_intent(intent_match, execution_context)
            
            # Display result with color coding
            if result.success:
                # Success messages - already colored in providers, but ensure green for simple success
                print(result.message)
            else:
                # Error messages - color red
                colored_message = format_message(result.message, success=False, is_error=True)
                print(colored_message)
            
            # Pipe to clipboard if requested
            if pipe_to_clipboard and result.success:
                if copy_to_clipboard(result.message):
                    success_msg = format_message("✓ Copied to clipboard", success=True)
                    print(success_msg)
            
            # Store in semantic memory
            if self.semantic_memory and result.success:
                self.semantic_memory.remember(
                    user_input=user_input,
                    intent_name=intent_match.intent_name,
                    provider_name=intent_match.provider_name,
                    output_summary=result.message[:500],
                    confidence=intent_match.confidence,
                    success=True,
                    entities=[{"type": e.type, "value": e.value} for e in intent_match.entities],
                    metadata=result.metadata
                )
            
            # Add to AI context
            if self.ai_bridge:
                self.ai_bridge.add_context(
                    user_input=user_input,
                    intent_name=intent_match.intent_name,
                    result=result.message[:200],
                    entities=[{"type": e.type, "value": e.value} for e in intent_match.entities]
                )
            
            # Log transaction
            self.transaction_logger.log_transaction(
                user_input=user_input,
                intent_name=intent_match.intent_name,
                provider_name=intent_match.provider_name,
                confidence=intent_match.confidence,
                success=result.success,
                result_message=result.message,
                entities=[{"type": e.type, "value": e.value} for e in intent_match.entities],
                metadata=result.metadata
            )
            
            # Track in session
            self.session.add_command(
                user_input,
                intent_match.intent_name,
                result.success,
                intent_match.confidence
            )
            
            # Update context
            if result.data and "path" in result.data:
                from pathlib import Path
                self.session.last_directory = Path(result.data["path"])
                if self.ai_bridge:
                    self.ai_bridge.context_manager.update_directory(str(result.data["path"]))
            
        except Exception as e:
            error_msg = format_message(f"Error executing command: {e}", success=False, is_error=True)
            print(error_msg)
            logger.exception("Command execution failed")
            
            self.transaction_logger.log_transaction(
                user_input, intent_match.intent_name, intent_match.provider_name,
                intent_match.confidence, False, str(e)
            )
            self.session.add_command(user_input, intent_match.intent_name, False, intent_match.confidence)
        
        return True
    
    def _handle_ambiguity(self, ambiguous: AmbiguousMatch) -> None:
        """Handle ambiguous matches."""
        print(f"Did you mean...?\n")
        
        for i, suggestion in enumerate(ambiguous.suggestions, 1):
            print(f"  {i}. {suggestion.trigger_pattern} ({suggestion.confidence:.2f})")
            print(f"     → {suggestion.intent_name} via {suggestion.provider_name}")
        
        print(f"\nPlease rephrase your command to be more specific.")
    
    async def _replay_history(self, command: str) -> bool:
        """Replay a command from history."""
        try:
            index = int(command[1:])
            history = self.transaction_logger.read_history(limit=100)
            
            if index < 1 or index > len(history):
                print(f"Invalid history index: {index}. Valid range: 1-{len(history)}")
                return True
            
            replay_cmd = history[index - 1]["user_input"]
            print(f"Replaying: {replay_cmd}")
            
            return await self.process_command(replay_cmd)
        except ValueError:
            print("Invalid history replay format. Use: !N (e.g., !5)")
            return True
    
    def _show_help(self) -> None:
        """Display help information."""
        print("\nIntent Shell - Available Commands:")
        print("=" * 60)
        
        providers = self.registry.get_all_providers()
        for provider in providers:
            print(f"\n{provider.name.upper()}: {provider.description}")
            triggers = provider.get_triggers()
            for trigger in triggers[:3]:
                print(f"  • {trigger.pattern}")
        
        print("\nSPECIAL COMMANDS:")
        print("  • help / ? - Show this help")
        print("  • stats - Show session statistics")
        print("  • manifest - Show system manifest")
        print("  • history - Show command history")
        print("  • !N - Replay command N")
        print("  • exit / quit - Exit")
        print()
    
    def _show_stats(self) -> None:
        """Display session statistics."""
        stats = self.session.get_stats()
        print("\nSession Statistics:")
        print("=" * 60)
        print(f"Session ID: {stats['session_id']}")
        print(f"Duration: {stats['duration']:.1f} seconds")
        print(f"Total Commands: {stats['total_commands']}")
        print(f"Successful: {stats['successful_commands']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        
        # Safety summary
        safety_summary = self.planner.get_safety_summary()
        print(f"\nSafety:")
        print(f"  Last Action Failed: {safety_summary['last_action_failed']}")
        print(f"  RED Actions: {safety_summary['red_actions']}")
        
        # Circuit breaker status
        if self.self_healing_executor:
            print(f"\nCircuit Breaker: Active")
        
        print()
    
    def _show_manifest(self) -> None:
        """Display system manifest."""
        manifest = self.registry.generate_manifest()
        
        print("\nIntent Shell Manifest:")
        print("=" * 60)
        print(f"Version: {manifest['version']}")
        print(f"Total Commands: {manifest['total_commands']}")
        print(f"Providers: {len(manifest['providers'])}")
        print()
    
    def _show_history(self) -> None:
        """Display command history."""
        history = self.transaction_logger.read_history(limit=20)
        
        if not history:
            print("No command history available.")
            return
        
        print("\nCommand History (last 20):")
        print("=" * 60)
        
        for i, tx in enumerate(history, 1):
            status = "✓" if tx.get("success") else "✗"
            timestamp = tx.get("timestamp", "")[:19]
            cmd = tx.get("user_input", "")
            
            print(f"{i:3d}. {status} [{timestamp}] {cmd}")
        
        print("\nUse !N to replay a command")
        print()
    
    async def run(self) -> None:
        """Start the async REPL loop."""
        # Enable royal blue terminal color
        enable_royal_blue_terminal()
        
        try:
            self.print_banner()
            
            while self.running:
                try:
                    # Prompt with royal blue color
                    colored_prompt = TerminalColors.colorize(self.prompt, TerminalColors.ROYAL_BLUE) if TerminalColors.supports_color() else self.prompt
                    user_input = await asyncio.to_thread(input, colored_prompt)
                    self.running = await self.process_command(user_input)
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit.")
                    continue
                except EOFError:
                    exit_msg = format_message("\nExiting Intent Shell...", success=True)
                    print(exit_msg)
                    break
                except Exception as e:
                    error_msg = format_message(f"Error: {e}", success=False, is_error=True)
                    print(error_msg)
                    logger.exception("REPL error")
                    continue
            
            if self.debug:
                print()
                self._show_stats()
        finally:
            # Reset terminal color on exit
            reset_terminal_color()


def main(
    debug: bool = False,
    enable_ai: bool = True,
    enable_memory: bool = True,
    enable_self_healing: bool = True
) -> None:
    """Entry point for Intent Shell REPL."""
    shell = IntentShell(
        debug=debug,
        enable_ai=enable_ai,
        enable_memory=enable_memory,
        enable_self_healing=enable_self_healing
    )
    asyncio.run(shell.run())


if __name__ == "__main__":
    main()
