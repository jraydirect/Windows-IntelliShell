"""CLI entry point with AI and memory options."""

import argparse
import asyncio
import sys
from intent_shell.main import IntentShell
from intent_shell.providers.registry import ProviderRegistry
from intent_shell.parser import SemanticParser
from intent_shell.planner import ExecutionPlanner
from intent_shell.utils.logging import setup_logging
from intent_shell.utils.clipboard import should_pipe_to_clipboard, copy_to_clipboard
from intent_shell.utils.display import format_message
from intent_shell import __version__
import logging

logger = logging.getLogger(__name__)


async def execute_single_command(
    command: str,
    debug: bool = False,
    enable_ai: bool = True
) -> int:
    """
    Execute a single command and exit.
    
    Args:
        command: Command to execute
        debug: Enable debug mode
        enable_ai: Enable AI bridge
        
    Returns:
        Exit code
    """
    # Initialize components
    from intent_shell.memory import SemanticMemory
    from intent_shell.ai_bridge import AIBridge
    
    semantic_memory = None
    try:
        semantic_memory = SemanticMemory()
        if not semantic_memory.is_available():
            semantic_memory = None
    except:
        pass
    
    registry = ProviderRegistry()
    registry.auto_discover(semantic_memory=semantic_memory)
    
    ai_bridge = None
    if enable_ai:
        try:
            ai_bridge = AIBridge(provider_registry=registry)
            if not ai_bridge.is_available():
                ai_bridge = None
        except:
            pass
    
    parser = SemanticParser(registry, ai_bridge=ai_bridge)
    planner = ExecutionPlanner(registry)
    
    # Check for clipboard piping
    pipe_to_clipboard, cleaned_input = should_pipe_to_clipboard(command)
    
    # Parse intent
    parse_result = parser.parse(cleaned_input, use_llm_fallback=(ai_bridge is not None))
    
    if parse_result is None:
        error_msg = format_message(f"Unknown command: '{command}'", success=False, is_error=True)
        print(error_msg)
        
        if debug:
            debug_msg = format_message("\n[DEBUG] Top scoring intents:", success=False, is_warning=True)
            print(debug_msg)
            scores = parser.get_debug_scores(cleaned_input)
            for intent, pattern, score in scores:
                print(f"  - {intent}: {pattern} ({score:.2f})")
        
        return 1
    
    # Handle ambiguous
    from intent_shell.parser import AmbiguousMatch, IntentMatch
    if isinstance(parse_result, AmbiguousMatch):
        warning_msg = format_message("Ambiguous command. Please be more specific.", success=False, is_warning=True)
        print(warning_msg)
        return 1
    
    intent_match: IntentMatch = parse_result
    
    # Show debug info
    if debug:
        print(f"[DEBUG] Intent: {intent_match.intent_name}")
        print(f"[DEBUG] Provider: {intent_match.provider_name}")
        print(f"[DEBUG] Confidence: {intent_match.confidence:.2f}")
        print(f"[DEBUG] Source: {intent_match.source}")
        print()
    
    # Execute
    try:
        execution_context = {
            "original_input": command,
            "entities": intent_match.entities,
        }
        
        result = await planner.execute_intent(intent_match, execution_context)
        
        # Display with color coding
        if result.success:
            print(result.message)
        else:
            error_msg = format_message(result.message, success=False, is_error=True)
            print(error_msg)
        
        # Pipe to clipboard if requested
        if pipe_to_clipboard and result.success:
            if copy_to_clipboard(result.message):
                success_msg = format_message("✓ Copied to clipboard", success=True)
                print(success_msg)
        
        return 0 if result.success else 1
    except Exception as e:
        error_msg = format_message(f"Error: {e}", success=False, is_error=True)
        print(error_msg)
        logger.exception("Command execution failed")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="intent",
        description="Intent Shell - Semantic command shell with AI reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  intent                           Start interactive shell
  intent -c "open desktop"         Execute single command
  intent --debug                   Start with debug output
  intent --no-ai                   Disable AI fallback
  intent --no-memory               Disable semantic memory
        """
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"Intent Shell v{__version__}",
    )
    
    parser.add_argument(
        "-c", "--command",
        type=str,
        help="Execute a single command and exit",
        metavar="COMMAND",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with telemetry",
    )
    
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI bridge (LLM fallback)",
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable semantic memory (vector storage)",
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Custom log file path",
        metavar="PATH",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    from pathlib import Path
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(debug=args.debug, log_file=log_file)
    
    logger.info(f"Intent Shell v{__version__} starting")
    
    try:
        if args.command:
            # Single command mode
            exit_code = asyncio.run(execute_single_command(
                args.command,
                debug=args.debug,
                enable_ai=(not args.no_ai)
            ))
            return exit_code
        else:
            # Interactive REPL mode
            from intent_shell.main import main as repl_main
            repl_main(
                debug=args.debug,
                enable_ai=(not args.no_ai),
                enable_memory=(not args.no_memory)
            )
            return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
