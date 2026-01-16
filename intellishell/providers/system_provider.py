"""System provider for process management and advanced OS operations."""

import os
import subprocess
import ctypes
from typing import Optional, Dict, Any, List
from intellishell.providers.base import (
    BaseProvider,
    IntentTrigger,
    ExecutionResult,
    ProviderCapability
)
import logging

logger = logging.getLogger(__name__)


class SystemProvider(BaseProvider):
    """Provider for system-level operations requiring elevated privileges."""
    
    @property
    def name(self) -> str:
        return "system"
    
    @property
    def description(self) -> str:
        return "System management and process control"
    
    def _initialize_triggers(self) -> None:
        """Initialize system-related triggers."""
        self.capabilities = [
            ProviderCapability.WRITE,
            ProviderCapability.ASYNC,
        ]
        
        self.triggers = [
            IntentTrigger(
                pattern="kill process",
                intent_name="kill_process",
                weight=1.0,
                aliases=["stop process", "terminate process", "end process"]
            ),
            IntentTrigger(
                pattern="kill notepad",
                intent_name="kill_by_name",
                weight=1.0,
                aliases=["stop notepad", "close notepad", "kill chrome", "kill calculator", 
                        "kill firefox", "kill edge", "kill explorer", "kill discord", 
                        "kill brave", "kill cursor", "kill ollama"]
            ),
            IntentTrigger(
                pattern="most memory",
                intent_name="kill_most_memory",
                weight=0.9,
                aliases=["highest memory", "stop memory hog"]
            ),
            IntentTrigger(
                pattern="list processes",
                intent_name="list_processes",
                weight=1.0,
                aliases=["show processes", "running processes", "ps"]
            ),
            IntentTrigger(
                pattern="check admin",
                intent_name="check_admin",
                weight=1.0,
                aliases=["am i admin", "admin status", "is admin"]
            ),
        ]
    
    def _is_admin(self) -> bool:
        """Check if running with administrator privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    
    async def execute(
        self,
        intent_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute system operation."""
        try:
            if intent_name == "kill_process":
                return await self._kill_process(context)
            elif intent_name == "kill_by_name":
                return await self._kill_by_name(context)
            elif intent_name == "kill_most_memory":
                return await self._kill_most_memory()
            elif intent_name == "list_processes":
                return await self._list_processes()
            elif intent_name == "check_admin":
                return await self._check_admin()
            else:
                return ExecutionResult(
                    success=False,
                    message=f"Unknown intent: {intent_name}"
                )
        except PermissionError as e:
            return self._handle_permission_error(intent_name, e)
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"Execution error: {e}"
            )
    
    def _handle_permission_error(
        self,
        intent_name: str,
        error: Exception
    ) -> ExecutionResult:
        """Handle permission errors with helpful message."""
        message = f"""Permission Denied: {error}

This command requires Administrator privileges.

To run IntelliShell as Administrator:
1. Close this shell
2. Right-click Command Prompt or PowerShell
3. Select "Run as Administrator"
4. Run: intent

Or use: runas /user:Administrator intent
"""
        return ExecutionResult(
            success=False,
            message=message,
            metadata={"requires_admin": True, "intent": intent_name}
        )
    
    async def _kill_process(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Kill a process by PID."""
        # Extract PID from entities or original input
        pid = None
        
        # Try entities first
        if context and "entities" in context:
            for entity in context["entities"]:
                if entity.type == "number":
                    try:
                        pid = int(entity.value)
                        break
                    except ValueError:
                        pass
        
        # If no entity, try parsing from original input
        if pid is None and context and "original_input" in context:
            import re
            # Look for numbers in the input
            numbers = re.findall(r'\b\d+\b', context["original_input"])
            if numbers:
                try:
                    pid = int(numbers[-1])  # Take the last number
                except ValueError:
                    pass
        
        if pid is None:
            return ExecutionResult(
                success=False,
                message="No process ID specified. Usage: kill process 1234"
            )
        
        try:
            import psutil
            process = psutil.Process(pid)
            process_name = process.name()
            
            # Safety check: don't kill critical processes
            critical_processes = ['csrss.exe', 'winlogon.exe', 'services.exe', 'lsass.exe']
            if process_name.lower() in critical_processes:
                return ExecutionResult(
                    success=False,
                    message=f"Safety check: Cannot kill critical system process '{process_name}'"
                )
            
            process.terminate()
            process.wait(timeout=3)
            
            return ExecutionResult(
                success=True,
                message=f"Terminated process '{process_name}' (PID: {pid})"
            )
        except ImportError:
            return ExecutionResult(
                success=False,
                message="psutil not installed. Install with: pip install psutil"
            )
        except psutil.NoSuchProcess:
            return ExecutionResult(
                success=False,
                message=f"No process found with PID: {pid}"
            )
        except psutil.AccessDenied:
            raise PermissionError(f"Access denied to process {pid}")
    
    async def _kill_by_name(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Kill processes by name."""
        # Extract process name from input
        process_name = None
        if context and "original_input" in context:
            input_lower = context["original_input"].lower()
            
            # Handle "kill explorer.exe" format
            import re
            exe_match = re.search(r'kill\s+(\w+)\.exe', input_lower)
            if exe_match:
                process_name = f"{exe_match.group(1)}.exe"
            else:
                # Extract common process names
                common_apps = ['notepad', 'calculator', 'chrome', 'firefox', 'edge', 
                              'explorer', 'discord', 'brave', 'cursor', 'ollama']
                for app in common_apps:
                    if app in input_lower:
                        process_name = f"{app}.exe"
                        break
        
        if not process_name:
            return ExecutionResult(
                success=False,
                message="No process name specified. Usage: kill notepad"
            )
        
        try:
            import psutil
            killed = []
            
            for proc in psutil.process_iter(['name', 'pid']):
                try:
                    if proc.info['name'].lower() == process_name.lower():
                        proc.terminate()
                        killed.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if killed:
                return ExecutionResult(
                    success=True,
                    message=f"Terminated {len(killed)} instance(s) of '{process_name}' (PIDs: {killed})"
                )
            else:
                return ExecutionResult(
                    success=False,
                    message=f"No running processes found named '{process_name}'"
                )
        except ImportError:
            return ExecutionResult(
                success=False,
                message="psutil not installed. Install with: pip install psutil"
            )
    
    async def _kill_most_memory(self) -> ExecutionResult:
        """Kill the process using the most memory (with safety checks)."""
        try:
            import psutil
            
            # Get all processes sorted by memory usage
            processes = []
            for proc in psutil.process_iter(['name', 'pid', 'memory_info', 'username']):
                try:
                    mem = proc.info['memory_info'].rss / (1024 * 1024)  # MB
                    processes.append({
                        'name': proc.info['name'],
                        'pid': proc.info['pid'],
                        'memory_mb': mem,
                        'username': proc.info.get('username', 'N/A')
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if not processes:
                return ExecutionResult(
                    success=False,
                    message="No processes found"
                )
            
            # Sort by memory
            processes.sort(key=lambda p: p['memory_mb'], reverse=True)
            
            # Get top process
            top = processes[0]
            
            # Safety checks
            critical_processes = ['csrss.exe', 'winlogon.exe', 'services.exe', 'lsass.exe', 'system']
            if top['name'].lower() in critical_processes:
                return ExecutionResult(
                    success=False,
                    message=f"Safety check: Top memory process is critical system process '{top['name']}' (cannot kill)"
                )
            
            # Show top 5 for context
            top_5_msg = "Top 5 memory consumers:\n"
            for i, p in enumerate(processes[:5], 1):
                top_5_msg += f"  {i}. {p['name']} - {p['memory_mb']:.1f} MB (PID: {p['pid']})\n"
            
            # Ask for confirmation via message
            return ExecutionResult(
                success=False,
                message=f"{top_5_msg}\nTo kill '{top['name']}' using {top['memory_mb']:.1f} MB, run: kill process {top['pid']}",
                data={"top_process": top, "top_5": processes[:5]}
            )
        except ImportError:
            return ExecutionResult(
                success=False,
                message="psutil not installed. Install with: pip install psutil"
            )
    
    async def _list_processes(self) -> ExecutionResult:
        """List running processes."""
        try:
            import psutil
            
            processes = []
            for proc in psutil.process_iter(['name', 'pid', 'memory_info']):
                try:
                    mem_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    processes.append({
                        'name': proc.info['name'],
                        'pid': proc.info['pid'],
                        'memory_mb': mem_mb
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by memory
            processes.sort(key=lambda p: p['memory_mb'], reverse=True)
            
            # Format as rich table
            from intellishell.utils.display import format_process_table
            formatted_table = format_process_table(processes[:10], "Top 10 Processes by Memory")
            
            return ExecutionResult(
                success=True,
                message=formatted_table,
                data={"processes": processes[:10], "total": len(processes)}
            )
        except ImportError:
            return ExecutionResult(
                success=False,
                message="psutil not installed. Install with: pip install psutil"
            )
    
    async def _check_admin(self) -> ExecutionResult:
        """Check if running as administrator."""
        is_admin = self._is_admin()
        
        if is_admin:
            message = "✓ Running with Administrator privileges"
        else:
            message = "✗ Not running as Administrator\n\nTo run as admin: Right-click terminal → Run as Administrator"
        
        return ExecutionResult(
            success=True,
            message=message,
            data={"is_admin": is_admin}
        )
