#!/usr/bin/env python3
"""
Simple Telnet Client with AI Toggle

A minimal telnet client that passes raw socket data directly to the terminal.
- No parsing or filtering of telnet/ANSI sequences
- Terminal handles everything natively
- Ctrl+A toggles AI mode
"""

import socket
import threading
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import deque
import re
from datetime import datetime

# Platform-specific imports
if sys.platform == "win32":
    import msvcrt
    import ctypes
    from ctypes import wintypes
else:
    import termios
    import tty
    import select


class SessionRecorder:
    """Records agent sessions for training data collection."""
    
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.session_dir / f"session_{timestamp}.jsonl"
        self.experience_file = self.session_dir / f"experience_{timestamp}.jsonl"
        
        # Session state
        self.session_start = time.time()
        self.total_commands = 0
        self.ai_commands = 0
        self.session_data = []
        self.experience_buffer = []
        
    def record_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """Record an interaction for training data."""
        record = {
            'timestamp': time.time(),
            'session_time': time.time() - self.session_start,
            'type': interaction_type,
            'data': data
        }
        
        # Write to file immediately (streaming)
        with open(self.session_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    
    def record_ai_decision(self, context: str, command: str, outcome: str = None):
        """Record AI decision-making for reinforcement learning."""
        experience = {
            'timestamp': time.time(),
            'context': context,
            'action': command,
            'outcome': outcome,  # Will be filled in later when we see results
            'session_id': self.session_file.stem
        }
        
        self.experience_buffer.append(experience)
        
        # Write experience data
        with open(self.experience_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(experience) + '\n')
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            'session_duration': time.time() - self.session_start,
            'total_commands': self.total_commands,
            'ai_commands': self.ai_commands,
            'session_file': str(self.session_file),
            'experience_file': str(self.experience_file)
        }


class TelnetClient:
    """Simple telnet client with AI assist and enhanced data collection."""
    
    def __init__(self, config_path: str):
        """Initialize the telnet client."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up logging
        self._setup_logging()
        
        # Connection settings
        self.host = self.config['connection']['mud_server']['host']
        self.port = self.config['connection']['mud_server']['port']
        
        # State
        self.socket = None
        self.connected = False
        self.running = False
        self.ai_mode = False
        self.ai_thread = None
        
        # Terminal state
        self.original_terminal_settings = None
        
        # Context for AI
        self.context_buffer = deque(maxlen=1000)
        
        # Enhanced data collection
        self.data_collection_enabled = self.config.get('data_collection', {}).get('enabled', True)
        if self.data_collection_enabled:
            data_dir = Path(self.config.get('data_collection', {}).get('directory', 'data/agent_sessions'))
            self.session_recorder = SessionRecorder(data_dir)
            print(f"📊 Session recording enabled: {self.session_recorder.session_file}")
        else:
            self.session_recorder = None
        
        # Trained model integration
        self.ai_model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # AI state
        self.current_goal = 'explore_and_fight'
        self.pending_commands = 0  # Track commands waiting for response
        self.max_pending_commands = 5
        
        self.logger.info("Simple Telnet Client initialized with AI intelligence")
        
        # Try to load the trained model
        self._load_trained_model()
        
        # Show model status
        if self.model_loaded:
            print("✓ Trained GPT-2 model loaded successfully")
        else:
            print("⚠ Using rule-based AI only (trained model not available)")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")
    
    def _setup_logging(self):
        """Set up logging for the agent."""
        log_level = getattr(logging, self.config['monitoring']['logging']['level'])
        
        # Create logger
        self.logger = logging.getLogger('TelnetClient')
        self.logger.setLevel(log_level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # File handler
            file_handler = logging.FileHandler('agent.log')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler (only for errors and warnings to avoid cluttering terminal)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def connect(self) -> bool:
        """Connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info(f"Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the server."""
        self.connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
        self.logger.info("Disconnected")
    
    def _setup_terminal(self):
        """Set up terminal for raw mode."""
        if sys.platform == "win32":
            self._setup_terminal_windows()
        else:
            self._setup_terminal_unix()
    
    def _restore_terminal(self):
        """Restore terminal to original state."""
        if sys.platform == "win32":
            self._restore_terminal_windows()
        else:
            self._restore_terminal_unix()
    
    def _setup_terminal_windows(self):
        """Set up Windows terminal for raw mode."""
        # Get handles
        self.stdin_handle = ctypes.windll.kernel32.GetStdHandle(-10)
        self.stdout_handle = ctypes.windll.kernel32.GetStdHandle(-11)
        
        # Save original modes
        self.original_stdin_mode = wintypes.DWORD()
        self.original_stdout_mode = wintypes.DWORD()
        ctypes.windll.kernel32.GetConsoleMode(self.stdin_handle, ctypes.byref(self.original_stdin_mode))
        ctypes.windll.kernel32.GetConsoleMode(self.stdout_handle, ctypes.byref(self.original_stdout_mode))
        
        # Set raw mode
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_stdin_mode = self.original_stdin_mode.value & ~(0x0002 | 0x0004 | 0x0001)  # Disable line input, echo, processed input
        new_stdout_mode = self.original_stdout_mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
        
        ctypes.windll.kernel32.SetConsoleMode(self.stdin_handle, new_stdin_mode)
        ctypes.windll.kernel32.SetConsoleMode(self.stdout_handle, new_stdout_mode)
    
    def _restore_terminal_windows(self):
        """Restore Windows terminal."""
        if hasattr(self, 'original_stdin_mode'):
            ctypes.windll.kernel32.SetConsoleMode(self.stdin_handle, self.original_stdin_mode.value)
            ctypes.windll.kernel32.SetConsoleMode(self.stdout_handle, self.original_stdout_mode.value)
    
    def _setup_terminal_unix(self):
        """Set up Unix terminal for raw mode."""
        self.original_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
    
    def _restore_terminal_unix(self):
        """Restore Unix terminal."""
        if self.original_terminal_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_terminal_settings)
    
    def start(self):
        """Start the telnet client."""
        if not self.connect():
            return False
        
        self.running = True
        
        try:
            # Set up terminal
            self._setup_terminal()
            
            # Show header
            sys.stdout.write('\033[2J\033[H')  # Clear screen
            sys.stdout.write("=" * 60 + "\r\n")
            sys.stdout.write("GREY LORD TELNET CLIENT\r\n")
            sys.stdout.write("=" * 60 + "\r\n")
            sys.stdout.write(f"Connected to {self.host}:{self.port}\r\n")
            sys.stdout.write("Ctrl+T: Toggle AI | Ctrl+C: Quit\r\n")
            sys.stdout.write("AI Mode: OFF\r\n")
            sys.stdout.write("=" * 60 + "\r\n")
            sys.stdout.flush()
            
            # Start threads
            self._start_socket_to_stdout_thread()
            self._start_stdin_to_socket_thread()
            
            # Main loop
            while self.running and self.connected:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the client."""
        self.running = False
        if self.ai_mode:
            self._disable_ai_mode()
        self._restore_terminal()
        self.disconnect()
        sys.stdout.write("\r\nDisconnected.\r\n")
        sys.stdout.flush()
    
    def _start_socket_to_stdout_thread(self):
        """Thread to copy data from socket to stdout."""
        def socket_to_stdout():
            while self.running and self.connected:
                try:
                    self.socket.settimeout(0.1)
                    data = self.socket.recv(4096)
                    if not data:
                        self.connected = False
                        break
                    
                    # Store for AI context (decode for text analysis)
                    try:
                        data_str = data.decode('utf-8', errors='replace')
                        self.context_buffer.append(data_str)
                        
                        # Record server data for training
                        self._record_interaction('server_data', 
                                               raw_data=data.hex(),
                                               text_data=data_str,
                                               length=len(data))
                        
                        # Track server responses to manage command rate limiting
                        if self.pending_commands > 0:
                            # Simple heuristic: if we get a prompt-like response, a command was processed
                            if any(indicator in data_str.lower() for indicator in ['[hp=', 'obvious exits', '>', ']:']):
                                self.pending_commands = max(0, self.pending_commands - 1)
                        
                    except Exception as e:
                        self.logger.error(f"Context storage error: {e}")
                        # Continue anyway - this is not critical
                    
                    # Send raw data directly to stdout - let terminal handle everything
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Socket read error: {e}")
                    break
        
        thread = threading.Thread(target=socket_to_stdout, daemon=True)
        thread.start()
    
    def _start_stdin_to_socket_thread(self):
        """Thread to copy data from stdin to socket."""
        def stdin_to_socket():
            while self.running and self.connected:
                try:
                    if sys.platform == "win32":
                        if msvcrt.kbhit():
                            char = msvcrt.getch()
                            
                            # Debug: show what we received
                            # sys.stderr.write(f"DEBUG: Received byte: {char} (0x{char[0]:02x})\n")
                            # sys.stderr.flush()
                            
                            # Check for control characters
                            if char == b'\x14':  # Ctrl+T (Toggle AI)
                                self._toggle_ai_mode()
                                continue
                            elif char == b'\x03':  # Ctrl+C
                                self.running = False
                                break
                            elif len(char) == 1 and char[0] == 20:  # Alternative check for Ctrl+T
                                self._toggle_ai_mode()
                                continue
                            
                            # If not in AI mode, send to socket
                            if not self.ai_mode:
                                self.socket.send(char)
                    else:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            char = sys.stdin.buffer.read(1)
                            
                            # Check for control characters
                            if char == b'\x14':  # Ctrl+T (Toggle AI)
                                self._toggle_ai_mode()
                                continue
                            elif char == b'\x03':  # Ctrl+C
                                self.running = False
                                break
                            
                            # If not in AI mode, send to socket
                            if not self.ai_mode:
                                self.socket.send(char)
                    
                    time.sleep(0.01)
                    
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Stdin read error: {e}")
                    break
        
        thread = threading.Thread(target=stdin_to_socket, daemon=True)
        thread.start()
    
    def _toggle_ai_mode(self):
        """Toggle AI mode."""
        if self.ai_mode:
            self._disable_ai_mode()
        else:
            self._enable_ai_mode()
    
    def _enable_ai_mode(self):
        """Enable AI mode."""
        self.ai_mode = True
        
        # Show AI mode message
        sys.stdout.write("\r\n" + "=" * 50 + "\r\n")
        sys.stdout.write("AI MODE: ENABLED - Using trained model\r\n")
        sys.stdout.write("Press Ctrl+T to disable\r\n")
        sys.stdout.write("=" * 50 + "\r\n")
        sys.stdout.flush()
        
        # Start AI thread
        self.ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        self.ai_thread.start()
        
        self.logger.info("AI mode enabled")
    
    def _disable_ai_mode(self):
        """Disable AI mode."""
        self.ai_mode = False
        
        # Show manual mode message
        sys.stdout.write("\r\n" + "=" * 50 + "\r\n")
        sys.stdout.write("AI MODE: DISABLED - Manual control\r\n")
        sys.stdout.write("Press Ctrl+T to enable\r\n")
        sys.stdout.write("=" * 50 + "\r\n")
        sys.stdout.flush()
        
        self.logger.info("AI mode disabled")
    
    def _ai_loop(self):
        """AI loop - send commands automatically using trained model."""
        while self.ai_mode and self.running and self.connected:
            try:
                time.sleep(1.0)  # Wait 1 second between commands
                
                if self.ai_mode and self.pending_commands < self.max_pending_commands:
                    # Get current context for decision recording
                    current_context = self._build_model_context()
                    command = self._get_model_decision()
                    
                    if command and self._is_safe_command(command):
                        # Record AI decision for training data
                        if self.session_recorder:
                            self.session_recorder.record_ai_decision(current_context, command)
                            self.session_recorder.ai_commands += 1
                        
                        # Record command interaction
                        self._record_interaction('ai_command',
                                               command=command,
                                               context=current_context[:500],  # Truncate for storage
                                               pending_commands=self.pending_commands)
                        
                        # Show what AI is doing
                        sys.stdout.write(f"[AI] {command}\r\n")
                        sys.stdout.flush()
                        
                        # Send command and track it
                        self.socket.send(f"{command}\r\n".encode())
                        self.pending_commands += 1
                    else:
                        # AI doesn't have a good idea right now - that's fine, stay quiet
                        if command:
                            self.logger.debug(f"AI generated invalid/unsafe command: {repr(command)}")
                            # Record rejected commands too
                            self._record_interaction('ai_command_rejected',
                                                   command=command,
                                                   reason='unsafe' if not self._is_safe_command(command) else 'invalid')
                    
            except Exception as e:
                self.logger.error(f"AI loop error: {e}")
                break
    
    def _get_model_decision(self) -> str:
        """Get a decision from the trained model."""
        try:
            # Check if model is actually loaded
            if not self.model_loaded or not self.ai_model or not self.tokenizer:
                self.logger.debug("Model not loaded, using fallback")
                return None
            
            # Build context from recent game interactions
            context = self._build_model_context()
            if not context:
                self.logger.debug("No context available")
                return None
            
            # Generate with the model
            import torch
            
            # Tokenize context
            try:
                input_tokens = self.tokenizer.encode(context)
                if not input_tokens:
                    self.logger.debug("Failed to tokenize context")
                    return None
                
                input_ids = torch.tensor([input_tokens])
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
            except Exception as e:
                self.logger.error(f"Tokenization error: {e}")
                return None
            
            # Generate response with better parameters
            with torch.no_grad():
                # Ensure pad_token_id and eos_token_id are not None
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
                
                # Use fallback values if they are None
                if pad_token_id is None:
                    pad_token_id = 0
                if eos_token_id is None:
                    eos_token_id = 1
                
                outputs = self.ai_model.generate(
                    input_ids,
                    max_new_tokens=15,  # Shorter to avoid gibberish
                    min_new_tokens=1,   # Ensure we get something
                    temperature=0.8,    # Less randomness
                    do_sample=True,
                    top_p=0.9,          # Nucleus sampling
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    no_repeat_ngram_size=3,  # Avoid repetition
                    repetition_penalty=1.1   # Slight penalty for repetition
                )
            
            # Decode the generated response
            generated_ids = outputs[0][input_ids.shape[1]:]
            if len(generated_ids) == 0:
                self.logger.debug("Model generated no tokens")
                return None
                
            response = self.tokenizer.decode(generated_ids.cpu().tolist())
            if not response:
                self.logger.debug("Failed to decode model response")
                return None
            
            # Extract clean command
            command = self._extract_command_from_response(response)
            if command:
                self.logger.debug(f"Model generated: {repr(response)} -> {repr(command)}")
            else:
                self.logger.debug(f"Failed to extract command from: {repr(response)}")
            
            return command
            
        except Exception as e:
            self.logger.error(f"Model generation error: {e}")
            return None
    
    def _build_model_context(self) -> str:
        """Build context for the model based on recent game state."""
        try:
            # Get recent server output (last 3 messages to keep it focused)
            recent_context = list(self.context_buffer)[-3:]
            if not recent_context:
                return None
            
            # Clean the context - remove ANSI sequences and control characters
            cleaned_context = []
            for msg in recent_context:
                if msg is not None:  # Check for None messages
                    cleaned = self._clean_text_for_model(msg)
                    if cleaned:
                        cleaned_context.append(cleaned)
            
            if not cleaned_context:
                return None
            
            context_text = ''.join(cleaned_context)
            
            # Format like the training data - simple format
            context = f"{context_text}<|client|>"
            
            # Keep context reasonable length (model was trained on specific lengths)
            if len(context) > 300:
                context = context[-300:]
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context building error: {e}")
            return None
    
    def _clean_text_for_model(self, text: str) -> str:
        """Clean text for model input by removing ANSI sequences and control chars."""
        if not text:
            return ""
            
        import re
        
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        
        # Remove other control characters except newlines
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\r\n')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _get_state_summary(self) -> str:
        """Get a simple state summary for model context."""
        # Just return empty for now - let the model work with raw context
        return ""
    
    def _extract_command_from_response(self, response: str) -> str:
        """Extract a clean command from model response."""
        # Stop at common delimiters
        for delimiter in ['<|server|>', '<|endoftext|>', '\n', '<|client|>']:
            if delimiter in response:
                response = response.split(delimiter)[0]
        
        # Clean up the command
        command = response.strip()
        
        # Remove common prefixes
        prefixes = ['> ', 'Command: ', 'Player: ']
        for prefix in prefixes:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
        
        # Handle byte tokens - convert back to actual bytes and then to text
        if 'byte_' in command:
            command = self._convert_byte_tokens_to_text(command)
        
        # Basic validation
        if len(command) > 50:  # Commands shouldn't be too long
            command = command[:50]
        
        return command if command else None
    
    def _convert_byte_tokens_to_text(self, text: str) -> str:
        """Convert byte tokens back to readable text, filtering out telnet protocol."""
        import re
        
        # Find all byte tokens
        byte_pattern = r'byte_(\d+)'
        byte_matches = re.findall(byte_pattern, text)
        
        if not byte_matches:
            return text
        
        # Convert byte tokens to actual bytes
        try:
            byte_values = [int(match) for match in byte_matches]
            byte_data = bytes(byte_values)
            
            # Filter out telnet protocol bytes (255 = IAC, 253 = DO, 251 = WILL, etc.)
            filtered_bytes = []
            i = 0
            while i < len(byte_data):
                byte_val = byte_data[i]
                
                # Skip telnet protocol sequences
                if byte_val == 255:  # IAC (Interpret As Command)
                    # Skip the next 1-2 bytes that are part of telnet protocol
                    if i + 1 < len(byte_data):
                        next_byte = byte_data[i + 1]
                        if next_byte in [251, 252, 253, 254]:  # WILL, WONT, DO, DONT
                            i += 3  # Skip IAC + command + option
                            continue
                        else:
                            i += 2  # Skip IAC + command
                            continue
                    else:
                        i += 1
                        continue
                
                # Skip other control characters but keep printable ones
                if byte_val >= 32 and byte_val <= 126:  # Printable ASCII
                    filtered_bytes.append(byte_val)
                elif byte_val in [10, 13]:  # Keep newlines
                    filtered_bytes.append(byte_val)
                
                i += 1
            
            # Convert back to text
            if filtered_bytes:
                clean_text = bytes(filtered_bytes).decode('utf-8', errors='ignore').strip()
                # Only return if it looks like a valid command
                if clean_text and len(clean_text) > 0 and not all(c in '\r\n\t ' for c in clean_text):
                    return clean_text
            
        except (ValueError, UnicodeDecodeError):
            pass
        
        # If conversion fails or results in telnet protocol, return None
        return None
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if a command is safe to execute."""
        if not command:
            return False
        
        command = command.strip()
        
        # Check if it's still byte tokens (conversion failed)
        if 'byte_' in command:
            return False
        
        # Check for ANSI escape sequences
        if '\x1b' in command or '[0m' in command:
            return False
        
        # Check minimum and maximum length
        if len(command) < 1 or len(command) > 50:
            return False
        
        # Must contain at least one letter (avoid pure gibberish)
        if not any(c.isalpha() for c in command):
            return False
        
        # Block obviously dangerous commands
        command_lower = command.lower()
        dangerous_commands = ['quit', 'exit', 'shutdown', 'reboot', 'delete', 'format']
        if any(dangerous in command_lower for dangerous in dangerous_commands):
            return False
        
        # Block commands with suspicious characters that might break things
        if any(char in command for char in ['<', '>', '|', '&', ';', '`']):
            return False
        
        return True
    
    def _load_trained_model(self):
        """Load the trained GPT-2 model and tokenizer."""
        try:
            model_path = self.config.get('model_settings', {}).get('model_path')
            tokenizer_path = self.config.get('model_settings', {}).get('tokenizer_path')
            
            if not model_path or not tokenizer_path:
                self.logger.warning("Model paths not configured, using rule-based AI only")
                return
            
            # Import here to avoid dependency issues
            from transformers import AutoModelForCausalLM
            import torch
            
            # Load model
            if Path(model_path).exists():
                self.logger.info(f"Loading trained model from {model_path}")
                self.ai_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.ai_model = self.ai_model.cuda()
                    self.logger.info("Model loaded on GPU")
                else:
                    self.logger.info("Model loaded on CPU")
                
                # Load tokenizer
                self.tokenizer = self._load_custom_tokenizer(tokenizer_path)
                
                if self.tokenizer:
                    self.model_loaded = True
                    self.logger.info("Trained model and tokenizer loaded successfully")
                else:
                    self.logger.error("Failed to load tokenizer")
            else:
                self.logger.warning(f"Model path not found: {model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}")
            self.logger.info("Falling back to rule-based AI")
    
    def _load_custom_tokenizer(self, tokenizer_path: str):
        """Load the custom Grey Lord tokenizer."""
        try:
            vocab_file = Path(tokenizer_path) / "vocab_to_int.json"
            
            if vocab_file.exists():
                with open(vocab_file, 'r') as f:
                    vocab_to_int = json.load(f)
                
                class GreyLordTokenizer:
                    def __init__(self, vocab_to_int):
                        self.vocab_to_int = vocab_to_int
                        self.int_to_vocab = {v: k for k, v in vocab_to_int.items()}
                        self.pad_token_id = vocab_to_int.get('<|pad|>', 0)
                        self.eos_token_id = vocab_to_int.get('<|endoftext|>', 1)
                        self.unk_token_id = vocab_to_int.get('<|unk|>', 1)
                    
                    def encode(self, text: str) -> list:
                        """Encode text to token IDs."""
                        if not text:
                            return []
                        tokens = []
                        for char in text:
                            tokens.append(self.vocab_to_int.get(char, self.unk_token_id))
                        return tokens
                    
                    def decode(self, token_ids: list) -> str:
                        """Decode token IDs to text."""
                        if not token_ids:
                            return ""
                        chars = []
                        for tid in token_ids:
                            char = self.int_to_vocab.get(tid, '<|unk|>')
                            if char not in ['<|pad|>', '<|endoftext|>', '<|unk|>']:
                                chars.append(char)
                        return ''.join(chars)
                
                return GreyLordTokenizer(vocab_to_int)
            else:
                self.logger.error(f"Tokenizer vocab file not found: {vocab_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load custom tokenizer: {e}")
            return None
    
    def _record_interaction(self, interaction_type: str, **kwargs):
        """Record interaction if data collection is enabled."""
        if self.session_recorder:
            self.session_recorder.record_interaction(interaction_type, kwargs)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Telnet Client with AI')
    parser.add_argument('--config', default='agent_config.json', help='Configuration file')
    args = parser.parse_args()
    
    try:
        client = TelnetClient(args.config)
        client.start()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 