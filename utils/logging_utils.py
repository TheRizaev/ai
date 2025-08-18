"""
Logging utilities for the Medical AI Agent.
"""
import os
import logging
import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=None, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (INFO, DEBUG, etc.)
        log_file: Path to log file
    """
    # Convert string level to logging level
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        log_level = log_level or logging.INFO
        
    # Create logs directory if not exists
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    else:
        # Default log file in logs directory
        log_dir = os.path.join(Path(__file__).resolve().parent.parent, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'medcenter_ai_{timestamp}.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%H:%M:%S'
    )
    
    # Configure file handler
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized (level: {logging.getLevelName(log_level)}, file: {log_file})")
    return root_logger


class ConversationLogger:
    """Logger for conversation history."""
    
    def __init__(self, log_dir=None):
        """
        Initialize conversation logger.
        
        Args:
            log_dir: Directory to store conversation logs
        """
        if not log_dir:
            log_dir = os.path.join(Path(__file__).resolve().parent.parent, 'logs', 'conversations')
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        
        # Create a new conversation file with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'conversation_{timestamp}.txt')
        
        # Create the file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Conversation started at {datetime.datetime.now()}\n")
            f.write("-" * 80 + "\n\n")
    
    def log_user_input(self, text):
        """Log user input."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"USER ({datetime.datetime.now().strftime('%H:%M:%S')}): {text}\n\n")
    
    def log_system_event(self, event_type, details=None):
        """Log system event."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"SYSTEM ({datetime.datetime.now().strftime('%H:%M:%S')}): {event_type}")
            if details:
                f.write(f" - {details}")
            f.write("\n\n")
    
    def log_agent_response(self, text):
        """Log agent response."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"AGENT ({datetime.datetime.now().strftime('%H:%M:%S')}): {text}\n\n")
    
    def get_conversation_history(self, max_entries=10):
        """
        Get recent conversation history.
        
        Args:
            max_entries: Maximum number of entries to return
            
        Returns:
            List of conversation entries
        """
        history = []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_entry = []
        current_speaker = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("-" * 10):
                continue
                
            if line.startswith("USER") or line.startswith("AGENT") or line.startswith("SYSTEM"):
                # If we have a previous entry, add it to history
                if current_entry and current_speaker:
                    history.append((current_speaker, " ".join(current_entry)))
                    current_entry = []
                
                # Extract the speaker from the current line
                if line.startswith("USER"):
                    current_speaker = "USER"
                    # Remove the USER prefix
                    content = line[line.find(":")+1:].strip()
                    if content:
                        current_entry.append(content)
                elif line.startswith("AGENT"):
                    current_speaker = "AGENT"
                    # Remove the AGENT prefix
                    content = line[line.find(":")+1:].strip()
                    if content:
                        current_entry.append(content)
                elif line.startswith("SYSTEM"):
                    current_speaker = "SYSTEM"
                    # Remove the SYSTEM prefix
                    content = line[line.find(":")+1:].strip()
                    if content:
                        current_entry.append(content)
            else:
                # Continuation of the current entry
                current_entry.append(line)
        
        # Add the last entry if it exists
        if current_entry and current_speaker:
            history.append((current_speaker, " ".join(current_entry)))
        
        # Return the most recent entries, limited by max_entries
        return history[-max_entries:]