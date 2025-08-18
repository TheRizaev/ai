"""
Audio handling utilities for the Medical AI Agent.
"""
import os
import time
import wave
import logging
import tempfile
from threading import Thread
from typing import Optional, Callable

import pyaudio
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

logger = logging.getLogger(__name__)

class AudioRecorder:
    """Class for recording audio from microphone."""
    
    def __init__(self, channels=1, rate=8000, chunk_size=4000):
        """
        Initialize audio recorder.
        
        Args:
            channels: Number of audio channels
            rate: Sample rate in Hz
            chunk_size: Size of audio chunks to process
        """
        self.channels = channels
        self.rate = rate
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.pyaudio = None
        self.stream = None
        self.recording = False
        self.frames = []
    
    def start(self, callback: Optional[Callable] = None):
        """
        Start recording audio.
        
        Args:
            callback: Optional callback function receiving audio data
        """
        if self.recording:
            logger.warning("Already recording")
            return
            
        self.pyaudio = pyaudio.PyAudio()
        self.frames = []
        self.recording = True
        
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Started recording")
            
            def record_thread():
                while self.recording:
                    try:
                        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                        self.frames.append(data)
                        if callback:
                            callback(data)
                    except Exception as e:
                        logger.error(f"Error recording audio: {e}")
                        break
            
            Thread(target=record_thread, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.stop()
            raise
    
    def stop(self) -> Optional[bytes]:
        """
        Stop recording and return the recorded audio data.
        
        Returns:
            Audio data as bytes or None if no recording
        """
        if not self.recording:
            logger.warning("Not recording")
            return None
            
        self.recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
            
        logger.info(f"Stopped recording, captured {len(self.frames)} frames")
        
        if not self.frames:
            return None
            
        return b''.join(self.frames)
    
    def save_to_wav(self, filepath: str) -> bool:
        """
        Save recorded audio to WAV file.
        
        Args:
            filepath: Path to save the WAV file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.frames:
            logger.warning("No audio data to save")
            return False
            
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
                
            logger.info(f"Audio saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False


class AudioPlayer:
    """Class for playing audio."""
    
    @staticmethod
    def play_audio_segment(audio_segment: AudioSegment):
        """
        Play AudioSegment.
        
        Args:
            audio_segment: AudioSegment object to play
        """
        try:
            logger.info(f"Playing audio, duration: {len(audio_segment)/1000:.2f}s")
            play(audio_segment)
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
    
    @staticmethod
    def play_wav_file(filepath: str):
        """
        Play WAV file.
        
        Args:
            filepath: Path to WAV file
        """
        try:
            audio = AudioSegment.from_wav(filepath)
            logger.info(f"Playing {filepath}, duration: {len(audio)/1000:.2f}s")
            play(audio)
        except Exception as e:
            logger.error(f"Failed to play WAV file: {e}")
    
    @staticmethod
    def play_bytes(audio_data: bytes, channels: int = 1, rate: int = 8000):
        """
        Play audio from bytes data.
        
        Args:
            audio_data: Audio data as bytes
            channels: Number of audio channels
            rate: Sample rate in Hz
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                
            # Write WAV file
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(rate)
                wf.writeframes(audio_data)
                
            # Play the file
            AudioPlayer.play_wav_file(tmp_path)
            
            # Clean up
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Failed to play audio bytes: {e}")


class VoiceActivityDetector:
    """Simple voice activity detector to detect speech."""
    
    def __init__(self, threshold=500, min_silence_duration=1.0, sample_rate=8000):
        """
        Initialize voice activity detector.
        
        Args:
            threshold: Energy threshold to consider as speech
            min_silence_duration: Minimum silence duration in seconds to end detection
            sample_rate: Audio sample rate in Hz
        """
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.silent_frames = 0
        self.speaking = False
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Check if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data as bytes
            
        Returns:
            True if speech detected, False otherwise
        """
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Calculate energy
        energy = np.sum(np.abs(audio_data)) / len(audio_data)
        
        # Check if energy is above threshold
        is_above_threshold = energy > self.threshold
        
        # Update state based on energy
        if is_above_threshold:
            self.speaking = True
            self.silent_frames = 0
        else:
            # Count silent frames
            chunk_duration = len(audio_data) / self.sample_rate
            self.silent_frames += chunk_duration
            
            # Check if silence has been long enough
            if self.silent_frames > self.min_silence_duration:
                self.speaking = False
                
        return self.speaking
    
    def reset(self):
        """Reset the detector state."""
        self.silent_frames = 0
        self.speaking = False