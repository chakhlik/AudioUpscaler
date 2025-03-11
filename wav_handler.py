import wave
import numpy as np
from typing import Tuple

class WavHandler:
    def __init__(self):
        self.channels = 2  # Stereo
        self.sampwidth = 3  # 24-bit

    def validate_wav_file(self, filename: str, expected_rate: int = 44100, expected_width: int = 2) -> Tuple[bool, str]:
        """Validate if the WAV file meets our requirements."""
        try:
            with wave.open(filename, 'rb') as wav_file:
                if wav_file.getnchannels() != 2:
                    return False, "File must be stereo (2 channels)"
                if expected_width and wav_file.getsampwidth() != expected_width:
                    return False, f"File must be {expected_width * 8}bit"
                if wav_file.getframerate() != expected_rate:
                    return False, f"File must be {expected_rate/1000:.1f}kHz"

                # Print file information for verification
                print(f"\nWAV File Info:")
                print(f"Channels: {wav_file.getnchannels()}")
                print(f"Sample width: {wav_file.getsampwidth() * 8}bit")
                print(f"Sample rate: {wav_file.getframerate()/1000:.1f}kHz")
                return True, "Valid WAV file"
        except Exception as e:
            return False, f"Error validating file: {str(e)}"

    def read_wav_chunk(self, wav_file: wave.Wave_read, chunk_size: int) -> np.ndarray:
        """Read a chunk of frames from the WAV file."""
        frames = wav_file.readframes(chunk_size)
        if not frames:
            return np.array([], dtype=np.float64)

        # Convert to float64 without scaling, preserving raw integer values
        return np.frombuffer(frames, dtype=np.int16).astype(np.float64)*256

    def write_wav_chunk(self, wav_file: wave.Wave_write, chunk: np.ndarray):
        """Write a chunk of frames to the WAV file."""
        # Convert float64 to 24-bit range directly
        clipped = np.clip(chunk, -8388608, 8388607).astype(np.int32)

        # Convert to integer format
        bytes_view = clipped[:, None].view(np.uint8)  # (N, 4) - 4 байта на число

        # Convert to bytes, keeping only the 3 most significant bytes (24-bit)
        #bytes_data = int_data.astype('>i4').to_bytes()[1:]
        #bytes_data = clipped[:, None].view(np.uint8)[:, 0:3]
        bytes_data = bytes_view[:, 0:3].tobytes()

        wav_file.writeframes(bytes_data)

    def setup_output_wav(self, input_filename: str, output_filename: str) -> Tuple[wave.Wave_read, wave.Wave_write]:
        """Setup input and output WAV files."""
        input_wav = wave.open(input_filename, 'rb')
        output_wav = wave.open(output_filename, 'wb')

        # Configure output WAV file
        output_wav.setnchannels(self.channels)
        output_wav.setsampwidth(self.sampwidth)  # 24-bit
        output_wav.setframerate(input_wav.getframerate()*4)  # 176.4kHz

        return input_wav, output_wav

    def get_total_frames(self, wav_file: wave.Wave_read) -> int:
        """Get total number of frames in the WAV file."""
        return wav_file.getnframes()