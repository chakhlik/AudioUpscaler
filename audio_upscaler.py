from wav_handler import WavHandler
from interpolator import AudioInterpolator
from typing import Tuple, Literal
import os
import sys
import time

class AudioUpscaler:
    def __init__(self, interpolation_method: Literal['cubic', 'akima', 'pchip', 'repeat'] = 'cubic'):
        self.wav_handler = WavHandler()
        self.interpolator = AudioInterpolator(method=interpolation_method)
        self.chunk_size = 1024  # Process 1024 frames at a time
        self.start_time = None

    def process_file(self, input_filename: str, output_filename: str) -> Tuple[bool, str]:
        """Process the input WAV file and create an upscaled output file."""

        # Validate input file is 16-bit
        valid, message = self.wav_handler.validate_wav_file(input_filename, expected_width=2)
        if not valid:
            return False, message

        try:
            # Setup input and output files
            input_wav, output_wav = self.wav_handler.setup_output_wav(input_filename, output_filename)

            total_frames = self.wav_handler.get_total_frames(input_wav)
            processed_frames = 0

            # Reset interpolator state and start time
            self.interpolator.reset()
            self.start_time = time.time()

            while True:
                # Read chunk of input data
                input_chunk = self.wav_handler.read_wav_chunk(input_wav, self.chunk_size)
                if len(input_chunk) == 0:
                    break

                # Interpolate the chunk
                output_chunk = self.interpolator.interpolate_chunk(input_chunk)

                # Write the interpolated chunk
                self.wav_handler.write_wav_chunk(output_wav, output_chunk)

                # Update progress
                processed_frames += len(input_chunk) // 2  # Divide by 2 for stereo
                self._update_progress(processed_frames, total_frames)

            input_wav.close()
            output_wav.close()

            # Verify output file is 24-bit/176.4kHz
            valid, verify_message = self.wav_handler.validate_wav_file(
                output_filename, 
                expected_rate=176400,
                expected_width=3  # Verify 24-bit output
            )
            if not valid:
                os.remove(output_filename)
                return False, f"Output file verification failed: {verify_message}"

            return True, "File processed successfully"

        except Exception as e:
            return False, f"Error processing file: {str(e)}"

    def _update_progress(self, current: int, total: int):
        """Update the progress bar with elapsed time."""
        progress = (current / total) * 100
        elapsed = time.time() - self.start_time

        # Calculate estimated time remaining
        if progress > 0:
            total_time = elapsed * 100 / progress
            remaining = total_time - elapsed
        else:
            remaining = 0

        # Format the progress bar
        bar = "=" * int(progress // 2)
        bar = f"[{bar:<50}]"

        # Format elapsed and remaining time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining))

        # Print progress
        sys.stdout.write(f'\rProgress: {bar} {progress:.1f}% | Time: {elapsed_str} | ETA: {remaining_str}')
        sys.stdout.flush()