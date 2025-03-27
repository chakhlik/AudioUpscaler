
import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from typing import Literal

class AudioInterpolator:
    def __init__(self, method: Literal['cubic', 'akima', 'pchip', 'repeat'] = 'cubic'):
        # Store last 9 output samples (176.4kHz)
        self.previous_samples = np.zeros((9, 2), dtype=np.float64)  # 9 samples, 2 channels
        self.chunk_end_sample = np.zeros((1, 2), dtype=np.float64)  # last sample from previos chunk
        self.channels = 2
        self.method = method

    def reset(self):
        """Reset the interpolator state."""
        self.previous_samples = np.zeros((9, 2), dtype=np.float64)

    def interpolate_chunk(self, input_chunk: np.ndarray) -> np.ndarray:
        """
        Interpolate between samples to increase sample rate by 4x.
        Uses various interpolation methods.

        Each chunk must be interpolated sequentially:
        - Uses points 1-9 from previous output samples (176.4kHz)
        - Points 13,17 from input stream (44.1kHz)
        - Generates points 10,11,12 through interpolation
        - Outputs points 10,11,12,13 to the stream
        
        Processes both left and right channels in parallel using NumPy vectorization.
        """
        if len(input_chunk) == 0:
            return np.array([], dtype=np.float64)

        # Reshape input to separate channels
        input_samples = np.concatenate([self.chunk_end_sample, input_chunk.reshape(-1, self.channels)])

        # Calculate output size: for each input sample (except the last one)
        # we generate 4 output samples (3 interpolated + 1 original)
        output_len = (len(input_samples) - 1) * 4
        output = np.zeros((output_len, self.channels), dtype=np.float64)
        
        # Create common x coordinates once for both channels
        x_points = np.concatenate([
            np.arange(1, 10),  # Points 1-9 from previous samples
            np.array([13, 17])  # Points from input stream
        ])
        
        # Create interpolation points for new samples (10,11,12)
        x_new = np.array([10, 11, 12])
        
        # Process all samples in each channel at once
        for i in range(len(input_samples) - 1):
            output_idx = i * 4
            
            if self.method == 'repeat':
                # Simply repeat the first input sample 4 times for both channels
                output[output_idx:output_idx+4, :] = input_samples[i, :].reshape(1, 2)
            else:
                # Process both channels in parallel
                for channel in range(self.channels):
                    # Combine previous samples with current pair of input samples
                    y_points = np.concatenate([
                        self.previous_samples[:, channel],  # Previous 9 samples
                        input_samples[i:i+2, channel]       # Current 2 samples
                    ])
                    
                    # Perform interpolation based on selected method
                    if self.method == 'cubic':
                        interpolator = CubicSpline(x_points, y_points)
                    elif self.method == 'akima':
                        interpolator = Akima1DInterpolator(x_points, y_points)
                    elif self.method == 'pchip':
                        interpolator = PchipInterpolator(x_points, y_points)
                    else:
                        raise ValueError(f"Unknown interpolation method: {self.method}")
                    
                    # Generate interpolated samples (10,11,12)
                    output[output_idx:output_idx+3, channel] = interpolator(x_new)
                    
                    # Add the first input sample (13) as the fourth output sample
                    output[output_idx+3, channel] = input_samples[i, channel]

            # Update previous samples for both channels

            if self.method == 'repeat':
                # For repeat method, use the same value for all 4 points
                self.previous_samples = np.roll(self.previous_samples, -4, axis=0)
                self.previous_samples[-4:, :] = np.tile(input_samples[i, :], (4, 1))
            else:
                # Update previous samples: shift window by 4 and add the 4 new points
                self.previous_samples = np.roll(self.previous_samples, -4, axis=0)
                # Points 10,11,12,13 for both channels
                self.previous_samples[-4:-1, :] = output[output_idx:output_idx+3, :]
                self.previous_samples[-1, :] = input_samples[i, :]
        
        # Store the last sample from this chunk for both channels
        self.chunk_end_sample[0, :] = input_samples[-1, :]
        
        return output.flatten()
