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
        """
        if len(input_chunk) == 0:
            return np.array([], dtype=np.float64)

        # Reshape input to separate channels
        input_samples = np.concatenate([self.chunk_end_sample, input_chunk.reshape(-1, self.channels)])

        # Calculate output size: for each input sample (except the last one)
        # we generate 4 output samples (3 interpolated + 1 original)
        output_len = (len(input_samples) - 1) * 4
        output = np.zeros((output_len, self.channels), dtype=np.float64)

        # Process each channel separately
        for channel in range(self.channels):
            output_idx = 0

            # Process pairs of input samples sequentially
            for i in range(len(input_samples) - 1):
                if self.method == 'repeat':
                    # Simply repeat the first input sample 4 times
                    output[output_idx:output_idx+4, channel] = input_samples[i, channel]
                else:
                    # Create x coordinates for all points:
                    # - 9 previous points (1-9)
                    # - 2 input points (13, 17)
                    x_points = np.concatenate([
                        np.arange(1, 10),  # Points 1-9 from previous samples
                        np.array([13, 17])  # Points from input stream
                    ])

                    # Combine previous samples with current pair of input samples
                    y_points = np.concatenate([
                        self.previous_samples[:, channel],  # Previous 9 samples
                        input_samples[i:i+2, channel]       # Current 2 samples
                    ])

                    # Create interpolation points for new samples (10,11,12)
                    x_new = np.array([10, 11, 12])

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
                    # This ensures continuity as this sample will be used in the next interpolation
                    output[output_idx+3, channel] = input_samples[i, channel]

                # Update output index
                output_idx += 4

                # Update previous samples: shift window by 4 and add the 4 new points
                self.previous_samples = np.roll(self.previous_samples, -4, axis=0)
                if self.method == 'repeat':
                    # For repeat method, use the same value for all 4 points
                    self.previous_samples[-4:, channel] = input_samples[i, channel]
                else:
                    self.previous_samples[-4:, channel] = np.concatenate([
                        output[output_idx-4:output_idx-1, channel],  # Points 10,11,12
                        [input_samples[i, channel]]                  # Point 13
                    ])

            self.chunk_end_sample[0, channel] = input_samples[len(input_samples)-1, channel]

        return output.flatten()