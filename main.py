import argparse
import os
import sys
from audio_upscaler import AudioUpscaler

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Upscale WAV files from 44.1kHz to 172.4kHz'
    )
    parser.add_argument(
        'input_file',
        help='Input WAV file (44.1kHz/16bit stereo)'
    )
    parser.add_argument(
        'output_file',
        help='Output WAV file (172.4kHz/16bit stereo)'
    )
    parser.add_argument(
        '--method',
        choices=['cubic', 'akima', 'pchip', 'repeat'],
        default='cubic',
        help='Interpolation method to use'
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1

    # Check if output directory is writable
    output_dir = os.path.dirname(args.output_file) or '.'
    if not os.access(output_dir, os.W_OK):
        print(f"Error: Cannot write to output directory '{output_dir}'")
        return 1

    # Create upscaler and process file
    upscaler = AudioUpscaler(interpolation_method=args.method)
    success, message = upscaler.process_file(args.input_file, args.output_file)

    # Print final message
    print(f"\n{message}")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())