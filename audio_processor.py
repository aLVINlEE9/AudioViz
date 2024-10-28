import os
import subprocess
from pydub import AudioSegment

class AudioProcessor:
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def compress_audio(input_path: str, max_size_bytes: int = MAX_FILE_SIZE) -> list:
        """
        Compress audio file using multiple compression strategies
        Returns list of compressed file paths with their start times
        """
        file_size = AudioProcessor.get_file_size(input_path)
        if file_size <= max_size_bytes:
            return [(input_path, 0)]  # Single file with start time 0
            
        print(f"Original file size: {file_size/1024/1024:.2f}MB")
        
        # Create temp directory
        temp_dir = "temp_audio"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate output path
        file_name = os.path.basename(input_path)
        base_name, ext = os.path.splitext(file_name)
        
        # First try pydub strategies
        try:
            compressed_path = AudioProcessor._try_pydub_compression(
                input_path, temp_dir, base_name, ext, max_size_bytes
            )
            if compressed_path:
                return [(compressed_path, 0)]  # Single compressed file with start time 0
        except Exception as e:
            print(f"Pydub compression failed: {e}")
        
        # If compression fails, split and process multiple segments
        return AudioProcessor._split_and_compress(
            input_path, temp_dir, base_name, ext, max_size_bytes
        )
    
    @staticmethod
    def _try_pydub_compression(input_path, temp_dir, base_name, ext, max_size_bytes):
        """Try compression using pydub"""
        compressed_path = os.path.join(temp_dir, f"{base_name}_compressed{ext}")
        
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Compression strategies
        strategies = [
            # Basic strategies
            lambda a: a.set_channels(1),
            lambda a: a.set_frame_rate(24000),
            lambda a: a.set_frame_rate(16000),
            lambda a: a.set_frame_rate(16000).set_channels(1),
            lambda a: a.set_frame_rate(16000).set_channels(1).set_sample_width(2),
            
            # More aggressive strategies
            lambda a: a.set_frame_rate(8000).set_channels(1),
            lambda a: a.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"Trying pydub compression strategy {i}...")
            compressed_audio = strategy(audio)
            compressed_audio.export(compressed_path, format=ext.lstrip('.'))
            
            new_size = AudioProcessor.get_file_size(compressed_path)
            print(f"Compressed file size: {new_size/1024/1024:.2f}MB")
            
            if new_size <= max_size_bytes:
                print(f"Successfully compressed using pydub strategy {i}")
                return compressed_path
        
        return None
    
    @staticmethod
    def _try_ffmpeg_compression(input_path, temp_dir, base_name, ext, max_size_bytes):
        """Try compression using FFmpeg with multiple strategies"""
        # FFmpeg compression strategies with increasing compression
        ffmpeg_strategies = [
            # Strategy 1: Basic compression
            {
                'ac': '1',
                'ar': '16000',
                'b:a': '64k'
            },
            # Strategy 2: More compression
            {
                'ac': '1',
                'ar': '12000',
                'b:a': '48k'
            },
            # Strategy 3: Heavy compression
            {
                'ac': '1',
                'ar': '8000',
                'b:a': '32k'
            },
            # Strategy 4: Extreme compression
            {
                'ac': '1',
                'ar': '8000',
                'b:a': '24k'
            },
            # Strategy 5: Maximum compression
            {
                'ac': '1',
                'ar': '8000',
                'b:a': '16k',
                'compression_level': '10'
            }
        ]
        
        for i, strategy in enumerate(ffmpeg_strategies, 1):
            print(f"Trying FFmpeg compression strategy {i}...")
            ffmpeg_output = os.path.join(temp_dir, f"{base_name}_ffmpeg_{i}{ext}")
            
            try:
                # Build FFmpeg command
                cmd = ['ffmpeg', '-y', '-i', input_path]
                
                # Add compression parameters
                for key, value in strategy.items():
                    if key != 'compression_level':
                        cmd.extend([f'-{key}', value])
                
                # Add output file
                cmd.append(ffmpeg_output)
                
                # Run FFmpeg
                subprocess.run(cmd, check=True, capture_output=True)
                
                final_size = AudioProcessor.get_file_size(ffmpeg_output)
                print(f"FFmpeg strategy {i} compressed size: {final_size/1024/1024:.2f}MB")
                
                if final_size <= max_size_bytes:
                    print(f"Successfully compressed using FFmpeg strategy {i}")
                    return ffmpeg_output
                
                # Clean up failed attempt
                if os.path.exists(ffmpeg_output):
                    os.remove(ffmpeg_output)
                    
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg strategy {i} failed: {e}")
                if os.path.exists(ffmpeg_output):
                    os.remove(ffmpeg_output)
                continue
        
        # If we reach here, try splitting the file
        return AudioProcessor._split_and_compress(
            input_path, temp_dir, base_name, ext, max_size_bytes
        )
    
    @staticmethod
    def _split_and_compress(input_path, temp_dir, base_name, ext, max_size_bytes):
        """Split the audio file into smaller chunks and compress"""
        print("Attempting to split and compress file...")
        
        try:
            # Load the audio file
            audio = AudioSegment.from_file(input_path)
            duration_ms = len(audio)
            
            # Calculate number of segments needed (aim for 20MB per segment)
            target_size = 20 * 1024 * 1024  # 20MB
            num_segments = max(2, int(AudioProcessor.get_file_size(input_path) / target_size) + 1)
            segment_duration = duration_ms // num_segments
            
            compressed_segments = []
            
            # Process each segment
            for i in range(num_segments):
                start_ms = i * segment_duration
                end_ms = min((i + 1) * segment_duration, duration_ms)
                
                segment = audio[start_ms:end_ms]
                output_path = os.path.join(temp_dir, f"{base_name}_part{i+1}{ext}")
                
                # Export with aggressive compression
                segment = segment.set_channels(1).set_frame_rate(8000)
                segment.export(
                    output_path,
                    format=ext.lstrip('.'),
                    parameters=["-b:a", "16k"]
                )
                
                final_size = AudioProcessor.get_file_size(output_path)
                if final_size <= max_size_bytes:
                    print(f"Successfully compressed segment {i+1}: {final_size/1024/1024:.2f}MB")
                    # Store segment path and its start time in seconds
                    compressed_segments.append((output_path, start_ms / 1000.0))
                else:
                    raise ValueError(f"Segment {i+1} still too large after compression")
            
            return compressed_segments
            
        except Exception as e:
            print(f"Split and compress failed: {e}")
            raise ValueError("Could not compress file to required size with any method")