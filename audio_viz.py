import openai
from openai import OpenAI
import os
import time
import random
import requests
from typing import List, Tuple, Dict
import json
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from pydub import AudioSegment
from audio_processor import AudioProcessor
import subprocess
from text_analyzer import TextAnalyzer
import logging

class AudioVisualizer:
    """
    A class for processing audio files and generating visual representations.
    
    The pipeline includes:
    1. Audio compression
    2. Speech-to-text transcription using Whisper
    3. Text segmentation
    4. Context analysis
    5. Image generation using DALL-E
    """

    def __init__(self, config: dict, verbose: bool = False):
        """
        Initialize audio visualization pipeline with configuration.
        
        Required config structure:
        {
            "openai_api_key": str,  # Required for Whisper and DALL-E
            "text_analysis": {
                "provider": str,     # "openai" or "anthropic"
                "api_key": str,      # Required if provider is "anthropic"
                "model": str,        # Optional, uses default if not provided
                "max_tokens": int,   # Maximum tokens for LLM response
                "max_segments": int  # Target number of segments to generate
            },
            "image_generation": {    # Optional, defaults provided
                "size": str,         # "1024x1024", "1024x1792", or "1792x1024"
                "quality": str,      # "standard" or "hd"
                "style": str         # "vivid" or "natural"
            }
        }

        Args:
            config (dict): Configuration dictionary for the pipeline
            verbose (bool, optional): Enable detailed process logging. Defaults to False.

        Raises:
            ValueError: If required configuration parameters are missing
        """
        self.verbose = verbose
        self.config = self._validate_config(config)
        self.openai_client = OpenAI(api_key=self.config["openai_api_key"])
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_visualizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.temp_dir = "temp_audio"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.audio_processor = AudioProcessor()
        
        if self.verbose:
            self.logger.info("=== AudioVisualizer Initialization ===")
            self.logger.info(f"Text Analysis Provider: {config['text_analysis']['provider']}")
            self.logger.info(f"Image Generation Quality: {config['image_generation']['quality']}")

    def _validate_config(self, config: dict) -> dict:
        """
        Validate and set default configuration values.

        Args:
            config (dict): Raw configuration dictionary

        Returns:
            dict: Validated configuration with defaults applied

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if "openai_api_key" not in config:
            raise ValueError("OpenAI API key is required")
            
        default_config = {
            "text_analysis": {
                "provider": "openai",
                "model": "gpt-4",
                "max_tokens": 1000,
                "max_segments": 5
            },
            "image_generation": {
                "size": "1024x1024",
                "quality": "standard",
                "style": "vivid"
            }
        }
        
        config = {**default_config, **config}
        
        text_analysis = config["text_analysis"]
        if text_analysis["provider"] not in ["openai", "anthropic"]:
            raise ValueError("text_analysis provider must be either 'openai' or 'anthropic'")
            
        if text_analysis["provider"] == "anthropic":
            if "api_key" not in text_analysis:
                raise ValueError("Anthropic API key is required when using Claude")
            text_analysis["client"] = Anthropic(api_key=text_analysis["api_key"])
            if "model" not in text_analysis:
                text_analysis["model"] = "claude-3-sonnet-20240229"
                
        return config

    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Transcribe audio file using Whisper API with compression if needed.

        Args:
            audio_path (str): Path to the input audio file

        Returns:
            dict: Whisper API response containing transcribed segments and complete text

        Raises:
            Exception: If audio compression or transcription fails
        """
        try:
            self.logger.info(f"[Transcribe] Processing: {audio_path}")

            try:
                processed_audio = self.audio_processor.compress_audio(audio_path)
                processed_audio_path = processed_audio[0][0] if isinstance(processed_audio, list) else processed_audio
                is_compressed = processed_audio_path != audio_path
            except Exception as e:
                self.logger.error(f"[Compression] Failed: {str(e)}")
                self._log_process("compression", "error", {"error": str(e)})
                raise
                
            if is_compressed:
                original_size = self.audio_processor.get_file_size(audio_path)
                compressed_size = self.audio_processor.get_file_size(processed_audio_path)
                compression_info = {
                    "original_size_mb": original_size/1024/1024,
                    "compressed_size_mb": compressed_size/1024/1024
                }
                self._log_process("compression", "success", compression_info)
                self.logger.info(f"[Compress] {compression_info['original_size_mb']:.2f}MB -> {compression_info['compressed_size_mb']:.2f}MB")
            
            with open(processed_audio_path, "rb") as audio_file:
                response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
                
            if is_compressed and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
                
            self._log_process("transcription", "success", {"file": audio_path})
            self.logger.info("[Transcribe] Completed successfully")
            
            return response
            
        except Exception as e:
            self.logger.error(f"[Transcribe] Failed: {str(e)}")
            self._log_process("transcription", "error", {"file": audio_path, "error": str(e)})
            raise

    def segment_transcript(self, transcription: dict) -> List[dict]:
        """
        Segment transcript into logical parts for visualization.
        
        Args:
            transcription (dict): Whisper API transcription response
            
        Returns:
            List[dict]: List of segments containing:
                - start (float): Start time in seconds
                - end (float): End time in seconds
                - text (str): Segment text content
        """
        try:
            self.logger.info(f"[Segment] Processing {len(transcription.segments)} segments")
                
            segments = []
            current_segment = {
                "start": transcription.segments[0].start,
                "end": transcription.segments[0].end,
                "text": transcription.segments[0].text
            }
            
            for i in range(1, len(transcription.segments)):
                seg = transcription.segments[i]
                
                # Create new segment if gap > 1s or at sentence end
                if (seg.start - current_segment["end"] > 1.0 or 
                    current_segment["text"].rstrip().endswith(('.', '!', '?', '。', '！', '？'))):
                    segments.append(current_segment)
                    current_segment = {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    }
                else:
                    current_segment["end"] = seg.end
                    current_segment["text"] += " " + seg.text
                    
            segments.append(current_segment)
            
            self.logger.info(f"[Segment] Created {len(segments)} segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"[Segment] Failed: {str(e)}")
            self._log_process("segmentation", "error", {"error": str(e)})
            raise

    def generate_image(self, prompt: str, filename: str, output_dir: str, 
                    max_retries: int = 3, retry_delay: float = 2.0,
                    api_timeout: float = 30.0, download_timeout: float = 30.0) -> str:
        """
        Generate image using DALL-E API with retry logic and timeouts.

        Args:
            prompt (str): Image generation prompt
            filename (str): Output filename
            output_dir (str): Output directory path
            max_retries (int, optional): Maximum retry attempts. Defaults to 3
            retry_delay (float, optional): Delay between retries in seconds. Defaults to 2.0
            api_timeout (float, optional): Timeout for API request in seconds. Defaults to 30.0
            download_timeout (float, optional): Timeout for image download in seconds. Defaults to 30.0

        Returns:
            str: Path to generated image file

        Raises:
            Exception: If image generation fails after all retries
        """
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                if attempt > 0:
                    self.logger.info(f"[Generate] Retry {attempt}/{max_retries}: {filename}")
                else:
                    self.logger.info(f"[Generate] Creating: {filename}")

                img_config = self.config["image_generation"]
                
                # Validate configuration
                valid_sizes = ["1024x1024", "1024x1792", "1792x1024"]
                if img_config["size"] not in valid_sizes:
                    self.logger.warning(f"[Generate] Invalid size {img_config['size']}, using 1024x1024")
                    img_config["size"] = "1024x1024"
                    
                valid_qualities = ["standard", "hd"]
                if img_config["quality"] not in valid_qualities:
                    self.logger.warning(f"[Generate] Invalid quality {img_config['quality']}, using standard")
                    img_config["quality"] = "standard"
                    
                valid_styles = ["vivid", "natural"]
                if img_config["style"] not in valid_styles:
                    self.logger.warning(f"[Generate] Invalid style {img_config['style']}, using vivid")
                    img_config["style"] = "vivid"

                if attempt == 0:
                    self.logger.info(f"[Generate] Config: {img_config['size']}, {img_config['quality']}, {img_config['style']}")
                
                # Generate image with timeout
                try:
                    response = self.openai_client.images.generate(
                        model="dall-e-3", 
                        prompt=prompt,
                        n=1,
                        size=img_config["size"],
                        quality=img_config["quality"],
                        style=img_config["style"],
                        timeout=api_timeout  # Add timeout for API request
                    )
                except TimeoutError as te:
                    raise Exception(f"API request timed out after {api_timeout} seconds") from te
                
                # Save image with timeout
                image_url = response.data[0].url
                image_path = os.path.join(output_dir, filename)
                
                try:
                    img_response = requests.get(
                        image_url, 
                        timeout=download_timeout  # Add timeout for download
                    )
                    img_response.raise_for_status()
                except requests.Timeout as rt:
                    raise Exception(f"Image download timed out after {download_timeout} seconds") from rt
                
                with open(image_path, 'wb') as f:
                    f.write(img_response.content)
                    
                self._log_process("image_generation", "success", {
                    "prompt": prompt,
                    "filename": filename,
                    "config": img_config,
                    "attempts": attempt + 1
                })

                self.logger.info(f"[Generate] Saved to: {image_path}")
                return image_path
                
            except Exception as e:
                last_error = e
                attempt += 1
                
                error_type = type(e).__name__
                error_msg = str(e)
                self.logger.error(f"[Generate] Error on attempt {attempt} ({error_type}): {error_msg}")
                
                self._log_process("image_generation", "error", {
                    "prompt": prompt,
                    "error": error_msg,
                    "error_type": error_type,
                    "attempt": attempt,
                    "config": img_config
                })
                
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    base_wait = retry_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 0.1 * base_wait)  # 10% jitter
                    retry_wait = base_wait + jitter
                    self.logger.info(f"[Generate] Waiting {retry_wait:.1f}s before retry...")
                    time.sleep(retry_wait)
                
        self.logger.error(f"[Generate] Failed after {max_retries} attempts")
        raise last_error

    def _format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to timestamp format.

        Args:
            seconds (float): Time in seconds

        Returns:
            str: Formatted timestamp (MMSSCC)
        """
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 100)
        return f"{minutes:02d}{seconds:02d}{milliseconds:02d}"
        
    def _log_process(self, process: str, status: str, details: dict):
        """
        Log process details to file.

        Args:
            process (str): Name of the process
            status (str): Status of the process ("success" or "error")
            details (dict): Additional process details
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "process": process,
            "status": status,
            "details": details
        }
        
        log_file = os.path.join(self.log_dir, f"{process}.log")
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')


    def process_audio(self, audio_path: str, output_dir: str):
        """
        Process audio file through the complete visualization pipeline.
        
        Args:
            audio_path (str): Path to input audio file
            output_dir (str): Directory for output files

        Raises:
            Exception: If any stage of the pipeline fails
        """
        self.logger.info("\n=== Starting Audio Visualization Pipeline ===")
        self.logger.info(f"Input: {audio_path}")
        self.logger.info(f"Output: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        
        try:
            text_analyzer = TextAnalyzer(self.config, verbose=self.verbose)
            
            self.logger.info("\n=== Stage 1/5: Audio Processing ===")
            audio_segments = self.audio_processor.compress_audio(audio_path)
            
            all_segments = []
            for segment_path, start_time in audio_segments:
                self.logger.info(f"\n=== Stage 2/5: Transcription (segment at {start_time:.2f}s) ===")
                transcription = self.transcribe_audio(segment_path)
                
                for segment in transcription.segments:
                    segment.start += start_time
                    segment.end += start_time
                
                self.logger.info("\n=== Stage 3/5: Segmentation ===")
                segments = self.segment_transcript(transcription)
                all_segments.extend(segments)
            
            self.logger.info("\n=== Stage 4/5: Context Analysis ===")
            style_guide, visual_concepts, selected_segments = text_analyzer.process_content(all_segments)
            
            # Save style guide
            style_guide_path = os.path.join(output_dir, "style_guide.json")
            with open(style_guide_path, 'w', encoding='utf-8') as f:
                json.dump(style_guide, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"\n=== Stage 5/5: Visualization ({len(visual_concepts)} concepts) ===")
                
            for i, (concept, segment) in enumerate(zip(visual_concepts, selected_segments), 1):
                self.logger.info(f"\n--- Processing Concept {i}/{len(visual_concepts)} ---")
                    
                enhanced_prompt = f"{concept['prompt']}, {concept.get('style_notes', '')}"
                start_str = self._format_timestamp(segment["start"])
                end_str = self._format_timestamp(segment["end"])
                filename = f"{start_str}_{end_str}.png"
                            
                self.logger.info(f"[Visualize] Generating image for segment {start_str}-{end_str}")
                image_path = self.generate_image(enhanced_prompt, filename, output_dir)
                
                # Create metadata for the segment
                metadata = {
                    "timing": {
                        "start": segment["start"],
                        "end": segment["end"],
                        "formatted": f"{start_str}-{end_str}"
                    },
                    "concept": concept,
                    "style_guide": style_guide,
                    "image_path": image_path,
                    "text_analysis": {
                        "provider": self.config["text_analysis"]["provider"],
                        "model": self.config["text_analysis"]["model"]
                    },
                    "image_generation": self.config["image_generation"]
                }
                
                # Save metadata
                metadata_path = os.path.join(output_dir, f"{start_str}_{end_str}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
            self.logger.info("\n=== Process Completed Successfully ===")
            self.logger.info(f"Style guide saved to: {style_guide_path}")
            self.logger.info(f"Results saved to: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"\n=== Process Failed ===")
            self.logger.error(f"Error: {str(e)}")
            raise

    def cleanup(self):
        """
        Clean up temporary files created during processing.
        
        Notes:
            - Removes temporary files but preserves directories
            - Errors during deletion are logged but not raised
            - Should be called after processing is complete
        """
        self.logger.info("\n=== Starting Cleanup ===")
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        self.logger.info(f"[Cleanup] Removed: {file_path}")
                except Exception as e:
                    self.logger.error(f"[Cleanup] Error deleting {file_path}: {e}")
        self.logger.info("=== Cleanup Complete ===")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Example configuration using Claude
    claude_config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "text_analysis": {
            "provider": "anthropic",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1500,
            "max_segments": 5
        },
        "image_generation": {
            "size": "1792x1024",
            "quality": "hd",
            "style": "vivid"
        }
    }
    
    try:
        visualizer = AudioVisualizer(claude_config, verbose=True)
        visualizer.process_audio("/Users/alvinlee/Downloads/Untitled notebook (2).wav", "output_images_v5")
    finally:
        visualizer.cleanup()