import json
import re
import time
import logging
from typing import List, Tuple, Dict

class TextAnalyzer:
    """
    A class for analyzing text content and generating visual style guidelines.
    
    This class processes text through multiple stages:
    1. Content filtering
    2. Context analysis
    3. Segment analysis
    4. Visual concept generation
    
    Uses either OpenAI or Anthropic LLMs for text analysis.
    """

    # Prompt template for overall style guide generation
    CONTEXT_PROMPT = '''
    You are a visual art director analyzing text to create a cohesive visual narrative.
    
    YOUR TASK:
    Analyze the full text and create a consistent visual style guide that will unify all images.
    
    FULL TEXT:
    {text}
    
    RESPOND WITH A JSON OBJECT CONTAINING:
    {{
        "overall_theme": "Main thematic elements and emotional tone",
        "visual_style": {{
            "art_style": "Consistent artistic approach",
            "color_palette": "3-5 key colors that represent the theme",
            "composition": "Preferred composition guidelines",
            "lighting": "Lighting style that matches the mood",
            "symbolic_elements": ["Recurring symbols or motifs to use"],
            "environment": "Common environmental elements"
        }},
        "mood_progression": {{
            "start": "Opening emotional tone",
            "middle": "Development of mood",
            "end": "Concluding emotional tone"
        }}
    }}
    '''
    
    # Prompt template for individual segment analysis
    SEGMENT_PROMPT = '''
    You are a visual scene creator working within an established style guide.
    
    STYLE GUIDE:
    {style_guide}
    
    TEXT TO VISUALIZE:
    {text}
    
    CREATE A SCENE THAT:
    1. Follows the established visual style
    2. Uses the defined color palette
    3. Incorporates recurring symbolic elements
    4. Maintains consistency with the overall theme
    5. Reflects the appropriate mood for this point in the narrative
    
    RESPOND WITH A JSON ARRAY:
    [
        {{
            "summary": "Brief description of this moment",
            "prompt": "Detailed DALL-E prompt incorporating style guide elements",
            "style_notes": "Specific style considerations for this scene"
        }}
    ]
    '''

    def __init__(self, config: dict, verbose: bool = False):
        """
        Initialize the TextAnalyzer with configuration.
        
        Required config structure:
        {
            "text_analysis": {
                "provider": str,     # "openai" or "anthropic"
                "api_key": str,      # Required for chosen provider
                "model": str,        # Model name to use
                "max_tokens": int,   # Maximum tokens for LLM response
                "max_segments": int  # Target number of segments to generate
            }
        }

        Args:
            config (dict): Configuration dictionary
            verbose (bool, optional): Enable detailed logging. Defaults to False.
        """
        self.config = config
        self.verbose = verbose
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def filter_text(self, text: str) -> str:
        """
        Filter sensitive content from text.
        
        Args:
            text (str): Raw input text

        Returns:
            str: Text with sensitive content replaced by appropriate placeholders
        """
        profanity = {
            # Korean profanity mappings
            'ㅅㅂ': '[expression]',
            'ㅆㅂ': '[expression]',
            '시발': '[expression]',
            '씨발': '[expression]',
            '병신': '[expression]',
            '지랄': '[expression]',
            '좆': '[expression]',
            '존나': '[very]',
            '니미': '[expression]',
            '엿': '[expression]',
            
            # English profanity mappings
            'fuck': '[expression]',
            'shit': '[expression]',
            'damn': '[expression]',
            'bitch': '[person]',
            'bastard': '[person]',
            'ass': '[expression]',
            
            # Common variants
            'f*ck': '[expression]',
            's*it': '[expression]',
            'f***': '[expression]',
            'sh*t': '[expression]',
            'b*tch': '[person]',
        }
        
        filtered_text = text.lower()
        for word, replacement in profanity.items():
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            filtered_text = pattern.sub(replacement, filtered_text)
            
        return filtered_text

    def analyze_context(self, segments: List[dict]) -> dict:
        """
        Analyze the overall context and generate visual style guide.

        Args:
            segments (List[dict]): List of text segments with keys:
                - text (str): Segment content
                - start (float): Start time
                - end (float): End time

        Returns:
            dict: Style guide containing:
                - overall_theme (str): Main theme
                - visual_style (dict): Art style, colors, composition, etc.
                - mood_progression (dict): Emotional progression
        """
        self.logger.info(f"[Context] Starting analysis of {len(segments)} segments")
        self.logger.info("[Context] Filtering and combining text...")
            
        filtered_texts = []
        total_chars = 0
        for i, segment in enumerate(segments, 1):
            filtered_text = self.filter_text(segment["text"])
            filtered_texts.append(filtered_text)
            total_chars += len(filtered_text)
            if i % 10 == 0:
                self.logger.info(f"[Context] Processed {i}/{len(segments)} segments")
        
        combined_text = " ".join(filtered_texts)
        
        self.logger.info(f"[Context] Combined text: {total_chars} characters")
        self.logger.info(f"[Context] Using {self.config['text_analysis']['provider'].upper()} model: {self.config['text_analysis']['model']}")
        
        response = self._get_llm_response(self.CONTEXT_PROMPT.format(text=combined_text))
        style_guide = json.loads(response)
        
        self.logger.info("\n=== Style Guide Generated ===")
        self.logger.info(f"Theme: {style_guide['overall_theme'][:50]}...")
        self.logger.info(f"Art Style: {style_guide['visual_style']['art_style']}")
        self.logger.info(f"Color Palette: {style_guide['visual_style']['color_palette']}")
            
        return style_guide

    def analyze_segment(self, segment: dict, style_guide: dict) -> List[dict]:
        """
        Generate visual concepts for a specific text segment.

        Args:
            segment (dict): Text segment with keys:
                - text (str): Segment content
                - start (float): Start time
                - end (float): End time
            style_guide (dict): Visual style guidelines

        Returns:
            List[dict]: List of visual concepts, each containing:
                - summary (str): Scene description
                - prompt (str): DALL-E generation prompt
                - style_notes (str): Additional style guidance
        """
        self.logger.info(f"[Segment] Processing segment ({len(segment['text'])} chars)")
        
        filtered_text = self.filter_text(segment["text"])
        self.logger.info("[Segment] Text filtered, generating concepts...")
        
        formatted_prompt = self.SEGMENT_PROMPT.format(
            style_guide=json.dumps(style_guide, indent=2),
            text=filtered_text
        )
        
        response = self._get_llm_response(formatted_prompt)
        concepts = json.loads(response)
        
        self.logger.info(f"[Segment] Generated {len(concepts)} concepts")
        for i, concept in enumerate(concepts, 1):
            self.logger.info(f"[Segment] Concept {i}: {concept['summary'][:50]}...")
        
        return concepts

    def select_segments(self, segments: List[dict], max_segments: int) -> List[dict]:
        """
        Merge segments into specified number of larger segments.
        
        Args:
            segments (List[dict]): Original text segments
            max_segments (int): Target number of segments

        Returns:
            List[dict]: Merged segments with original timing and combined text
        """
        if len(segments) <= max_segments:
            return segments
        
        total_segments = len(segments)
        segments_per_group = total_segments // max_segments
        
        merged_segments = []
        start_idx = 0
        
        for i in range(max_segments):
            if i == max_segments - 1:
                group = segments[start_idx:]
            else:
                end_idx = start_idx + segments_per_group
                group = segments[start_idx:end_idx]
                start_idx = end_idx
            
            merged_segment = {
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "text": " ".join(seg["text"] for seg in group)
            }
            
            merged_segments.append(merged_segment)
            
            self.logger.info(f"\n--- Segment Group {i+1}/{max_segments} ---")
            self.logger.info(f"Combined {len(group)} segments")
            self.logger.info(f"Time range: {merged_segment['start']:.1f}s - {merged_segment['end']:.1f}s")
            self.logger.info(f"Text length: {len(merged_segment['text'])} chars")
        
        return merged_segments

    def process_content(self, segments: List[dict]) -> Tuple[dict, List[dict], List[dict]]:
        """
        Process all content through the complete analysis pipeline.
        
        Args:
            segments (List[dict]): Text segments to process

        Returns:
            Tuple containing:
                - dict: Visual style guide
                - List[dict]: Generated visual concepts
                - List[dict]: Selected/merged segments used for generation
        """
        self.logger.info("\n=== Starting Content Analysis Pipeline ===")
        self.logger.info(f"Total segments: {len(segments)}")
        
        # Generate style guide
        self.logger.info("\n=== Phase 1: Style Guide Generation ===")
        style_guide = self.analyze_context(segments)
        
        # Generate visual concepts
        self.logger.info("\n=== Phase 2: Visual Concept Generation ===")
        
        # Select segments for visualization
        selected_segments = self.select_segments(
            segments, 
            self.config["text_analysis"]["max_segments"]
        )
        
        self.logger.info(f"Selected {len(selected_segments)} segments")
        
        all_concepts = []
        for i, segment in enumerate(selected_segments, 1):
            self.logger.info(f"\n--- Processing Segment {i}/{len(selected_segments)} ---")
            self.logger.info(f"Time range: {segment['start']:.2f}s - {segment['end']:.2f}s")
                
            concepts = self.analyze_segment(segment, style_guide)
            all_concepts.extend(concepts)
            
            self.logger.info(f"Total concepts: {len(all_concepts)}")
        
        self.logger.info("\n=== Content Analysis Complete ===")
        self.logger.info("Style guide generated")
        self.logger.info(f"Total concepts: {len(all_concepts)}")
        
        return style_guide, all_concepts, selected_segments

    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from configured language model.

        Args:
            prompt (str): Prompt to send to LLM

        Returns:
            str: LLM response text

        Raises:
            Exception: If LLM request fails
        """
        text_config = self.config["text_analysis"]
        
        self.logger.info(f"[LLM] Request to {text_config['provider'].upper()}")
        self.logger.info(f"[LLM] Max tokens: {text_config['max_tokens']}")
        
        start_time = time.time()
        
        try:
            if text_config["provider"] == "anthropic":
                response = text_config["client"].messages.create(
                    model=text_config["model"],
                    max_tokens=text_config["max_tokens"],
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.1
                )
                result = response.content[0].text.strip()
            else:  # OpenAI
                response = self.openai_client.chat.completions.create(
                    model=text_config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=text_config["max_tokens"],
                    temperature=0.1
                )
                result = response.choices[0].message.content
                
            elapsed_time = time.time() - start_time
            
            self.logger.info(f"[LLM] Response received in {elapsed_time:.2f}s")
            self.logger.info(f"[LLM] Response length: {len(result)} chars")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[LLM] Error: {str(e)}")
            raise