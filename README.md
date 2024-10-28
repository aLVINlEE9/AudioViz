# AudioViz

AudioViz is a powerful Python-based tool that transforms audio content into a sequence of AI-generated images, creating visual narratives from sound. Using state-of-the-art AI models from OpenAI and Anthropic, it analyzes audio content and generates corresponding visual representations.

## Features

- **Smart Audio Processing**
  - Automatic audio compression and segmentation
  - Support for various audio formats
  - Intelligent handling of large audio files
  
- **Advanced Content Analysis**
  - Speech-to-text transcription using OpenAI's Whisper
  - Context-aware text analysis
  - Automated visual style guide generation
  
- **AI Image Generation**
  - High-quality image generation using DALL-E 3
  - Consistent visual styling across sequences
  - Customizable image parameters (size, quality, style)

## Use Cases

- YouTube Content Creation
- Podcast Visualization
- Educational Content
- Story Visualization
- Music Video Creation
- Speech Visualization

## Requirements

### Essential Dependencies
- Python 3.11+
- FFmpeg (Required for audio processing)
- OpenAI API key
- Anthropic API key (optional, for Claude-based text analysis)

### FFmpeg Installation
- **macOS (using Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install ffmpeg
  ```
- **Windows**:
  Download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/aLVINlEE9/AudioViz.git
   cd AudioViz
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
   ```

## Usage

Basic usage example:

```python
from audio_viz import AudioVisualizer

config = {
    "openai_api_key": "your_openai_api_key",
    "text_analysis": {
        "provider": "openai",  # or "anthropic"
        "max_tokens": 1000,
        "max_segments": 5
    },
    "image_generation": {
        "size": "1024x1024",
        "quality": "standard",
        "style": "vivid"
    }
}

visualizer = AudioVisualizer(config, verbose=True)
visualizer.process_audio("input.wav", "output_directory")
```

## Configuration Options

### Text Analysis
- `provider`: Choose between "openai" or "anthropic"
- `max_tokens`: Maximum tokens for LLM response
- `max_segments`: Target number of visual segments

### Image Generation
- `size`: "1024x1024", "1024x1792", or "1792x1024"
- `quality`: "standard" or "hd"
- `style`: "vivid" or "natural"

## Roadmap

- Enhanced prompting system for more detailed and accurate scene descriptions
- Video output support (combining audio with generated images)
- Additional image generation models support (Midjourney, Stable Diffusion)
- Custom style presets and templates
- Batch processing support
- Multiple language support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the repository or contact the maintainers.