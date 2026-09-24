"""OmniVoice TTS nodes for ComfyUI."""

from .omnivoice_tts import OmniVoiceLongformTTS
from .voice_clone_node import OmniVoiceVoiceCloneTTS
from .voice_design_node import OmniVoiceVoiceDesignTTS
from .multi_speaker_node import OmniVoiceMultiSpeakerTTS

__all__ = [
    "OmniVoiceLongformTTS",
    "OmniVoiceVoiceCloneTTS",
    "OmniVoiceVoiceDesignTTS",
    "OmniVoiceMultiSpeakerTTS",
]
