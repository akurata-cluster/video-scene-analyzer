import os
import tempfile
import ffmpeg
from faster_whisper import WhisperModel

class AudioProcessor:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """Initialize the faster-whisper model."""
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def extract_audio_chunk(self, video_path: str, start_time: float, end_time: float) -> str:
        """Extracts an audio chunk from the video and returns a temporary file path."""
        duration = end_time - start_time
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            (
                ffmpeg
                .input(video_path, ss=start_time, t=duration)
                .output(temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k', loglevel='quiet')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            # Maybe there is no audio in the video
            return None
        
        return temp_audio_path

    def transcribe_chunk(self, video_path: str, start_time: float, end_time: float) -> str:
        """Extracts and transcribes a chunk of audio from the video."""
        temp_audio_path = self.extract_audio_chunk(video_path, start_time, end_time)
        if not temp_audio_path or not os.path.exists(temp_audio_path):
            return ""

        try:
            # Check file size to avoid trying to transcribe empty audio
            if os.path.getsize(temp_audio_path) < 100:
                return ""
            
            segments, info = self.model.transcribe(temp_audio_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments]).strip()
            return transcript
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
