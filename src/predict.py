import os
import tempfile
from typing import Any, Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.collections.asr.models.aed_multitask_models import (
    MultiTaskTranscriptionConfig,
)


class Predictor(BasePredictor):
    def setup(self):
        print("Loading VAD model...")
        self.vad_model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )

        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = self.utils
        print("VAD Model loading done!")

        print("Loading model...")
        self.canary_model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
        self.canary_model.preprocessor.to("cuda")
        print("Model loading done!")

        self.supported_formats = (
            ".mp3",
            ".wav",
            ".m4a",
            ".flac",
            ".ogg",
            ".aac",
        )
        self.supported_languages = (
            "pl",
            "en",
            "fr",
        )  # Subset of all supported languages by canary-1b-v2 model
        self.chunk_length = 20  # Maximum duration of one audio chunk in seconds
        self.target_sample_rate = 16000  # Sample rate required by canary-1b-v2 model
        self.meter = pyln.Meter(
            self.target_sample_rate
        )  # Helper object for audio normalization.
        self.target_LUFS = -23  # Number of LUFS to normalize audo volume to

    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        source_lang: Optional[str] = Input(
            description="Language of speech in input audio file", default="pl"
        ),
        target_lang: Optional[str] = Input(
            description="Desired language of output transcription", default="pl"
        ),
    ) -> Any:
        audio_path = str(audio)
        source_lang = source_lang.lower()
        target_lang = source_lang.lower()

        # Check if format is supported
        if os.path.splitext(audio_path)[1].lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported format. Supported formats are: {" ".join(self.supported_formats)}."
            )

        # Check if source language is supported
        if source_lang not in self.supported_languages:
            raise ValueError(
                f"Unsupported source language. Supported languages are: {" ".join(self.supported_languages)}."
            )

        # Check if target language is supported
        if target_lang not in self.supported_languages:
            raise ValueError(
                f"Unsupported target language. Supported languages are: {" ".join(self.supported_languages)}."
            )

        cfg: MultiTaskTranscriptionConfig = self.canary_model.get_transcribe_config()

        cfg.prompt = [
            {
                "slots": {
                    "source_lang": f"<|{source_lang}|>",
                    "target_lang": f"<|{target_lang}|>",
                },
                "role": "user",
            }
        ]

        audio_data, original_sr = sf.read(audio_path)

        # Convert to mono
        if audio_data.ndim > 1:
            print("Conversion to mono needed...")
            audio_data = audio_data.mean(axis=1)
            print("Conversion to mono done!")

        # Resample to 16 kHz
        if original_sr != self.target_sample_rate:
            print("Resampling needed...")
            audio_data = librosa.resample(
                audio_data.T, orig_sr=original_sr, target_sr=self.target_sample_rate
            ).T
            print("Resampling done!")

        # Identify fragments of speech with voice activation detection model
        print("Starting VAD...")
        speech_timestamps = self.get_speech_timestamps(
            audio_data,
            self.vad_model,
            threshold=0.5,
            sampling_rate=self.target_sample_rate,
            min_speech_duration_ms=100,
            max_speech_duration_s=20,
            speech_pad_ms=400,
            min_silence_duration_ms=200,
        )
        print(f"Created {len(speech_timestamps)} segments")
        print("VAD finished!")

        with tempfile.TemporaryDirectory() as tmp_chunks_dir:
            print("Saving segments as .wav audio chunks to disk...")
            chunk_audio = np.ndarray([])
            chunk_index = 0

            for segment in speech_timestamps:
                segment_size = segment["end"] - segment["start"]

                if (
                    chunk_audio.size + segment_size
                    > self.chunk_length * self.target_sample_rate
                ):
                    chunk_path = os.path.join(
                        tmp_chunks_dir, f"chunk_{chunk_index}.wav"
                    )
                    loudness = self.meter.integrated_loudness(chunk_audio)
                    chunk_audio = pyln.normalize.loudness(
                        chunk_audio, loudness, self.target_LUFS
                    )
                    sf.write(
                        chunk_path,
                        chunk_audio,
                        self.target_sample_rate,
                    )
                    print(f"Saved .wav file: {chunk_path}")
                    chunk_index += 1
                    chunk_audio = np.ndarray([])  # Clear audio chunk data
                else:
                    chunk_audio = np.append(
                        chunk_audio, audio_data[segment["start"] : segment["end"]]
                    )

            audio_chunks_count = len(os.listdir(tmp_chunks_dir))

            print(f"Created {audio_chunks_count} audio chunks")

            print("Starting transcription...")
            audio_chunk_paths = []
            for audio_chunk_index in range(audio_chunks_count):
                audio_chunk_path = os.path.join(
                    tmp_chunks_dir, f"chunk_{audio_chunk_index}.wav"
                )

                audio_chunk_paths.append(audio_chunk_path)

            predictions = self.canary_model.transcribe(
                audio=audio_chunk_paths, batch_size=16, override_config=cfg
            )

            transcriptions = [p.text for p in predictions]
            print("Transcription done!")

        return "".join(transcriptions).strip()
