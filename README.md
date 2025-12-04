<!--
+++
author = "Krzysztof Borowski"
title = "ASR Inference Service"
date = "2025-11-29"
description = "Automatic Speech Recognition (ASR) inference service using nvidia canary-1b-v2 model."
summary = "This service generates transcripts from audio in multiple languages. It is also useful to perform translations. All from elegant web api."
draft="false"
tags = [
    "python", 
    "cog",
    "nvidia-nemo"
]
categories = [
    "webdev",
    "AI"
]
+++
-->

# asr-inference-service

A robust Automatic Speech Recognition (ASR) inference service built with [Cog](https://github.com/replicate/cog). This service leverages NVIDIA's **Canary-1B** model to provide multilingual transcription and translation, wrapped in a sophisticated audio processing pipeline.

> **Note:** This is an **educational project** created to explore NVIDIA ASR models and the Cog framework. While functional, it is not currently optimized for high-scale production environments.

[Github Repo](https://github.com/Lokinado/asr-inference-service)

## Features

* **State-of-the-Art Model:** Uses `nvidia/canary-1b-v2` for high-quality transcription.
* **Advanced Preprocessing:**
    * **VAD (Voice Activity Detection):** Uses [Silero VAD](https://github.com/snakers4/silero-vad) to intelligently segment long audio files, removing silence and improving transcription accuracy for long recordings.
    * **Audio Normalization:** automatically normalizes audio chunks to -23 LUFS using `pyloudnorm` to ensure consistent input volume for the model.
    * **Resampling:** Automatically converts inputs to mono 16kHz.
* **Containerized:** Built with Cog for easy deployment as a Docker container.

## Hardware Requirements

* **GPU:** NVIDIA GPU is **required**.
    * **Minimum VRAM:** 6 GB.
    * **Recommended:** 12 GB+ (Tested on RTX 4070 Ti).
* **CPU:** CPU-only inference is **not supported** in this version.

## Supported Formats & Languages

**Audio Formats:**
`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`

**Languages:**
* Polish (`pl`)
* English (`en`)
* French (`fr`)

## Getting Started

### Prerequisites

* [Docker](https://docs.docker.com/get-docker/)
* [Cog](https://github.com/replicate/cog#install)

### Building the Image

To build the Docker image for the service run this command in `src` directory:

```bash
cog build -t asr-inference-service
```

### Running the Service

Start the inference server (requires NVIDIA Container Toolkit):

```bash
docker run -d -p 5000:5000 --gpus all asr-inference-service
```

## API Usage

Once the container is running, you can interact with it via HTTP requests.

### Endpoint: `/predictions`

**Method:** `POST`

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `audio` | String (URL/DataURI) | **Required** | The audio file to transcribe. |
| `source_lang` | String | `"pl"` | Language of speech in input audio (`pl`, `en`, `fr`). |
| `target_lang` | String | `"pl"` | Desired language of output transcription. |

### Example Request

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d @- \
  http://localhost:5000/predictions <<EOF
{
  "input": {
    "audio": "https://raw.githubusercontent.com/Lokinado/asr-inference-service/main/samples/obama.mp3",
    "source_lang": "en",
    "target_lang": "en"
  }
}
EOF
```

## Examples

Here you can see how the model performs on sample audio.

**Input Audio:**

Input audio can be found in samples folder. [Input audio file link](https://raw.githubusercontent.com/Lokinado/asr-inference-service/main/samples/obama.mp3)

<audio controls>
  <source src="https://raw.githubusercontent.com/Lokinado/asr-inference-service/main/samples/obama.mp3" type="audio/mpeg">
  Audio playback not available on this browser.
</audio>

**Output Transcription:**

> This week I travelled to Chicago to deliver my final farewell address to the nation, following the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people.at diners and on distant military outposts. All these conversations are what have kept me honest, kept me inspired and kept me going. Every day I learned from you. You made me a better president and you made me a better man.and the hope of the American people. I've seen neighbors looking out for each other as we rescued our economy from the worst crisis of our lives. I've hugged cancer survivors who finally know the security of affordable health care.I've seen the hopeful faces of young graduates and our newest military officers. I've mourned with grieving families searching for answers, and I found grace in a Charleston church. I've seen our scientists help a paralyzed man regain his sense of touch, and our wounded warriors walk again.I've learned from students who are building robots and curing diseases and who will change the world in ways we can't even imagine. I've seen the youngest of children remind us of our obligations to care for our refugees, to work in peace and, above all, to look out for each other.in the slow, hard, sometimes frustrating but always vital work of self government. But we can't take our democracy for granted. All of us, regardless of party, should throw ourselves into the work of citizenship, not just when there's an election.but over the full span of a lifetime. If you're tired of arguing with strangers on the Internet, try to talk with one in real life. If something needs fixing, lace up your shoes and do some organizing.Our success depends on our participation, regardless of which way the pendulum of power swings. It falls on each of us to be guardians of our democracy, to embrace the joyous task we've been given to continually try to improve this great nation of ours.citizen. It has been the honor of my life to serve you as president. Eight years later, I am even more optimistic about our country's promise, and I look forward to working along your side as a citizen for all my days that remain. Thanks, everybody. God bless you.

In case you find any issues with results of this asr inference service feel free to write an issue. I will try to adress as many problems as possible!

## How it Works

1.  **Ingestion:** The audio file is loaded, converted to mono, and resampled to 16kHz.
2.  **Segmentation:** Silero VAD scans the audio to create timestamps for speech segments, ignoring silence.
3.  **Chunking & Normalization:**
      * Segments are grouped into chunks (max 20 seconds).
      * Each chunk is loudness-normalized to -23 LUFS.
4.  **Inference:** The NVIDIA Canary model processes the batches of audio chunks.
5.  **Aggregation:** Text results are concatenated and returned as a single string.