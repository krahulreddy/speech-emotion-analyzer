import io
import os

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from textblob import TextBlob
import json

def speech_sentiment_analyser(audio_clip):
    # Instantiates a client
    client = speech.SpeechClient()

    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        'resources',
        'test.wav')

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    sentiment_analysis_data = []

    for result in response.results:
        print('Transcript: {}'.format(result.alternatives[0].transcript))
        trans = TextBlob(format(result.alternatives[0].transcript))
        print("Sentiment", trans.sentiment)
        sentiment_analysis_data += set({format(result.alternatives[0].transcript),
            trans.sentiment})

    print('Actual Response: {}'.format(result))
    print("Sentiment data: ", json.dumps(sentiment_analysis_data))
    return sentiment_analysis_data
