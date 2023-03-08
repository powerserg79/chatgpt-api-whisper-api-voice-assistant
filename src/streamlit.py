import streamlit as st
import openai, config, subprocess
import sounddevice as sd
import soundfile as sf
import os

openai.api_key = config.OPENAI_API_KEY

messages = [
    {
        "role": "system",
        "content": "You are a therapist. Respond to all input in 25 words or less.",
    }
]


def transcribe(audio):
    global messages

    transcript = openai.Audio.transcribe("whisper-1", audio)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.Completion.create(
        model="davinci", prompt=transcript["text"], max_tokens=1024
    )

    system_message = response["choices"][0]["text"]
    messages.append({"role": "system", "content": system_message})

    subprocess.call(["say", system_message])

    chat_transcript = ""
    for message in messages:
        if message["role"] != "system":
            chat_transcript += message["role"] + ": " + message["content"] + "\n\n"

    return chat_transcript


st.title("Therapist Chatbot")

recording = None  # Variable to hold the recording

while True:
    if st.button("Record", key="record_button", help="Hold down to record"):
        if recording is None:
            # Start recording audio using the microphone
            fs = 44100  # Sample rate of audio
            duration = 0.1  # Duration of each audio buffer in seconds
            channels = 1  # Number of audio channels
            recording = sd.Stream(
                samplerate=fs, blocksize=int(fs * duration), channels=channels
            )

            with recording:
                while st.button(
                    "Record", key="record_button", help="Recording... Release to stop"
                ):
                    audio_data = recording.read(int(fs * duration))
                    if audio_data.size > 0:
                        if recording.dtype == "float32":
                            # Convert float32 audio data to int16 to save as WAV file
                            audio_data = (audio_data * 32767).astype("int16")

                        # Save audio data to file
                        if not os.path.exists("recordings"):
                            os.mkdir("recordings")
                        audio_file = os.path.join("recordings", "recording.wav")
                        with sf.SoundFile(
                            audio_file, mode="x", samplerate=fs, channels=channels
                        ) as file:
                            file.write(audio_data)

        else:
            # Stop recording audio
            recording.stop()
            st.write("Recording stopped.")

            # Call transcribe function with audio file as input
            chat_transcript = transcribe(audio_file)

            # Display chat transcript
            st.subheader("Chat Transcript")
            st.text_area("", chat_transcript)

            recording = None  # Reset recording variable

# Display instructions for recording
st.write(
    "To record an audio message, hold down the 'Record' button and speak into your microphone. Release the button to stop recording and transcribe the audio."
)
