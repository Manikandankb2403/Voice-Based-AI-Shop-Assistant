import React, { useState } from 'react';
import axios from 'axios';

const AudioInput = ({ setMessages }) => {
  const [audio, setAudio] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState('');

  const startRecording = () => {
    setIsRecording(true);
    // Logic for starting audio recording
    // For simplicity, this can use the browser's MediaRecorder API or an external library
  };

  const stopRecording = () => {
    setIsRecording(false);
    // Logic for stopping audio recording and storing the audio file
    // Once audio is recorded, you can set it using setAudio
  };

  const sendAudio = async () => {
    if (!audio) return;
    const formData = new FormData();
    formData.append('audio', audio);

    try {
      const { data } = await axios.post('http://localhost:5000/audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      if (data.response) {
        setMessages((prev) => [...prev, { text: data.response, sender: 'bot' }]);
      } else {
        setError(data.error || 'Error processing audio');
      }
    } catch (err) {
      setError('Error sending audio');
    }
  };

  return (
    <div>
      <button onClick={startRecording} disabled={isRecording}>Start Recording</button>
      <button onClick={stopRecording} disabled={!isRecording}>Stop Recording</button>
      <button onClick={sendAudio}>Send Audio</button>
      {error && <p>{error}</p>}
    </div>
  );
};

export default AudioInput;