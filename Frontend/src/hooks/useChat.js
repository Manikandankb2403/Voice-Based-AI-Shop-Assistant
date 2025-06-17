import { useState, useEffect } from 'react';
import { sendMessage } from '../services/api';

export const useChat = (session, setSession) => {
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState('');

  const sendChatMessage = async (input, type) => {
    setMessages((prev) => [...prev, { text: input, isBot: false }]);
    setError('');
    try {
      const response = await sendMessage(input, type, session);
      setMessages((prev) => [
        ...prev,
        { text: response.message, isBot: true }
      ]);
      setSession(response.session);
    } catch (err) {
      setError(err.message || 'Error sending message');
    }
  };

  useEffect(() => {
    if (session.messages) {
      setMessages(session.messages);
    }
  }, [session]);

  return { messages, sendChatMessage, error };
};
