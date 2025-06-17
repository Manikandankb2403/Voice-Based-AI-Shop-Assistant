import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FaMicrophone, FaMicrophoneSlash, FaShoppingCart } from "react-icons/fa";

const ChatPage = () => {
  const [message, setMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [audioFile, setAudioFile] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [customerName, setCustomerName] = useState("");
  const [cartDetails, setCartDetails] = useState("");
  const [showCart, setShowCart] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);

  useEffect(() => {
    const storedCustomer = localStorage.getItem("customer");
    if (storedCustomer) {
      try {
        const parsed = JSON.parse(storedCustomer);
        if (parsed.name) setCustomerName(parsed.name);
      } catch (err) {
        console.error("Invalid customer data in localStorage");
      }
    }
  }, []);

  const addMessageToChat = ({ text, sender }) => {
    setChatHistory((prevHistory) => [...prevHistory, { text, sender }]);
  };

  const sendMessage = async () => {
    if (!message.trim()) return;
    try {
      setIsProcessing(true);
      addMessageToChat({ text: message, sender: "user" });
      setErrorMessage("");
      const response = await axios.post("http://localhost:5000/message", {
        query: message,
        input_method: "text",
      });

      if (response.data?.message) {
        addMessageToChat({ text: response.data.message, sender: "bot" });
      } else {
        addMessageToChat({ text: "Bot could not generate a response.", sender: "bot" });
      }

      setMessage("");
    } catch {
      setErrorMessage("An error occurred while sending the message. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const sendAudioMessage = async (blobOverride = null) => {
    const fileToSend = blobOverride || audioFile || audioBlob;
    if (!fileToSend) {
      setErrorMessage("Please record or select an audio file first.");
      return;
    }

    const formData = new FormData();
    formData.append("audio", fileToSend);

    try {
      setIsProcessing(true);
      addMessageToChat({ text: "Listening...", sender: "bot" });
      setErrorMessage("");
      const response = await axios.post("http://localhost:5000/audio", formData);

      setChatHistory((prev) => prev.filter((msg) => msg.text !== "Listening..."));

      if (response.data?.recognized_text) {
        addMessageToChat({ text: response.data.recognized_text, sender: "user" });
      }

      if (response.data?.message) {
        addMessageToChat({ text: response.data.message, sender: "bot" });
      }

    } catch (error) {
      setChatHistory((prev) => prev.filter((msg) => msg.text !== "Listening..."));
      const errorMsg = error.response?.data?.error || "An error occurred while sending the audio.";
      setErrorMessage(errorMsg);
      addMessageToChat({ text: errorMsg, sender: "bot" });
    } finally {
      setAudioBlob(null);
      setAudioFile(null);
      setAudioURL(null);
      setIsProcessing(false);
    }
  };

  const startRecording = async () => {
    if (isProcessing) return;

    setIsRecording(true);
    setErrorMessage("");
    setAudioBlob(null);
    setAudioFile(null);
    setAudioURL(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunks, { type: "audio/wav" });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
        sendAudioMessage(blob);
      };

      mediaRecorderRef.current.start();
    } catch {
      setErrorMessage("Error accessing the microphone. Please check your device permissions.");
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  const fetchCart = async () => {
    try {
      const response = await axios.get("http://localhost:5000/cart");
      setCartDetails(response.data.message);
      setShowCart(true);
    } catch {
      setCartDetails("Unable to load cart details. Please try again later.");
      setShowCart(true);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("customer");
    window.location.href = "/";
  };

  return (
    <div className="h-screen flex flex-col bg-green-50">
      {/* Header */}
      <div className="bg-green-600 p-4 text-white font-bold text-xl flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <span>Vasee Shop Assistant</span>
          {customerName && <span className="text-sm font-normal">👤 {customerName}</span>}
          <button
            onClick={fetchCart}
            className="text-white hover:text-yellow-300 text-xl"
            title="View Cart"
          >
            <FaShoppingCart />
          </button>
        </div>
        <button
          onClick={handleLogout}
          className="bg-white text-green-600 px-4 py-1 rounded-full text-sm hover:bg-gray-200"
        >
          Logout
        </button>
      </div>

      {/* Chat History */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {chatHistory.map((message, index) => (
          <div key={index} className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`rounded-2xl p-3 max-w-[80%] text-sm ${
                message.sender === "user" ? "bg-green-300 text-right" : "bg-white text-left"
              }`}
            >
              {message.text}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="p-3 bg-green-100 flex items-center space-x-3">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !isProcessing) sendMessage();
          }}
          placeholder="Type a message"
          className="flex-1 p-2 rounded-full border focus:outline-none focus:ring-2 focus:ring-green-400"
        />
        <button
          onClick={sendMessage}
          disabled={isProcessing}
          className={`p-3 rounded-full text-white transition-all ${
            isProcessing ? "bg-gray-400 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
          }`}
        >
          ➤
        </button>
        {!isRecording ? (
          <button
            onClick={startRecording}
            disabled={isProcessing}
            className={`p-3 text-2xl rounded-full transition-all ${
              isProcessing ? "bg-gray-400 cursor-not-allowed" : "bg-green-500 hover:bg-green-600"
            } text-white`}
          >
            <FaMicrophone />
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="p-3 text-2xl rounded-full bg-red-500 hover:bg-red-600 text-white"
          >
            <FaMicrophoneSlash />
          </button>
        )}
      </div>

      {/* Audio Player */}
      {audioURL && (
        <div className="p-3 bg-green-100">
          <audio ref={audioPlayerRef} src={audioURL} controls className="w-full" />
        </div>
      )}

      {/* Error Message */}
      {errorMessage && <p className="text-red-500 text-center p-3">{errorMessage}</p>}

      {/* Cart Modal */}
      {showCart && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex justify-center items-center z-50">
          <div className="bg-white p-6 rounded-xl shadow-lg max-w-md w-full">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-bold">🛒 Your Cart</h2>
              <button onClick={() => setShowCart(false)} className="text-gray-600 hover:text-red-500 text-xl">
                &times;
              </button>
            </div>
            <div className="text-sm whitespace-pre-line">{cartDetails}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatPage;
