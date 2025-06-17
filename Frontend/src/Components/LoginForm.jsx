import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import logo from "../Components/logo.png"; // Adjust the path as necessary

const LoginForm = () => {
    const [phoneNumber, setPhoneNumber] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const navigate = useNavigate();

    const handleLogin = async () => {
        if (!phoneNumber.trim()) {
            setErrorMessage('Phone number is required.');
            return;
        }

        try {
            const response = await axios.post('http://localhost:5000/login', {
                phone_number: phoneNumber,
            });

            if (response.data) {
                localStorage.setItem('customer', JSON.stringify(response.data));
                navigate('/chat');
            } else {
                setErrorMessage('Login failed. Please try again.');
            }
        } catch (error) {
            setErrorMessage('An error occurred. Please try again.');
        }
    };

    return (
        <div className="min-h-screen bg-green-50 flex items-center justify-center">
            <div className="bg-white p-8 rounded-lg shadow-lg w-full max-w-sm">
                {/* Top Center Image */}
                <div className="flex justify-center mb-4">
                    <img
                        src={logo} // Replace with your image path
                        alt="Logo"
                        className="h-20 w-20 object-contain"
                    />
                </div>

                <h2 className="text-3xl font-semibold text-center mb-6 text-green-700">Sign in</h2>

                <input
                    type="text"
                    value={phoneNumber}
                    onChange={(e) => setPhoneNumber(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') handleLogin();
                    }}
                    placeholder="Phone number"
                    className="w-full px-4 py-2 mb-4 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                />
                <button
                    onClick={handleLogin}
                    className="w-full bg-green-500 text-white py-2 rounded-md hover:bg-green-600 transition"
                >
                    Log in
                </button>

                {errorMessage && (
                    <p className="text-red-500 text-sm mt-3 text-center">{errorMessage}</p>
                )}

                <div className="mt-6 text-center text-gray-500 text-sm">
                    Don't have an account? <span className="text-green-500 cursor-pointer">Sign up</span>
                </div>
            </div>
        </div>
    );
};

export default LoginForm;
