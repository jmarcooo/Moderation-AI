require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors()); // Allows your HTML file to communicate with this server
app.use(express.json()); // Allows the server to read JSON data

// The Moderation Route
app.post('/api/moderate', async (req, res) => {
    const textToAnalyze = req.body.input;

    try {
        const response = await fetch('https://api.openai.com/v1/moderations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.OPENAI_API_KEY}` // Securely pulls from .env
            },
            body: JSON.stringify({ input: textToAnalyze })
        });

        if (!response.ok) {
            throw new Error(`OpenAI API Error: ${response.status}`);
        }

        const data = await response.json();
        res.json(data); // Sends the AI's result back to your frontend

    } catch (error) {
        console.error('Server Error:', error);
        res.status(500).json({ error: 'Failed to process moderation request' });
    }
});

app.listen(PORT, () => {
    console.log(`Secure Moderation Server running at http://localhost:${PORT}`);
});
