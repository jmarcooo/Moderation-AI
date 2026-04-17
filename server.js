require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors()); // Allows your HTML file to communicate with this server
app.use(express.json()); // Allows the server to read JSON data

// Rate limiting for API calls (simple in-memory implementation)
const rateLimitStore = new Map();
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const RATE_LIMIT_MAX = 20; // 20 requests per minute

function checkRateLimit(ip) {
    const now = Date.now();
    const userRecord = rateLimitStore.get(ip) || { count: 0, startTime: now };
    
    if (now - userRecord.startTime > RATE_LIMIT_WINDOW) {
        userRecord.count = 1;
        userRecord.startTime = now;
    } else {
        userRecord.count++;
    }
    
    rateLimitStore.set(ip, userRecord);
    return userRecord.count <= RATE_LIMIT_MAX;
}

// The Moderation Route with Enhanced Accuracy
app.post('/api/moderate', async (req, res) => {
    const textToAnalyze = req.body.input;
    const clientIP = req.ip || req.connection.remoteAddress;

    // Input validation
    if (!textToAnalyze || typeof textToAnalyze !== 'string') {
        return res.status(400).json({ error: 'Invalid input: text is required' });
    }

    if (textToAnalyze.length === 0) {
        return res.status(400).json({ error: 'Input text cannot be empty' });
    }

    if (textToAnalyze.length > 20000) {
        return res.status(400).json({ error: 'Input text exceeds maximum length of 20,000 characters' });
    }

    // Rate limiting check
    if (!checkRateLimit(clientIP)) {
        return res.status(429).json({ error: 'Rate limit exceeded. Please try again later.' });
    }

    try {
        const response = await fetch('https://api.openai.com/v1/moderations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${process.env.OPENAI_API_KEY}` // Securely pulls from .env
            },
            body: JSON.stringify({ 
                input: textToAnalyze,
                model: 'text-moderation-latest' // Use the latest and most accurate model
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('OpenAI API Error:', response.status, errorData);
            
            if (response.status === 401) {
                return res.status(500).json({ error: 'OpenAI API authentication failed. Check your API key.' });
            } else if (response.status === 429) {
                return res.status(500).json({ error: 'OpenAI rate limit exceeded.' });
            } else {
                throw new Error(`OpenAI API Error: ${response.status}`);
            }
        }

        const data = await response.json();
        
        // Validate response structure
        if (!data.results || !Array.isArray(data.results) || data.results.length === 0) {
            throw new Error('Invalid response format from OpenAI API');
        }

        // Add metadata for better tracking
        const enhancedResponse = {
            ...data,
            metadata: {
                timestamp: new Date().toISOString(),
                input_length: textToAnalyze.length,
                model_used: 'text-moderation-latest'
            }
        };

        res.json(enhancedResponse); // Sends the AI's result back to your frontend

    } catch (error) {
        console.error('Server Error:', error);
        
        if (error.message.includes('API key')) {
            res.status(500).json({ error: 'Moderation service configuration error' });
        } else {
            res.status(500).json({ error: 'Failed to process moderation request' });
        }
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: 'moderation-api'
    });
});

app.listen(PORT, () => {
    console.log(`Secure Moderation Server running at http://localhost:${PORT}`);
    console.log(`Health check available at http://localhost:${PORT}/api/health`);
});
