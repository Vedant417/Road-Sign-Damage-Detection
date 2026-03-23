const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 8000;

app.use(cors());

// Configure Multer for temp storage
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => cb(null, Date.now() + path.extname(file.originalname))
});

const upload = multer({ storage });

app.get('/', (req, res) => {
    res.json({ message: "Node.js Road Sign Damage Detection API running!" });
});

app.post('/api/detect', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded', status: 'Failed' });
    }

    const imagePath = req.file.path;
    
    // Utilize the virtual environment python if available
    const pythonExecutable = path.resolve(__dirname, '../backend/venv/Scripts/python.exe');
    const pythonScript = path.join(__dirname, 'run_yolo.py');
    const pythonToRun = fs.existsSync(pythonExecutable) ? pythonExecutable : 'python';

    const pythonProcess = spawn(pythonToRun, [pythonScript, imagePath]);
    
    let dataString = '';
    
    pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
    // YOLO often logs warnings to stderr
        console.warn(`[Python ML Log] ${data}`); 
    });
    
    pythonProcess.on('close', (code) => {
        // Cleanup file
        if (fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
        }
        
        try {
            // Safely extract the JSON from potentially messy stdout
            const lines = dataString.trim().split('\n');
            let jsonString = null;
            
            for (let i = lines.length - 1; i >= 0; i--) {
                const line = lines[i].trim();
                // Usually the final output is the JSON dictionary
                if (line.startsWith('{') && line.endsWith('}')) {
                    jsonString = line;
                    break;
                }
            }
            
            if (!jsonString) {
                  throw new Error("No JSON found in python script output");
            }
            
            const result = JSON.parse(jsonString);
            return res.json(result);
            
        } catch (e) {
            console.error("Parse Error:", e);
            console.error("Raw stdout was:", dataString);
            return res.status(500).json({ error: 'Failed to process image through ML model', raw: dataString, status: 'Failed' });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Node.js/Express server is running on http://localhost:${PORT}`);
});
