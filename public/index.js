const express = require('express');
const multer = require('multer');
const path = require('path');
const { exec } = require('child_process');

const app = express();

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Configure multer to use disk storage
const storage = multer.diskStorage({
    destination: 'uploads/',
    filename: (req, file, cb) => {
        cb(null, file.originalname); // Keep the original filename
    }
});

// Filter files based on their extension
const fileFilter = (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext === '.wav' || ext === '.mp3' || ext === '.mp4') {
        cb(null, true); // Accept the file
    } else {
        cb(new Error('Invalid file type. Please upload a WAV, MP3, or MP4 file.'), false); // Reject the file
    }
};

const upload = multer({ storage, fileFilter });

// Route to serve the HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/upload', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).send('No file uploaded.');
    }

    // Path to the uploaded file
    const filePath = path.join(__dirname, 'uploads', req.file.originalname);

    // Path to the Python script (resources/process_audio.py at the root level)
    const pythonScriptPath = path.join(__dirname, '..', 'resources', 'process_audio.py');

    // Execute the Python script to process the uploaded file
    exec(`python3 "${pythonScriptPath}" "${filePath}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing Python script: ${error}`);
            console.error(`stderr: ${stderr}`);
            return res.status(500).send('Error processing the file.');
        }

        const result = stdout.trim();
        res.send(`File uploaded and processed successfully. Result: ${result}`);
    });
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});