# Email Spoofing Tool

A Flask-based web application for email composition and management with AI-powered features.

## Features

- 📧 Email Composition with Customizable Templates
- 🤖 AI-Powered Subject Line Suggestions
- 📝 Content Analysis and Improvement Recommendations
- 🔍 Phishing Detection
- 📊 Email Statistics Dashboard
- 📁 Template Management System
- 📜 Email History Tracking

## Prerequisites

- Python 3.8 or higher
- Gmail account with 2-Step Verification enabled
- OpenAI API key (for AI features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spoofing.git
cd email-spoofing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the following variables in `.env`:
     - `EMAIL_USER`: Your Gmail address
     - `EMAIL_PASSWORD`: Your Gmail App Password
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `SECRET_KEY`: Your Flask secret key

4. Create Gmail App Password:
   - Enable 2-Step Verification in your Google Account
   - Go to Google Account Settings > Security > App Passwords
   - Generate a new App Password for "Mail"
   - Copy the 16-character password to your `.env` file

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Features available:
   - Compose and send emails
   - Get AI-powered subject suggestions
   - Analyze email content
   - Manage email templates
   - View sending history
   - Check email statistics

## Security Features

- Email authentication using Gmail's SMTP
- Secure credential management
- Phishing detection
- Input validation and sanitization

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational purposes only. Do not use it for malicious purposes or spam.