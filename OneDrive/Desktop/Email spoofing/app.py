import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from flask_sqlalchemy import SQLAlchemy
from wtforms import StringField, TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import formataddr
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# Load environment variables first
load_dotenv()

# Initialize OpenAI client after loading environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")
client = OpenAI(api_key=api_key)

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    print("NLTK download failed, but continuing anyway")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///emails.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Database models
class EmailTemplate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SentEmail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_email = db.Column(db.String(100), nullable=False)
    recipient_email = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    body = db.Column(db.Text, nullable=False)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    success = db.Column(db.Boolean, default=True)
    error_message = db.Column(db.Text, nullable=True)

# Forms
class EmailForm(FlaskForm):
    sender_name = StringField('Sender Name', validators=[DataRequired()])
    sender_email = StringField('Sender Email', validators=[DataRequired(), Email()])
    recipient_email = StringField('Recipient Email', validators=[DataRequired(), Email()])
    subject = StringField('Subject', validators=[DataRequired()])
    body = TextAreaField('Email Body', validators=[DataRequired()])
    template = SelectField('Use Template', choices=[('0', 'None')], coerce=int)
    submit = SubmitField('Send Email')

class TemplateForm(FlaskForm):
    name = StringField('Template Name', validators=[DataRequired()])
    subject = StringField('Subject', validators=[DataRequired()])
    body = TextAreaField('Email Body', validators=[DataRequired()])
    submit = SubmitField('Save Template')

# AI Functions
def get_gpt_suggestions(content, task="improve"):
    """Get suggestions from GPT for email improvement"""
    try:
        if task == "improve":
            prompt = f"""Analyze this email content and provide suggestions for improvement:

            {content}

            Please provide:
            1. Writing style improvements
            2. Professional tone suggestions
            3. Clarity enhancements
            4. Any potential issues to address"""
        elif task == "subject":
            prompt = f"""Generate 3 engaging subject lines for this email content:

            {content}

            The subject lines should be:
            1. Professional and clear
            2. Attention-grabbing but not clickbait
            3. Relevant to the content"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert email writing assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT API error: {str(e)}")
        # Provide fallback suggestions based on the task
        if task == "subject":
            words = content.split()[:5]  # Get first 5 words
            summary = " ".join(words) + "..." if len(words) == 5 else " ".join(words)
            return "\n".join([
                f"Re: {summary}",
                "Important: " + summary,
                "Update regarding " + summary
            ])
        else:
            return """Here are some general suggestions for improvement:
1. Keep your message clear and concise
2. Use professional language
3. Double-check for spelling and grammar
4. Include a clear call to action
5. Review before sending"""

def analyze_email_content(content):
    """Analyze email content using GPT and traditional metrics"""
    try:
        # Get GPT analysis
        gpt_analysis = get_gpt_suggestions(content, "improve")

        # Traditional metrics
        tokens = word_tokenize(content.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]

        word_count = len(filtered_tokens)
        avg_word_length = np.mean([len(word) for word in filtered_tokens]) if filtered_tokens else 0

        return {
            "gpt_analysis": gpt_analysis,
            "word_count": word_count,
            "avg_word_length": round(avg_word_length, 2),
            "recommendations": gpt_analysis.split("\n") if "\n" in gpt_analysis else [gpt_analysis]
        }
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "gpt_analysis": "Unable to generate AI analysis at this time. Please try again later.",
            "word_count": len(content.split()),
            "avg_word_length": round(sum(len(word) for word in content.split()) / len(content.split()) if content.split() else 0, 2),
            "recommendations": [
                "Keep your message clear and concise",
                "Use professional language",
                "Double-check for spelling and grammar",
                "Include a clear call to action",
                "Review before sending"
            ]
        }

def generate_subject_suggestions(content):
    """Generate subject line suggestions using GPT"""
    try:
        suggestions = get_gpt_suggestions(content, "subject")
        return [s.strip() for s in suggestions.split("\n") if s.strip()]
    except Exception as e:
        logger.error(f"Subject generation error: {str(e)}")
        # Provide basic fallback suggestions
        words = content.split()[:5]  # Get first 5 words
        summary = " ".join(words) + "..." if len(words) == 5 else " ".join(words)
        return [
            f"Re: {summary}",
            f"Important: {summary}",
            f"Update regarding {summary}"
        ]

def detect_phishing_patterns(content):
    """Detect common phishing patterns in email content"""
    phishing_indicators = [
        "urgent action required",
        "verify your account",
        "update your payment",
        "click here to avoid",
        "suspicious activity",
        "login to confirm",
        "your account will be suspended",
        "confirm your identity",
        "unusual login attempt"
    ]

    content_lower = content.lower()
    detected = []

    for indicator in phishing_indicators:
        if indicator in content_lower:
            detected.append(indicator)

    risk_score = min(len(detected) * 20, 100)  # 20% per indicator, max 100%

    return {
        "detected_patterns": detected,
        "risk_score": risk_score,
        "is_potential_phishing": risk_score > 40
    }

def cluster_templates():
    """Cluster email templates to find patterns"""
    templates = EmailTemplate.query.all()
    if len(templates) < 3:
        return {"message": "Need at least 3 templates for clustering"}

    # Extract text
    texts = [f"{t.subject} {t.body}" for t in templates]

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(texts)

    # Determine optimal number of clusters (simplified)
    n_clusters = min(3, len(texts) - 1)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Prepare results
    result = []
    for i, cluster_id in enumerate(clusters):
        result.append({
            "template_id": templates[i].id,
            "template_name": templates[i].name,
            "cluster": int(cluster_id)
        })

    return result

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    form = EmailForm()

    # Populate template choices
    templates = EmailTemplate.query.all()
    form.template.choices = [('0', 'None')] + [(str(t.id), t.name) for t in templates]

    if form.validate_on_submit():
        try:
            # Validate SMTP settings
            smtp_server = os.getenv('SMTP_SERVER')
            smtp_port = os.getenv('SMTP_PORT')
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')

            if not all([smtp_server, smtp_port, email_user, email_password]):
                flash('SMTP settings are not properly configured. Please check your .env file.', 'error')
                return redirect(url_for('index'))

            # Create message
            msg = MIMEMultipart()
            msg['From'] = formataddr((form.sender_name.data, form.sender_email.data))
            msg['To'] = form.recipient_email.data
            msg['Subject'] = form.subject.data

            # Ensure body is not None and properly encoded
            body = form.body.data or ''  # Use empty string if body is None
            msg.attach(MIMEText(body, 'html', 'utf-8'))

            # Connect to SMTP server
            server = smtplib.SMTP(smtp_server, int(smtp_port))
            server.starttls()
            server.login(email_user, email_password)

            # Send email
            server.send_message(msg)
            server.quit()

            # Log the sent email
            sent_email = SentEmail(
                sender_email=form.sender_email.data,
                recipient_email=form.recipient_email.data,
                subject=form.subject.data,
                body=body,
                success=True
            )
            db.session.add(sent_email)
            db.session.commit()

            flash('Email sent successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            error_message = str(e)
            logger.error(f"Email sending failed: {error_message}")

            # Log the failed attempt
            sent_email = SentEmail(
                sender_email=form.sender_email.data,
                recipient_email=form.recipient_email.data,
                subject=form.subject.data,
                body=form.body.data or '',
                success=False,
                error_message=error_message
            )
            db.session.add(sent_email)
            db.session.commit()

            flash(f'Error sending email: {error_message}', 'error')

    return render_template('index.html', form=form)

@app.route('/templates', methods=['GET', 'POST'])
def templates():
    form = TemplateForm()
    if form.validate_on_submit():
        template = EmailTemplate(
            name=form.name.data,
            subject=form.subject.data,
            body=form.body.data
        )
        db.session.add(template)
        db.session.commit()
        flash('Template saved successfully!', 'success')
        return redirect(url_for('templates'))

    templates = EmailTemplate.query.all()
    return render_template('templates.html', form=form, templates=templates)

@app.route('/template/<int:id>', methods=['GET'])
def get_template(id):
    template = EmailTemplate.query.get_or_404(id)
    return jsonify({
        'name': template.name,
        'subject': template.subject,
        'body': template.body
    })

@app.route('/template/<int:id>', methods=['DELETE'])
def delete_template(id):
    template = EmailTemplate.query.get_or_404(id)
    db.session.delete(template)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/analyze', methods=['POST'])
def analyze():
    content = request.json.get('content', '')
    analysis = analyze_email_content(content)
    phishing_check = detect_phishing_patterns(content)
    subject_suggestions = generate_subject_suggestions(content)

    return jsonify({
        'content_analysis': analysis,
        'phishing_check': phishing_check,
        'subject_suggestions': subject_suggestions
    })

@app.route('/history')
def history():
    emails = SentEmail.query.order_by(SentEmail.sent_at.desc()).all()
    return render_template('history.html', emails=emails)

@app.route('/dashboard')
def dashboard():
    # Get statistics
    total_emails = SentEmail.query.count()
    successful_emails = SentEmail.query.filter_by(success=True).count()
    failed_emails = SentEmail.query.filter_by(success=False).count()

    # Get template clusters if enough templates
    template_clusters = cluster_templates() if EmailTemplate.query.count() >= 3 else None

    # Get recent emails
    recent_emails = SentEmail.query.order_by(SentEmail.sent_at.desc()).limit(5).all()

    return render_template('dashboard.html',
                          total_emails=total_emails,
                          successful_emails=successful_emails,
                          failed_emails=failed_emails,
                          template_clusters=template_clusters,
                          recent_emails=recent_emails)

@app.route('/api/email-stats')
def email_stats():
    # Get email sending statistics by day
    emails = SentEmail.query.all()
    df = pd.DataFrame([{
        'date': email.sent_at.strftime('%Y-%m-%d'),
        'success': email.success
    } for email in emails])

    if df.empty:
        return jsonify({'dates': [], 'success_counts': [], 'failure_counts': []})

    # Group by date and success
    grouped = df.groupby(['date', 'success']).size().unstack(fill_value=0)

    # Prepare data for chart
    dates = grouped.index.tolist()
    success_counts = grouped.get(True, pd.Series([0] * len(dates))).tolist()
    failure_counts = grouped.get(False, pd.Series([0] * len(dates))).tolist()

    return jsonify({
        'dates': dates,
        'success_counts': success_counts,
        'failure_counts': failure_counts
    })

@app.route('/test-gpt')
def test_gpt():
    try:
        # Test the OpenAI integration
        test_content = "This is a test email to verify the integration."
        suggestions = get_gpt_suggestions(test_content, "improve")
        return jsonify({
            'status': 'success',
            'message': 'OpenAI integration is working!',
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'OpenAI integration error: {str(e)}'
        })

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)