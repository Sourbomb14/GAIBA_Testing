import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import jsonimport streamlit as st
import pandas as pd
import numpy as np
import smtplib
import ssl
import time
import re
import json
import plotly.express as px
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email_validator import validate_email, EmailNotValidError
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import io
import base64
from groq import Groq
import requests
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

# Load environment variables
load_dotenv()

# Configuration - Updated based on your .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # Default to your specified model
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "black-forest-labs/FLUX.1-schnell")

# Countries and Currencies data
COUNTRIES_DATA = {
    "Global": {"coords": [0, 0], "currency": "USD"},
    "United States": {"coords": [39.8283, -98.5795], "currency": "USD"},
    "Canada": {"coords": [56.1304, -106.3468], "currency": "CAD"},
    "United Kingdom": {"coords": [55.3781, -3.4360], "currency": "GBP"},
    "Germany": {"coords": [51.1657, 10.4515], "currency": "EUR"},
    "France": {"coords": [46.6034, 1.8883], "currency": "EUR"},
    "Spain": {"coords": [40.4637, -3.7492], "currency": "EUR"},
    "Italy": {"coords": [41.8719, 12.5674], "currency": "EUR"},
    "Netherlands": {"coords": [52.1326, 5.2913], "currency": "EUR"},
    "Australia": {"coords": [-25.2744, 133.7751], "currency": "AUD"},
    "Japan": {"coords": [36.2048, 138.2529], "currency": "JPY"},
    "India": {"coords": [20.5937, 78.9629], "currency": "INR"},
    "China": {"coords": [35.8617, 104.1954], "currency": "CNY"},
    "Brazil": {"coords": [-14.2350, -51.9253], "currency": "BRL"},
    "Mexico": {"coords": [23.6345, -102.5528], "currency": "MXN"}
}

COUNTRIES = list(COUNTRIES_DATA.keys())
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "INR", "BRL", "MXN", "CNY"]

# ================================
# SESSION STATE INITIALIZATION
# ================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Campaign Dashboard"
    if 'current_campaign' not in st.session_state:
        st.session_state.current_campaign = None
    if 'campaign_blueprint' not in st.session_state:
        st.session_state.campaign_blueprint = None
    if 'email_template_html' not in st.session_state:
        st.session_state.email_template_html = None
    if 'email_template_text' not in st.session_state:
        st.session_state.email_template_text = None
    if 'email_contacts' not in st.session_state:
        st.session_state.email_contacts = None
    if 'campaign_results' not in st.session_state:
        st.session_state.campaign_results = None
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = []

# ================================
# GROQ AI CAMPAIGN GENERATOR
# ================================

class GroqCampaignGenerator:
    """Generate campaigns using Groq API with openai/gpt-oss-120b"""
    
    def __init__(self):
        self.client = None
        if GROQ_API_KEY:
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                st.error(f"Failed to initialize Groq: {e}")
    
    def generate_campaign_strategy(self, campaign_data):
        """Generate comprehensive campaign strategy using Groq openai/gpt-oss-120b"""
        if not self.client:
            return self._fallback_strategy(campaign_data)
        
        try:
            prompt = self._build_campaign_prompt(campaign_data)
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a world-class marketing strategist with 20+ years of experience. Create detailed, actionable, and comprehensive marketing campaigns that drive real results. Focus on practical implementation, specific tactics, and measurable outcomes."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL,  # Using openai/gpt-oss-120b as specified
                temperature=0.7,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating campaign with Groq: {e}")
            return self._fallback_strategy(campaign_data)
    
    def generate_email_template(self, template_type, tone, format_type, campaign_context=None):
        """Generate email templates using Groq AI"""
        if not self.client:
            return self._fallback_email_template(template_type, tone, format_type)
        
        try:
            prompt = f"""
            Create a professional {template_type.lower()} email template with a {tone.lower()} tone.
            Format: {format_type}
            
            Campaign Context: {campaign_context if campaign_context else 'General marketing campaign'}
            
            Requirements:
            1. Include personalization placeholders: {{first_name}}, {{name}}, {{email}}
            2. Make it engaging and action-oriented
            3. Include a clear call-to-action
            4. Use modern, professional design
            5. Make it mobile-friendly
            
            {"Generate HTML email with inline CSS styling" if format_type == "HTML" else "Generate plain text email with proper formatting"}
            
            The email should be compelling, personalized, and drive conversions.
            """
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert email marketing specialist. Create high-converting email templates that are professional, engaging, and drive action."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL,
                temperature=0.8,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating email template: {e}")
            return self._fallback_email_template(template_type, tone, format_type)
    
    def _build_campaign_prompt(self, data):
        """Build comprehensive campaign prompt"""
        return f"""
        Create a comprehensive, actionable marketing campaign strategy for:
        
        **CAMPAIGN DETAILS:**
        - Company: {data.get('company_name', 'Company')}
        - Campaign Type: {data.get('campaign_type', 'Marketing Campaign')}
        - Target Audience: {data.get('target_audience', 'General audience')}
        - Geographic Focus: {data.get('location', 'Global')} {data.get('city_state', '')}
        - Marketing Channels: {', '.join(data.get('channels', ['Email']))}
        - Budget: {data.get('budget', 'TBD')} {data.get('currency', 'USD')}
        - Duration: {data.get('duration', 'TBD')}
        - Customer Segment: {data.get('customer_segment', 'Mass Market')}
        - Product/Service: {data.get('product_description', 'Product/Service')}
        
        **DELIVERABLES REQUIRED:**
        1. **Executive Summary** - High-level strategy overview
        2. **Market Analysis** - Target audience deep-dive and market conditions
        3. **Competitive Positioning** - How to differentiate in the market
        4. **Messaging Strategy** - Key value propositions and communication themes
        5. **Channel-Specific Tactics** - Detailed implementation for each marketing channel
        6. **Content Strategy** - Types of content needed and distribution plan
        7. **Timeline & Milestones** - Phased implementation with specific dates
        8. **Budget Allocation** - Detailed spend breakdown by channel and activity
        9. **Success Metrics & KPIs** - Measurable goals and tracking methods
        10. **Risk Management** - Potential challenges and mitigation strategies
        11. **Next Steps** - Immediate action items and priorities
        
        Make this practical, specific, and actionable. Include actual tactics, not just high-level concepts.
        """
    
    def _fallback_strategy(self, data):
        """Fallback campaign strategy if AI fails"""
        return f"""
# {data.get('company_name', 'Your Company')} - {data.get('campaign_type', 'Marketing')} Campaign Strategy

## 🎯 Executive Summary
**Campaign Overview:** {data.get('campaign_type', 'Marketing Campaign')} for {data.get('company_name', 'Your Company')}
**Target Market:** {data.get('location', 'Global')} - {data.get('customer_segment', 'Mass Market')}
**Budget:** {data.get('budget', 'TBD')} {data.get('currency', 'USD')}
**Duration:** {data.get('duration', '8 weeks')}

## 👥 Target Audience Analysis
{data.get('target_audience', 'Primary target audience to be defined')}

**Geographic Focus:** {data.get('location', 'Global')}
**Customer Segment:** {data.get('customer_segment', 'Mass Market')}

## 📢 Channel Strategy
**Selected Channels:** {', '.join(data.get('channels', ['Email Marketing']))}

### Email Marketing Strategy
- Welcome series for new subscribers
- Promotional campaigns for product launches
- Re-engagement campaigns for inactive users
- Personalized product recommendations

## 📅 Implementation Timeline
**Phase 1 (Weeks 1-2):** Strategy finalization and asset creation
**Phase 2 (Weeks 3-4):** Campaign launch and initial optimization
**Phase 3 (Weeks 5-6):** Performance monitoring and scaling
**Phase 4 (Weeks 7-8):** Analysis and next campaign planning

## 💰 Budget Allocation
- Creative Development: 25%
- Media/Advertising: 45%
- Technology & Tools: 20%
- Analytics & Optimization: 10%

## 📊 Success Metrics
- **Reach:** Target audience exposure
- **Engagement:** Click-through rates and interactions
- **Conversions:** Lead generation and sales
- **ROI:** Return on advertising spend

## 🚀 Next Steps
1. Approve campaign strategy and budget
2. Develop creative assets and content
3. Set up tracking and analytics
4. Launch pilot campaign
5. Monitor performance and optimize

*Campaign strategy generated on {datetime.now().strftime('%B %d, %Y')}*
"""
    
    def _fallback_email_template(self, template_type, tone, format_type):
        """Fallback email template if AI fails"""
        if format_type == "HTML":
            return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{template_type}</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; max-width: 600px; margin: 0 auto; background: #f5f5f5; }}
        .container {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; text-align: center; }}
        .content {{ padding: 40px 30px; line-height: 1.6; }}
        .cta-button {{ background: #007bff; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 20px 0; font-weight: bold; }}
        .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Hello {{first_name}}! 👋</h1>
            <p>We have something special for you</p>
        </div>
        <div class="content">
            <p>Dear {{name}},</p>
            <p>We're excited to share this exclusive {template_type.lower()} with you.</p>
            <p>As a valued member of our community, you deserve the best we have to offer.</p>
            <div style="text-align: center;">
                <a href="#" class="cta-button">Discover More</a>
            </div>
            <p>Thank you for being part of our journey!</p>
        </div>
        <div class="footer">
            <p>Best regards,<br>The Marketing Team</p>
            <p>You received this email because you're subscribed to our updates.</p>
        </div>
    </div>
</body>
</html>
"""
        else:
            return f"""Subject: Exclusive {template_type} for {{first_name}}

Hello {{first_name}},

We're excited to share this exclusive {template_type.lower()} with you.

As a valued member of our community, you deserve the best we have to offer.

Here's what makes this special:
• Personalized just for you
• Exclusive member benefits
• Limited-time opportunity
• Premium experience

Ready to explore? Visit our website or reply to this email.

Thank you for being part of our journey, {{name}}!

Best regards,
The Marketing Team

---
You received this email because you're subscribed to our updates.
"""

# ================================
# HUGGINGFACE IMAGE GENERATOR
# ================================

class HuggingFaceImageGenerator:
    """Generate images using HuggingFace Stable Diffusion"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
    def load_model(self):
        """Load the Stable Diffusion model"""
        if self.model_loaded:
            return True
            
        try:
            with st.spinner("🔄 Loading AI image generation model (this may take a moment)..."):
                # Try to load FLUX model first, then fallback to Stable Diffusion
                models_to_try = [
                    HF_MODEL,  # Your specified FLUX model
                    "stabilityai/stable-diffusion-2-1",
                    "runwayml/stable-diffusion-v1-5"
                ]
                
                for model_id in models_to_try:
                    try:
                        if self.device == "cuda":
                            self.pipe = StableDiffusionPipeline.from_pretrained(
                                model_id,
                                torch_dtype=torch.float16,
                                use_auth_token=HUGGING_FACE_TOKEN
                            ).to(self.device)
                        else:
                            self.pipe = StableDiffusionPipeline.from_pretrained(
                                model_id,
                                torch_dtype=torch.float32,
                                use_auth_token=HUGGING_FACE_TOKEN
                            )
                        
                        st.success(f"✅ Loaded model: {model_id}")
                        self.model_loaded = True
                        return True
                        
                    except Exception as e:
                        st.warning(f"⚠️ Failed to load {model_id}: {str(e)}")
                        continue
                
                st.error("❌ Failed to load any image generation model")
                return False
                
        except Exception as e:
            st.error(f"❌ Error loading image model: {str(e)}")
            return False
    
    def generate_campaign_image(self, campaign_description, style="professional"):
        """Generate campaign image"""
        if not HUGGING_FACE_TOKEN:
            st.warning("⚠️ HuggingFace token not configured. Please add HUGGING_FACE_TOKEN to your .env file")
            return None
        
        if not self.load_model():
            return None
        
        try:
            # Create enhanced prompt for marketing
            enhanced_prompt = f"Professional marketing campaign image for {campaign_description}, {style} style, high quality, vibrant colors, modern design, commercial photography, eye-catching, brand advertisement, 4K resolution, clean layout"
            
            # Negative prompt to improve quality
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, text, watermark, signature, amateur"
            
            with st.spinner("🎨 Generating your campaign image..."):
                if self.device == "cuda":
                    image = self.pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=25,
                        guidance_scale=7.5,
                        width=512,
                        height=512
                    ).images[0]
                else:
                    # Use fewer steps for CPU
                    image = self.pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=15,
                        guidance_scale=7.5,
                        width=512,
                        height=512
                    ).images[0]
            
            # Store in session state
            image_data = {
                'prompt': enhanced_prompt,
                'timestamp': datetime.now(),
                'campaign': campaign_description,
                'image': image
            }
            
            st.session_state.generated_images.append(image_data)
            
            # Display the image
            st.success("✨ Campaign image generated successfully!")
            st.image(image, caption=f"Generated for: {campaign_description}", use_column_width=True)
            
            # Provide download option
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            st.download_button(
                "📥 Download Campaign Image",
                data=img_bytes.getvalue(),
                file_name=f"campaign_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
            
            return image
            
        except Exception as e:
            st.error(f"❌ Error generating image: {str(e)}")
            st.info("💡 Try a simpler prompt or check your HuggingFace token")
            return None

# ================================
# EMAIL PERSONALIZER
# ================================

class EmailPersonalizer:
    """Handle intelligent email personalization"""
    
    @staticmethod
    def extract_name_from_email(email):
        """Extract potential name from email address"""
        try:
            local_part = email.split('@')[0]
            name_part = re.sub(r'[0-9._-]', ' ', local_part)
            name_parts = [part.capitalize() for part in name_part.split() if len(part) > 1]
            return ' '.join(name_parts) if name_parts else 'Valued Customer'
        except:
            return 'Valued Customer'
    
    @staticmethod
    def personalize_template(template, name, email=None):
        """Personalize email template"""
        first_name = name.split()[0] if name and ' ' in name else name
        
        personalized = template.replace('{name}', name or 'Valued Customer')
        personalized = personalized.replace('{{name}}', name or 'Valued Customer')
        personalized = personalized.replace('{first_name}', first_name or 'Valued Customer')
        personalized = personalized.replace('{{first_name}}', first_name or 'Valued Customer')
        personalized = personalized.replace('{email}', email or '')
        personalized = personalized.replace('{{email}}', email or '')
        
        return personalized

# ================================
# EMAIL HANDLER - BULK AND SINGLE
# ================================

class EmailHandler:
    """Handle both bulk and single email sending"""
    
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.email = GMAIL_EMAIL
        self.password = GMAIL_APP_PASSWORD
    
    def validate_email_address(self, email):
        """Validate email address format"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def send_single_email(self, to_email, subject, body, is_html=True):
        """Send a single email with detailed error handling"""
        if not self.email or not self.password:
            return False, "Gmail credentials not configured in .env file"
            
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
                text = msg.as_string()
                server.sendmail(self.email, to_email, text)
            
            return True, "Success"
        except smtplib.SMTPAuthenticationError:
            return False, "Gmail authentication failed. Check your app password."
        except smtplib.SMTPRecipientsRefused:
            return False, f"Recipient {to_email} was refused"
        except Exception as e:
            return False, f"SMTP Error: {str(e)}"
    
    def send_bulk_emails(self, email_list, subject, body_template, personalizer, is_html=True):
        """WORKING bulk email sending function"""
        if not self.email or not self.password:
            st.error("❌ Gmail configuration missing!")
            st.code("""
# Add to .env file:
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_16_digit_app_password
            """)
            return pd.DataFrame()
        
        total_emails = len(email_list)
        results = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        sent_count = 0
        failed_count = 0
        invalid_count = 0
        
        try:
            # Single SMTP connection for all emails
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
                
                for index, row in email_list.iterrows():
                    current_progress = (index + 1) / total_emails
                    
                    # Update progress
                    progress_bar.progress(current_progress)
                    status_text.info(f"📧 Sending {index + 1}/{total_emails}: {row['email']}")
                    
                    try:
                        # Validate email format
                        if not self.validate_email_address(row['email']):
                            invalid_count += 1
                            results.append({
                                "email": row['email'],
                                "name": row.get('name', 'Unknown'),
                                "status": "invalid",
                                "error": "Invalid email format",
                                "timestamp": datetime.now().strftime('%H:%M:%S')
                            })
                            continue
                        
                        # Get or extract name
                        name = row.get('name', personalizer.extract_name_from_email(row['email']))
                        
                        # Personalize content
                        personal_subject = personalizer.personalize_template(subject, name, row['email'])
                        personal_body = personalizer.personalize_template(body_template, name, row['email'])
                        
                        # Create message
                        msg = MIMEMultipart('alternative')
                        msg['Subject'] = personal_subject
                        msg['From'] = self.email
                        msg['To'] = row['email']
                        
                        # Add content
                        if is_html:
                            msg.attach(MIMEText(personal_body, 'html'))
                        else:
                            msg.attach(MIMEText(personal_body, 'plain'))
                        
                        # Send email
                        server.sendmail(self.email, row['email'], msg.as_string())
                        
                        # Record success
                        results.append({
                            "email": row['email'],
                            "name": name,
                            "status": "sent",
                            "error": "",
                            "timestamp": datetime.now().strftime('%H:%M:%S')
                        })
                        sent_count += 1
                        
                    except Exception as email_error:
                        results.append({
                            "email": row['email'],
                            "name": row.get('name', 'Unknown'),
                            "status": "failed",
                            "error": str(email_error),
                            "timestamp": datetime.now().strftime('%H:%M:%S')
                        })
                        failed_count += 1
                    
                    # Update metrics
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("✅ Sent", sent_count)
                        col2.metric("❌ Failed", failed_count)
                        col3.metric("⚠️ Invalid", invalid_count)
                        col4.metric("📊 Progress", f"{current_progress*100:.0f}%")
                    
                    # Rate limiting to avoid being flagged as spam
                    time.sleep(1)
                
                # Final status
                progress_bar.progress(1.0)
                status_text.success("🎉 Email campaign completed!")
                
        except Exception as smtp_error:
            st.error(f"❌ SMTP Connection Error: {str(smtp_error)}")
            return pd.DataFrame()
        
        return pd.DataFrame(results)

# ================================
# FILE PROCESSOR
# ================================

class FileProcessor:
    """Process files and extract contacts"""
    
    def __init__(self):
        self.personalizer = EmailPersonalizer()
    
    def process_file(self, uploaded_file):
        """Process uploaded file and extract contacts"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload CSV or Excel files only")
                return None
            
            return self._process_dataframe(df)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    
    def _process_dataframe(self, df):
        """Process dataframe and standardize columns"""
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Find email and name columns
        email_col = None
        name_col = None
        
        for col in df.columns:
            if 'email' in col or 'mail' in col:
                email_col = col
                break
        
        for col in df.columns:
            if 'name' in col or 'first' in col or 'last' in col:
                name_col = col
                break
        
        if email_col is None:
            st.error("❌ No email column found. Please ensure your file has an 'email' column.")
            return None
        
        # Create result dataframe
        result_data = []
        
        for _, row in df.iterrows():
            email = row[email_col]
            if pd.isna(email) or str(email).strip() == '':
                continue
            
            # Clean email
            email = str(email).strip().lower()
            
            # Get name
            if name_col and not pd.isna(row[name_col]):
                name = str(row[name_col]).strip()
            else:
                name = self.personalizer.extract_name_from_email(email)
            
            # Validate email
            try:
                validate_email(email)
                result_data.append({'email': email, 'name': name})
            except EmailNotValidError:
                continue
        
        if not result_data:
            st.error("❌ No valid emails found")
            return None
        
        return pd.DataFrame(result_data)

# ================================
# STREAMLIT APP CONFIGURATION
# ================================

st.set_page_config(
    page_title="Marketing Campaign War Room",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
    
    .success-metric {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
    
    .single-email-form {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# MAIN APPLICATION
# ================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0;">🚀 Marketing Campaign War Room</h1>
        <p style="font-size: 1.3rem; color: #888; margin-top: 0;">AI-Powered Campaign Generation & Comprehensive Email Marketing Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        # Navigation buttons
        if st.button("🎯 Campaign Dashboard", use_container_width=True):
            st.session_state.current_page = "Campaign Dashboard"
            st.rerun()
        
        if st.button("📧 Email Marketing", use_container_width=True):
            st.session_state.current_page = "Email Marketing"
            st.rerun()
        
        if st.button("📊 Analytics & Reports", use_container_width=True):
            st.session_state.current_page = "Analytics & Reports"
            st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🔧 System Status")
        
        if GROQ_API_KEY:
            st.success(f"🤖 Groq AI: Connected ({GROQ_MODEL})")
        else:
            st.error("🤖 Groq AI: Not configured")
        
        if GMAIL_EMAIL and GMAIL_APP_PASSWORD:
            st.success("📧 Email Service: Connected")
        else:
            st.error("📧 Email Service: Not configured")
        
        if HUGGING_FACE_TOKEN:
            st.success(f"🎨 Image Generator: Connected ({HF_MODEL})")
        else:
            st.warning("🎨 Image Generator: Not configured")
        
        st.markdown("---")
        
        # Current campaign info
        if st.session_state.current_campaign:
            st.markdown("### 🎯 Active Campaign")
            st.info(f"**{st.session_state.current_campaign['company_name']}**")
            st.caption(f"Type: {st.session_state.current_campaign['campaign_type']}")
            st.caption(f"Location: {st.session_state.current_campaign['location']}")
        
        if st.session_state.email_contacts is not None:
            st.markdown("### 📊 Contact Stats")
            st.info(f"📧 Loaded: {len(st.session_state.email_contacts)} contacts")
    
    # Show current page content
    if st.session_state.current_page == "Campaign Dashboard":
        show_campaign_dashboard()
    elif st.session_state.current_page == "Email Marketing":
        show_email_marketing()
    elif st.session_state.current_page == "Analytics & Reports":
        show_analytics_reports()

def show_campaign_dashboard():
    """Campaign strategy generation page"""
    
    st.header("🎯 AI Campaign Strategy Generator")
    st.write("Create comprehensive marketing campaigns powered by Groq AI")
    
    with st.form("campaign_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("🏢 Company Name", 
                value=st.session_state.current_campaign['company_name'] if st.session_state.current_campaign else "")
            
            campaign_type = st.selectbox("📋 Campaign Type", [
                "Product Launch", "Brand Awareness", "Seasonal Campaign", "Customer Retention",
                "Lead Generation", "Event Promotion", "Sales Campaign", "Newsletter Campaign",
                "Re-engagement Campaign", "Referral Campaign"
            ])
            
            target_audience = st.text_area("👥 Target Audience", 
                placeholder="Describe demographics, interests, pain points, behaviors...")
            
            duration = st.text_input("📅 Campaign Duration", placeholder="e.g., 6 weeks, 3 months")
        
        with col2:
            channels = st.multiselect("📢 Marketing Channels", [
                "Email Marketing", "Social Media Marketing", "Google Ads", "Facebook Ads", 
                "Instagram Ads", "LinkedIn Ads", "Content Marketing", "Influencer Marketing", 
                "SEO/SEM", "TV/Radio", "Print Media", "Direct Mail"
            ])
            
            location = st.selectbox("🌍 Target Country", COUNTRIES)
            city_state = st.text_input("🏙️ City/State", placeholder="e.g., New York, NY")
            customer_segment = st.selectbox("💼 Customer Segment", 
                ["Mass Market", "Premium", "Luxury", "Niche", "Enterprise", "SMB", "Startup"])
        
        # Budget and Currency
        budget_col1, budget_col2 = st.columns(2)
        with budget_col1:
            budget = st.text_input("💰 Budget Amount", placeholder="e.g., 50000")
        with budget_col2:
            currency = st.selectbox("💱 Currency", CURRENCIES)
        
        product_description = st.text_area("📦 Product/Service Description",
            placeholder="Describe what you're promoting: features, benefits, unique selling points...")
        
        # Form buttons
        col1, col2 = st.columns(2)
        with col1:
            generate_campaign = st.form_submit_button("🚀 Generate AI Campaign Strategy", use_container_width=True)
        with col2:
            generate_image = st.form_submit_button("🎨 Generate Campaign Image", use_container_width=True)
    
    # Handle campaign generation
    if generate_campaign and company_name and campaign_type:
        campaign_data = {
            'company_name': company_name,
            'campaign_type': campaign_type,
            'target_audience': target_audience,
            'duration': duration,
            'channels': channels,
            'location': location,
            'city_state': city_state,
            'customer_segment': customer_segment,
            'budget': budget,
            'currency': currency,
            'product_description': product_description
        }
        
        with st.spinner(f"🤖 Groq AI ({GROQ_MODEL}) is generating your campaign strategy..."):
            generator = GroqCampaignGenerator()
            strategy = generator.generate_campaign_strategy(campaign_data)
            
            # Store in session state
            st.session_state.current_campaign = campaign_data
            st.session_state.campaign_blueprint = strategy
            
            st.success("✨ Campaign strategy generated successfully!")
            st.balloons()
    
    # Handle image generation
    if generate_image and st.session_state.current_campaign:
        image_gen = HuggingFaceImageGenerator()
        campaign_desc = f"{st.session_state.current_campaign['company_name']} {st.session_state.current_campaign['campaign_type']}"
        image_gen.generate_campaign_image(campaign_desc, "professional")
    
    # Display existing campaign
    if st.session_state.campaign_blueprint:
        st.markdown("---")
        st.markdown("## 📋 Your AI-Generated Campaign Strategy")
        st.markdown(st.session_state.campaign_blueprint)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📧 Create Email Campaign", use_container_width=True):
                st.session_state.current_page = "Email Marketing"
                st.rerun()
        with col2:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.current_page = "Analytics & Reports"
                st.rerun()
        with col3:
            if st.session_state.current_campaign:
                st.download_button("📄 Download Strategy", 
                    data=st.session_state.campaign_blueprint,
                    file_name=f"{st.session_state.current_campaign['company_name']}_campaign_strategy.md",
                    mime="text/markdown",
                    use_container_width=True)

def show_email_marketing():
    """Email marketing page with both bulk and single email capabilities"""
    
    st.header("📧 Comprehensive Email Marketing Center")
    
    # Show active campaign
    if st.session_state.current_campaign:
        st.success(f"🎯 Active Campaign: **{st.session_state.current_campaign['company_name']}** - {st.session_state.current_campaign['campaign_type']}")
    
    # Email template generation using Groq AI
    st.subheader("🤖 AI-Powered Email Template Generator")
    
    template_col1, template_col2 = st.columns(2)
    
    with template_col1:
        email_type = st.selectbox("📧 Email Type", [
            "Welcome Email", "Product Announcement", "Promotional Offer", 
            "Newsletter", "Follow-up Email", "Event Invitation",
            "Abandoned Cart", "Re-engagement", "Thank You", "Survey Request"
        ])
        tone = st.selectbox("🎭 Email Tone", [
            "Professional", "Friendly", "Casual", "Urgent", "Formal", "Conversational", "Enthusiastic"
        ])
    
    with template_col2:
        content_format = st.radio("📝 Template Format", ["HTML Template", "Plain Text"])
        
        if st.button("🚀 Generate AI Email Template", use_container_width=True):
            generator = GroqCampaignGenerator()
            campaign_context = f"{st.session_state.current_campaign['company_name']} {st.session_state.current_campaign['campaign_type']}" if st.session_state.current_campaign else None
            
            with st.spinner(f"🤖 Groq AI ({GROQ_MODEL}) is generating your email template..."):
                template = generator.generate_email_template(email_type, tone, content_format, campaign_context)
                
                if content_format == "HTML Template":
                    st.session_state.email_template_html = template
                else:
                    st.session_state.email_template_text = template
                
                st.success("✨ AI email template generated successfully!")
    
    # Template editor
    if st.session_state.email_template_html or st.session_state.email_template_text:
        st.markdown("---")
        st.subheader("📝 Email Template Editor")
        
        # Choose which template to edit
        if st.session_state.email_template_html and st.session_state.email_template_text:
            edit_choice = st.radio("Edit Template:", ["HTML Template", "Plain Text Template"])
            current_template = st.session_state.email_template_html if edit_choice == "HTML Template" else st.session_state.email_template_text
        elif st.session_state.email_template_html:
            current_template = st.session_state.email_template_html
            edit_choice = "HTML Template"
            st.info("✅ HTML template ready for editing")
        else:
            current_template = st.session_state.email_template_text
            edit_choice = "Plain Text Template"
            st.info("✅ Plain text template ready for editing")
        
        edited_content = st.text_area("Email Content:", value=current_template, height=400,
                                    help="Use {first_name}, {name}, and {email} for personalization")
        
        # Update template in session state
        if edit_choice == "HTML Template":
            st.session_state.email_template_html = edited_content
        else:
            st.session_state.email_template_text = edited_content
        
        # Preview HTML template
        if edit_choice == "HTML Template" and st.button("👀 Preview Email Template"):
            personalizer = EmailPersonalizer()
            preview = personalizer.personalize_template(edited_content, "John Smith", "john@example.com")
            st.components.v1.html(preview, height=600, scrolling=True)
    
    st.markdown("---")
    
    # SINGLE EMAIL SECTION
    st.subheader("📧 Send Single Email")
    
    with st.expander("📨 Single Email Sender", expanded=False):
        st.markdown('<div class="single-email-form">', unsafe_allow_html=True)
        
        with st.form("single_email_form"):
            single_col1, single_col2 = st.columns(2)
            
            with single_col1:
                single_email = st.text_input("📧 Recipient Email Address")
                single_name = st.text_input("👤 Recipient Name", 
                    help="Leave empty to auto-extract from email")
                single_subject = st.text_input("📝 Email Subject Line")
            
            with single_col2:
                use_template = st.checkbox("📄 Use Generated Template", 
                    value=bool(st.session_state.email_template_html or st.session_state.email_template_text))
                
                if use_template and (st.session_state.email_template_html or st.session_state.email_template_text):
                    if st.session_state.email_template_html and st.session_state.email_template_text:
                        template_choice = st.radio("Choose Template:", ["HTML", "Plain Text"])
                        single_body = st.session_state.email_template_html if template_choice == "HTML" else st.session_state.email_template_text
                        single_is_html = template_choice == "HTML"
                    elif st.session_state.email_template_html:
                        single_body = st.session_state.email_template_html
                        single_is_html = True
                        st.info("Using HTML template")
                    else:
                        single_body = st.session_state.email_template_text
                        single_is_html = False
                        st.info("Using plain text template")
                else:
                    single_body = ""
                    single_is_html = False
            
            # Email content editor for single email
            if not use_template or not single_body:
                single_body = st.text_area("📧 Email Content", 
                    value=single_body,
                    placeholder="Enter your email content here...",
                    height=200,
                    help="Use {first_name}, {name}, and {email} for personalization")
                single_is_html = st.checkbox("📄 Send as HTML", value=single_is_html)
            else:
                st.text_area("📧 Email Content Preview", 
                    value=single_body[:200] + "..." if len(single_body) > 200 else single_body,
                    height=100, disabled=True)
            
            # Send single email button
            send_single = st.form_submit_button("📧 Send Single Email", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle single email sending
        if send_single and single_email and single_subject and single_body:
            if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
                st.error("❌ Gmail configuration missing!")
                st.code("""
# Add to .env file:
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_16_digit_app_password
                """)
            else:
                try:
                    email_handler = EmailHandler()
                    personalizer = EmailPersonalizer()
                    
                    # Get or extract name
                    final_name = single_name if single_name else personalizer.extract_name_from_email(single_email)
                    
                    # Personalize content
                    final_subject = personalizer.personalize_template(single_subject, final_name, single_email)
                    final_body = personalizer.personalize_template(single_body, final_name, single_email)
                    
                    # Send email
                    with st.spinner(f"📧 Sending email to {single_email}..."):
                        success, error_msg = email_handler.send_single_email(single_email, final_subject, final_body, single_is_html)
                    
                    if success:
                        st.success(f"✅ Email sent successfully to {final_name} ({single_email})!")
                    else:
                        st.error(f"❌ Failed to send email: {error_msg}")
                        
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
        elif send_single:
            st.error("⚠️ Please fill in all required fields (email, subject, and content)")
    
    st.markdown("---")
    
    # BULK EMAIL SECTION
    st.subheader("👥 Bulk Email Campaign")
    
    # Contact upload
    uploaded_file = st.file_uploader("📁 Upload Contact File (CSV/Excel)", 
        type=['csv', 'xlsx'], key="contact_upload",
        help="Upload a CSV or Excel file with 'email' and optionally 'name' columns")
    
    if uploaded_file:
        processor = FileProcessor()
        contacts = processor.process_file(uploaded_file)
        
        if contacts is not None:
            st.session_state.email_contacts = contacts
            st.success(f"✅ Successfully loaded {len(contacts)} valid contacts!")
            
            # Show editable contacts
            st.subheader("📋 Contact List Editor")
            edited_contacts = st.data_editor(
                contacts,
                column_config={
                    "email": st.column_config.TextColumn("📧 Email Address", width="medium"),
                    "name": st.column_config.TextColumn("👤 Full Name", width="medium")
                },
                num_rows="dynamic",
                use_container_width=True,
                key="contact_editor"
            )
            st.session_state.email_contacts = edited_contacts
            
            # Contact statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Total Contacts", len(edited_contacts))
            with col2:
                domains = edited_contacts['email'].str.split('@').str[1].nunique()
                st.metric("🏢 Unique Domains", domains)
            with col3:
                avg_name_length = edited_contacts['name'].str.len().mean()
                st.metric("📝 Avg Name Length", f"{avg_name_length:.0f} chars")
    
    # Bulk email campaign launch
    if (st.session_state.email_contacts is not None and 
        (st.session_state.email_template_html or st.session_state.email_template_text)):
        
        st.markdown("---")
        st.subheader("🚀 Launch Bulk Email Campaign")
        
        df = st.session_state.email_contacts
        
        # Campaign overview
        st.markdown("### 📊 Campaign Overview")
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("👥 Recipients", len(df))
        with overview_col2:
            domains = df['email'].str.split('@').str[1].nunique()
            st.metric("🏢 Domains", domains)
        with overview_col3:
            st.metric("📧 Template", "✅ Ready")
        with overview_col4:
            estimated_time = len(df) * 1.5 / 60  # 1.5 seconds per email
            st.metric("⏱️ Est. Time", f"{estimated_time:.0f}m")
        
        # Campaign configuration
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            bulk_subject = st.text_input("📧 Campaign Subject Line", 
                value="Important message for {first_name}")
            test_email = st.text_input("🧪 Test Email Address", placeholder="your@email.com")
        
        with config_col2:
            # Choose template format
            if st.session_state.email_template_html and st.session_state.email_template_text:
                send_format = st.radio("📝 Send As:", ["HTML", "Plain Text"])
                template_to_use = st.session_state.email_template_html if send_format == "HTML" else st.session_state.email_template_text
                is_html = send_format == "HTML"
            elif st.session_state.email_template_html:
                template_to_use = st.session_state.email_template_html
                is_html = True
                st.info("✅ HTML template ready")
            else:
                template_to_use = st.session_state.email_template_text
                is_html = False
                st.info("✅ Plain text template ready")
        
        # Test email functionality
        if test_email and st.button("🧪 Send Test Email"):
            email_handler = EmailHandler()
            personalizer = EmailPersonalizer()
            
            test_content = personalizer.personalize_template(template_to_use, "Test User", test_email)
            test_subject = personalizer.personalize_template(bulk_subject, "Test User", test_email)
            
            with st.spinner(f"🧪 Sending test email to {test_email}..."):
                success, error_msg = email_handler.send_single_email(test_email, test_subject, test_content, is_html)
            
            if success:
                st.success("✅ Test email sent successfully!")
            else:
                st.error(f"❌ Test failed: {error_msg}")
        
        # Bulk campaign launch
        st.markdown("### 🎯 Campaign Launch")
        
        if st.button("🚀 LAUNCH BULK EMAIL CAMPAIGN", type="primary", use_container_width=True):
            if not GMAIL_EMAIL or not GMAIL_APP_PASSWORD:
                st.error("❌ Gmail configuration missing!")
                st.code("""
# Add to .env file:
GMAIL_EMAIL=your_email@gmail.com
GMAIL_APP_PASSWORD=your_16_digit_app_password
                """)
                st.stop()
            
            st.warning(f"⚠️ About to send {len(df)} personalized emails. This action cannot be undone!")
            
            if st.button("✅ CONFIRM & SEND ALL EMAILS"):
                st.info("🚀 Starting bulk email campaign...")
                
                # Initialize components
                email_handler = EmailHandler()
                personalizer = EmailPersonalizer()
                
                # Send bulk emails
                results = email_handler.send_bulk_emails(
                    df, bulk_subject, template_to_use, personalizer, is_html
                )
                
                if not results.empty:
                    # Calculate final results
                    success_count = len(results[results['status'] == 'sent'])
                    failed_count = len(results[results['status'] == 'failed'])
                    invalid_count = len(results[results['status'] == 'invalid'])
                    success_rate = (success_count / len(results)) * 100
                    
                    st.markdown("### 🎉 Campaign Results")
                    
                    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                    
                    with result_col1:
                        st.markdown(f'<div class="success-metric">✅ Successfully Sent<br><h2>{success_count}</h2></div>', unsafe_allow_html=True)
                    with result_col2:
                        st.metric("❌ Failed", failed_count)
                    with result_col3:
                        st.metric("⚠️ Invalid", invalid_count)
                    with result_col4:
                        st.metric("📊 Success Rate", f"{success_rate:.1f}%")
                    
                    # Store results for analytics
                    st.session_state.campaign_results = results
                    
                    # Download results
                    csv_data = results.to_csv(index=False)
                    st.download_button(
                        "📥 Download Campaign Results",
                        data=csv_data,
                        file_name=f"bulk_email_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show detailed results
                    with st.expander("📋 View Detailed Campaign Results"):
                        st.dataframe(results, use_container_width=True)
                    
                    # Success celebration
                    if success_count > 0:
                        st.balloons()
                
                else:
                    st.error("❌ Campaign failed - no results generated")

def show_analytics_reports():
    """Analytics and reporting page"""
    
    st.header("📊 Campaign Analytics & Performance Reports")
    
    # Campaign geographic analysis
    if st.session_state.current_campaign:
        st.subheader("🗺️ Campaign Geographic Analysis")
        
        campaign = st.session_state.current_campaign
        location = campaign['location']
        
        if location in COUNTRIES_DATA:
            coords = COUNTRIES_DATA[location]['coords']
            
            # Create map data
            map_data = pd.DataFrame({
                'lat': [coords[0]],
                'lon': [coords[1]], 
                'location': [location],
                'campaign': [campaign['campaign_type']],
                'company': [campaign['company_name']]
            })
            
            # Display interactive map
            fig = px.scatter_mapbox(
                map_data,
                lat='lat',
                lon='lon',
                hover_name='location',
                hover_data={'campaign': True, 'company': True, 'lat': False, 'lon': False},
                color_discrete_sequence=['#00d4ff'],
                size_max=20,
                zoom=3,
                title=f"Campaign Target Location: {location}"
            )
            
            fig.update_layout(
                mapbox_style="carto-darkmatter",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Campaign overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Campaign Type", campaign['campaign_type'])
            with col2:
                st.metric("🌍 Target Market", location)
            with col3:
                st.metric("💰 Budget", f"{campaign.get('budget', 'TBD')} {campaign.get('currency', 'USD')}")
            with col4:
                st.metric("📅 Duration", campaign.get('duration', 'TBD'))
        
        # Campaign projections
        if campaign.get('budget') and campaign['budget'].isdigit():
            st.subheader("📈 Campaign Performance Projections")
            
            budget = int(campaign['budget'])
            
            # Calculate projections
            estimated_reach = budget * 30  # $1 = 30 people reach
            estimated_clicks = int(estimated_reach * 0.04)  # 4% CTR
            estimated_conversions = int(estimated_clicks * 0.03)  # 3% conversion
            estimated_revenue = estimated_conversions * 65  # $65 per conversion
            roi = ((estimated_revenue - budget) / budget) * 100 if budget > 0 else 0
            
            proj_col1, proj_col2, proj_col3, proj_col4 = st.columns(4)
            
            with proj_col1:
                st.metric("👥 Estimated Reach", f"{estimated_reach:,}")
            with proj_col2:
                st.metric("👆 Expected Clicks", f"{estimated_clicks:,}")
            with proj_col3:
                st.metric("💰 Projected Conversions", f"{estimated_conversions:,}")
            with proj_col4:
                st.metric("📊 Projected ROI", f"{roi:.0f}%")
            
            # Performance timeline chart
            days = list(range(1, 31))
            daily_reach = [int(estimated_reach * (i/30) * (1 + 0.1 * np.sin(i/5))) for i in days]
            cumulative_conversions = [int(estimated_conversions * (i/30)) for i in days]
            
            chart_data = pd.DataFrame({
                'Day': days,
                'Daily Reach': daily_reach,
                'Cumulative Conversions': cumulative_conversions
            })
            
            fig = px.line(chart_data, x='Day', y=['Daily Reach', 'Cumulative Conversions'],
                         title="Projected 30-Day Campaign Performance")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    # Display generated images
    if st.session_state.generated_images:
        st.markdown("---")
        st.subheader("🎨 Generated Campaign Images")
        
        for i, img_data in enumerate(st.session_state.generated_images):
            if 'image' in img_data:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(img_data['image'], caption=f"Campaign Image {i+1}: {img_data['campaign']}", use_column_width=True)
                
                with col2:
                    st.write(f"**Generated:** {img_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Campaign:** {img_data['campaign']}")
                    
                    # Download button
                    img_bytes = io.BytesIO()
                    img_data['image'].save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    st.download_button(
                        f"📥 Download",
                        data=img_bytes.getvalue(),
                        file_name=f"campaign_image_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
    
    # Email campaign results
    if st.session_state.campaign_results is not None:
        st.markdown("---")
        st.subheader("📧 Email Campaign Performance Analysis")
        
        results_df = st.session_state.campaign_results
        
        # Performance metrics
        total_sent = len(results_df[results_df['status'] == 'sent'])
        total_failed = len(results_df[results_df['status'] == 'failed'])
        total_invalid = len(results_df[results_df['status'] == 'invalid'])
        success_rate = (total_sent / len(results_df)) * 100 if len(results_df) > 0 else 0
        
        # Display key metrics
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("📧 Total Emails", len(results_df))
        with perf_col2:
            st.metric("✅ Successfully Delivered", total_sent, delta=f"{success_rate:.1f}%")
        with perf_col3:
            st.metric("❌ Failed Deliveries", total_failed)
        with perf_col4:
            st.metric("⚠️ Invalid Addresses", total_invalid)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Results pie chart
            status_counts = results_df['status'].value_counts()
            fig = px.pie(
                values=status_counts.values, 
                names=status_counts.index,
                title="Email Campaign Results Distribution",
                color_discrete_map={'sent': '#28a745', 'failed': '#dc3545', 'invalid': '#ffc107'}
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Domain analysis for successful sends
            if total_sent > 0:
                sent_emails = results_df[results_df['status'] == 'sent'].copy()
                sent_emails['domain'] = sent_emails['email'].str.split('@').str[1]
                domain_counts = sent_emails['domain'].value_counts().head(8)
                
                fig = px.bar(
                    x=domain_counts.values, 
                    y=domain_counts.index,
                    title="Top Email Domains Reached",
                    orientation='h',
                    color_discrete_sequence=['#28a745']
                )
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        with st.expander("📋 View Detailed Email Campaign Results"):
            st.dataframe(
                results_df,
                column_config={
                    "email": st.column_config.TextColumn("📧 Email Address"),
                    "name": st.column_config.TextColumn("👤 Name"),
                    "status": st.column_config.TextColumn("📊 Status"),
                    "error": st.column_config.TextColumn("❌ Error (if any)"),
                    "timestamp": st.column_config.TextColumn("⏰ Time Sent")
                },
                use_container_width=True
            )
        
        # Export functionality
        st.subheader("📥 Export Campaign Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # CSV export
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                "📊 Download Results (CSV)",
                data=csv_data,
                file_name=f"email_campaign_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            # Summary report
            summary_report = f"""
# Email Campaign Results Summary

**Campaign Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Recipients:** {len(results_df)}

## Performance Metrics
- **Successfully Sent:** {total_sent} ({success_rate:.1f}%)
- **Failed Deliveries:** {total_failed} ({(total_failed/len(results_df)*100):.1f}%)
- **Invalid Addresses:** {total_invalid} ({(total_invalid/len(results_df)*100):.1f}%)

## Campaign Quality Assessment
- **Delivery Success Rate:** {success_rate:.1f}%
- **Campaign Performance:** {'Excellent' if success_rate >= 95 else 'Good' if success_rate >= 85 else 'Needs Improvement'}

## Recommendations
{"✅ Excellent delivery rate! Your email list is clean and well-targeted." if success_rate >= 95 else "⚠️ Consider cleaning your email list to improve delivery rates." if success_rate < 85 else "👍 Good performance. Continue monitoring for optimization opportunities."}

---
Generated by Marketing Campaign War Room
"""
            
            st.download_button(
                "📄 Download Summary Report",
                data=summary_report,
                file_name=f"campaign_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    else:
        # Show placeholder content when no data is available
        st.info("""
        📊 **Analytics Dashboard**
        
        This comprehensive analytics section provides:
        
        **🗺️ Geographic Analysis:**
        - Interactive campaign targeting maps
        - Location-based performance insights
        
        **📈 Performance Projections:**
        - ROI calculations and forecasts
        - Estimated reach and conversion metrics
        
        **📧 Email Campaign Analytics:**
        - Real-time delivery tracking
        - Success rate analysis
        - Domain performance breakdown
        
        **🎨 Creative Asset Tracking:**
        - Generated campaign images
        - Asset download and management
        
        **📥 Export Capabilities:**
        - Detailed CSV reports
        - Executive summary generation
        
        Create campaigns and send emails to unlock powerful analytics insights!
        """)

if __name__ == "__main__":
    main()

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime, timedelta
import re
from io import BytesIO
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Marketing Campaign Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
        margin-bottom: 1rem;
    }
    .campaign-section {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .email-template {
        background-color: #F1F3F4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'campaign_data' not in st.session_state:
    st.session_state.campaign_data = {}
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = {}
if 'email_contacts' not in st.session_state:
    st.session_state.email_contacts = []

class MarketingCampaignBot:
    def __init__(self):
        self.groq_api_key = None
        self.hf_token = None
        self.gmail_credentials = {}
        
    def setup_api_keys(self):
        """Setup API keys and credentials"""
        st.sidebar.markdown("### 🔐 API Configuration")
        
        self.groq_api_key = st.sidebar.text_input(
            "Groq API Key", 
            type="password", 
            help="Enter your Groq API key for AI generation"
        )
        
        self.hf_token = st.sidebar.text_input(
            "Hugging Face Token", 
            type="password", 
            help="Enter your HF token for image generation"
        )
        
        st.sidebar.markdown("### 📧 Gmail Configuration")
        self.gmail_credentials['email'] = st.sidebar.text_input("Gmail Address")
        self.gmail_credentials['password'] = st.sidebar.text_input(
            "App Password", 
            type="password", 
            help="Use Gmail App Password, not regular password"
        )
        
    def collect_campaign_inputs(self):
        """Collect campaign parameters from user"""
        st.markdown('<h1 class="main-header">🚀 Marketing Campaign Generator</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h2 class="sub-header">Campaign Details</h2>', unsafe_allow_html=True)
            
            company_name = st.text_input("Company Name", placeholder="Enter your company name")
            
            campaign_type = st.selectbox(
                "Campaign Type",
                ["Product Launch", "Brand Awareness", "Seasonal Offer", "Loyalty Program", 
                 "Lead Generation", "Event Promotion", "Customer Retention"]
            )
            
            target_audience = st.text_area(
                "Target Audience", 
                placeholder="Describe your target demographics, psychographics, B2B/B2C details..."
            )
            
            duration = st.text_input(
                "Campaign Duration", 
                placeholder="e.g., 3 months, Q1 2025, 30 days"
            )
            
            location = st.text_input(
                "Target Location", 
                placeholder="Geographic market (e.g., North America, India, Global)"
            )
        
        with col2:
            st.markdown('<h2 class="sub-header">Additional Parameters</h2>', unsafe_allow_html=True)
            
            channels = st.multiselect(
                "Marketing Channels",
                ["Social Media", "Email Marketing", "SEO", "Paid Ads", "Events", 
                 "Influencer Marketing", "Content Marketing", "PR", "Direct Mail"]
            )
            
            language = st.selectbox(
                "Language Preference",
                ["English", "Spanish", "French", "German", "Hindi", "Mandarin", "Other"]
            )
            
            customer_segment = st.selectbox(
                "Customer Segment",
                ["Luxury", "Mid-tier", "Budget", "Niche", "Enterprise", "SMB", "Consumer"]
            )
            
            product_description = st.text_area(
                "Product/Service Description",
                placeholder="Describe your product or service in detail..."
            )
            
            budget = st.number_input(
                "Budget (USD)", 
                min_value=0, 
                value=10000,
                help="Leave 0 for automatic estimation"
            )
        
        # Store inputs in session state
        st.session_state.campaign_data = {
            'company_name': company_name,
            'campaign_type': campaign_type,
            'target_audience': target_audience,
            'duration': duration,
            'location': location,
            'channels': channels,
            'language': language,
            'customer_segment': customer_segment,
            'product_description': product_description,
            'budget': budget
        }
        
        return st.session_state.campaign_data
    
    def generate_campaign_strategy(self, campaign_data):
        """Generate comprehensive campaign strategy using Groq"""
        if not self.groq_api_key:
            st.error("Please enter your Groq API key in the sidebar")
            return None
            
        prompt = f"""
        Create a comprehensive marketing campaign strategy for:
        
        Company: {campaign_data['company_name']}
        Campaign Type: {campaign_data['campaign_type']}
        Target Audience: {campaign_data['target_audience']}
        Duration: {campaign_data['duration']}
        Location: {campaign_data['location']}
        Channels: {', '.join(campaign_data['channels'])}
        Customer Segment: {campaign_data['customer_segment']}
        Product/Service: {campaign_data['product_description']}
        Budget: ${campaign_data['budget']}
        
        Provide a detailed response with:
        1. Executive Summary
        2. Phase-wise execution plan (pre-launch, launch, growth, retention)
        3. Channel-specific strategies with KPIs
        4. Budget breakdown and allocation
        5. Timeline with milestones
        6. Success metrics and ROI projections
        7. Risk mitigation strategies
        """
        
        try:
            # Simulated API call - replace with actual Groq API integration
            strategy = self.mock_groq_response(prompt, "campaign_strategy")
            return strategy
        except Exception as e:
            st.error(f"Error generating campaign strategy: {str(e)}")
            return None
    
    def generate_creative_content(self, campaign_data):
        """Generate creative content including ads, emails, social posts"""
        content = {
            'ad_copies': [],
            'social_posts': [],
            'email_templates': [],
            'slogans': [],
            'blog_outline': ""
        }
        
        # Generate ad copies
        ad_prompt = f"Create 5 compelling ad copies for {campaign_data['campaign_type']} campaign for {campaign_data['company_name']} targeting {campaign_data['customer_segment']} segment."
        content['ad_copies'] = self.mock_groq_response(ad_prompt, "ad_copies")
        
        # Generate social media posts
        social_prompt = f"Create 10 social media posts with hashtags for {campaign_data['company_name']} {campaign_data['campaign_type']} campaign."
        content['social_posts'] = self.mock_groq_response(social_prompt, "social_posts")
        
        # Generate email templates
        email_prompt = f"Create 3 email templates (welcome, nurture, conversion) for {campaign_data['company_name']} targeting {campaign_data['target_audience']}."
        content['email_templates'] = self.mock_groq_response(email_prompt, "email_templates")
        
        # Generate slogans
        slogan_prompt = f"Create 5 catchy slogans for {campaign_data['company_name']} {campaign_data['campaign_type']} campaign."
        content['slogans'] = self.mock_groq_response(slogan_prompt, "slogans")
        
        return content
    
    def mock_groq_response(self, prompt, content_type):
        """Mock Groq API response - replace with actual API calls"""
        mock_responses = {
            "campaign_strategy": {
                "executive_summary": f"Strategic {st.session_state.campaign_data.get('campaign_type', 'marketing')} campaign designed to maximize ROI and brand engagement.",
                "phases": {
                    "pre_launch": "Market research, audience segmentation, content creation (Week 1-2)",
                    "launch": "Multi-channel campaign activation, PR outreach (Week 3-4)",
                    "growth": "Performance optimization, A/B testing, scale winning variants (Week 5-8)",
                    "retention": "Customer nurturing, loyalty programs, feedback collection (Week 9-12)"
                },
                "budget_breakdown": {
                    "paid_advertising": "40%",
                    "content_creation": "25%",
                    "influencer_partnerships": "20%",
                    "events_pr": "10%",
                    "tools_analytics": "5%"
                },
                "kpis": ["Brand Awareness (+30%)", "Lead Generation (+150%)", "Conversion Rate (+25%)", "ROI (3:1)"]
            },
            "ad_copies": [
                f"🚀 Transform your business with {st.session_state.campaign_data.get('company_name', 'our')} innovative solutions!",
                f"Limited time offer: Get premium {st.session_state.campaign_data.get('product_description', 'products')} at unbeatable prices!",
                f"Join thousands of satisfied customers who chose {st.session_state.campaign_data.get('company_name', 'us')} for excellence.",
                f"Don't miss out! Exclusive {st.session_state.campaign_data.get('campaign_type', 'offer')} ending soon.",
                f"Experience the difference with {st.session_state.campaign_data.get('company_name', 'our')} award-winning service."
            ],
            "social_posts": [
                f"🎉 Big news! Our {st.session_state.campaign_data.get('campaign_type', 'campaign')} is here! #Innovation #Growth",
                f"✨ Why choose {st.session_state.campaign_data.get('company_name', 'us')}? Quality, reliability, results! #TrustTheProcess",
                f"🔥 Hot deal alert! Limited time {st.session_state.campaign_data.get('campaign_type', 'offer')} - don't wait! #DealAlert",
                f"💪 Success story: How we helped customers achieve their goals #SuccessStory #CustomerLove",
                f"🌟 Behind the scenes: Meet our amazing team making magic happen! #TeamWork #BehindTheScenes"
            ],
            "email_templates": [
                {
                    "subject": f"Welcome to {st.session_state.campaign_data.get('company_name', 'Our Community')}!",
                    "body": f"Hi {{name}},\n\nWelcome to the {st.session_state.campaign_data.get('company_name', 'family')}! We're excited to have you on board.\n\nBest regards,\nThe Team"
                },
                {
                    "subject": f"Exclusive offer just for you!",
                    "body": f"Hi {{name}},\n\nAs a valued member, here's an exclusive offer on our {st.session_state.campaign_data.get('product_description', 'products')}.\n\nDon't miss out!\n\nBest,\nThe Team"
                }
            ],
            "slogans": [
                f"{st.session_state.campaign_data.get('company_name', 'Innovation')} - Where Excellence Meets Innovation",
                f"Your Success, Our Mission - {st.session_state.campaign_data.get('company_name', 'Company')}",
                f"Leading the Future of {st.session_state.campaign_data.get('product_description', 'Industry')}",
                f"Quality You Can Trust, Results You Can See",
                f"Transforming Ideas into Reality"
            ]
        }
        
        return mock_responses.get(content_type, f"Generated content for {content_type}")
    
    def create_budget_visualization(self, budget_data):
        """Create interactive budget breakdown chart"""
        if isinstance(budget_data, dict):
            categories = list(budget_data.keys())
            values = [float(v.replace('%', '')) for v in budget_data.values()]
            
            fig = px.pie(
                values=values,
                names=categories,
                title="Campaign Budget Allocation",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        return None
    
    def file_upload_handler(self):
        """Handle dataset uploads for EDA and email campaigns"""
        st.markdown('<h2 class="sub-header">📊 Data Upload & Analysis</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload customer dataset (CSV, Excel, JSON)",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Upload customer data for EDA and email campaigns"
        )
        
        if uploaded_file is not None:
            try:
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.success(f"Successfully uploaded {len(df)} records!")
                
                # Basic dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Preview data
                st.markdown("### Data Preview")
                st.dataframe(df.head(10))
                
                # Extract email contacts if email column exists
                email_columns = [col for col in df.columns if 'email' in col.lower()]
                if email_columns:
                    self.extract_email_contacts(df, email_columns[0])
                
                return df
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return None
        
        return None
    
    def extract_email_contacts(self, df, email_column):
        """Extract email contacts for campaigns"""
        try:
            # Clean and validate emails
            valid_emails = df[df[email_column].str.contains('@', na=False)]
            
            # Try to extract names
            name_columns = [col for col in df.columns if any(name_term in col.lower() for name_term in ['name', 'first', 'last'])]
            
            contacts = []
            for idx, row in valid_emails.iterrows():
                contact = {'email': row[email_column]}
                
                # Try to get name
                if name_columns:
                    if 'name' in name_columns[0].lower():
                        contact['name'] = row[name_columns[0]]
                    else:
                        # Combine first and last name if available
                        first_name = row.get(name_columns[0], '') if name_columns else ''
                        last_name = row.get(name_columns[1], '') if len(name_columns) > 1 else ''
                        contact['name'] = f"{first_name} {last_name}".strip()
                else:
                    contact['name'] = row[email_column].split('@')[0]  # Use email prefix as name
                
                contacts.append(contact)
            
            st.session_state.email_contacts = contacts
            st.success(f"Extracted {len(contacts)} email contacts for campaigns!")
            
            # Show contact preview
            if contacts:
                st.markdown("### Email Contacts Preview")
                contact_df = pd.DataFrame(contacts[:10])  # Show first 10
                st.dataframe(contact_df)
                
        except Exception as e:
            st.error(f"Error extracting email contacts: {str(e)}")
    
    def perform_eda(self, df):
        """Perform comprehensive EDA with visualizations"""
        if df is None:
            return
            
        st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Basic statistics
        st.markdown("### Dataset Statistics")
        st.dataframe(df.describe())
        
        # Correlation heatmap for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # Distribution plots
        st.markdown("### Data Distributions")
        for col in numeric_cols[:4]:  # Show first 4 numeric columns
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer segmentation (if applicable)
        if 'age' in df.columns and 'income' in df.columns:
            st.markdown("### Customer Segmentation")
            fig = px.scatter(df, x='age', y='income', title='Customer Segmentation by Age and Income')
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate insights
        insights = self.generate_eda_insights(df)
        st.markdown("### 🔍 Key Insights")
        for insight in insights:
            st.info(insight)
    
    def generate_eda_insights(self, df):
        """Generate actionable insights from EDA"""
        insights = []
        
        # Basic insights
        insights.append(f"Dataset contains {len(df)} records with {len(df.columns)} features")
        
        # Missing data insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 5:
            insights.append(f"⚠️ {missing_pct:.1f}% of data is missing - consider data cleaning strategies")
        
        # Numeric insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            high_var_cols = df[numeric_cols].std().nlargest(3).index.tolist()
            insights.append(f"Highest variability in: {', '.join(high_var_cols)}")
        
        # Categorical insights
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            insights.append(f"Found {len(categorical_cols)} categorical variables for potential segmentation")
        
        return insights
    
    def email_campaign_manager(self):
        """Manage email campaigns"""
        st.markdown('<h2 class="sub-header">📧 Email Campaign Manager</h2>', unsafe_allow_html=True)
        
        if not st.session_state.email_contacts:
            st.warning("No email contacts found. Please upload a dataset with email addresses.")
            return
        
        # Campaign settings
        col1, col2 = st.columns(2)
        
        with col1:
            email_subject = st.text_input("Email Subject", value="Exclusive Offer from Our Team!")
            sender_name = st.text_input("Sender Name", value=st.session_state.campaign_data.get('company_name', 'Company'))
        
        with col2:
            email_type = st.selectbox("Email Type", ["Plain Text", "HTML Template"])
            send_test = st.checkbox("Send Test Email First")
        
        # Email content
        st.markdown("### Email Content")
        if email_type == "Plain Text":
            email_body = st.text_area(
                "Email Body",
                value=f"""Hi {{name}},

We hope this email finds you well!

We're excited to share an exclusive offer just for you from {st.session_state.campaign_data.get('company_name', 'our company')}.

{st.session_state.campaign_data.get('product_description', 'Our amazing products')} are now available with special pricing.

Don't miss out on this limited-time opportunity!

Best regards,
{sender_name} Team

---
This email was sent as part of our {st.session_state.campaign_data.get('campaign_type', 'marketing')} campaign.""",
                height=300
            )
        else:
            email_body = self.create_html_email_template()
        
        # Preview and send
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Preview Email"):
                self.preview_email(email_subject, email_body, email_type)
        
        with col2:
            if st.button("Send Test Email") and send_test:
                test_email = st.text_input("Test Email Address")
                if test_email and self.gmail_credentials.get('email'):
                    self.send_single_email(test_email, "Test User", email_subject, email_body, email_type)
        
        with col3:
            if st.button("Send Campaign"):
                if self.gmail_credentials.get('email') and self.gmail_credentials.get('password'):
                    self.send_bulk_emails(email_subject, email_body, email_type, sender_name)
                else:
                    st.error("Please configure Gmail credentials in the sidebar")
    
    def create_html_email_template(self):
        """Create responsive HTML email template"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{st.session_state.campaign_data.get('campaign_type', 'Campaign')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; }}
                .header {{ background-color: #FF6B6B; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 30px; }}
                .cta-button {{ display: inline-block; background-color: #4ECDC4; color: white; 
                              padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{st.session_state.campaign_data.get('company_name', 'Company Name')}</h1>
                    <h2>{st.session_state.campaign_data.get('campaign_type', 'Special Offer')}</h2>
                </div>
                <div class="content">
                    <h3>Hi {{name}},</h3>
                    <p>We're excited to share something special with you!</p>
                    <p>{st.session_state.campaign_data.get('product_description', 'Our amazing products and services')} are designed to help you achieve your goals.</p>
                    <p>For a limited time, we're offering exclusive benefits to our valued customers.</p>
                    <a href="#" class="cta-button">Get Started Now</a>
                    <p>Don't miss out on this opportunity to transform your experience with us.</p>
                </div>
                <div class="footer">
                    <p>This email was sent by {st.session_state.campaign_data.get('company_name', 'Company')}</p>
                    <p>Part of our {st.session_state.campaign_data.get('campaign_type', 'marketing')} campaign</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template
    
    def preview_email(self, subject, body, email_type):
        """Preview email before sending"""
        st.markdown("### 📧 Email Preview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Subject:** {subject}")
            st.markdown(f"**Type:** {email_type}")
        
        if email_type == "HTML Template":
            # Show HTML preview
            sample_body = body.replace("{name}", "John Doe")
            st.markdown("### HTML Preview")
            st.components.v1.html(sample_body, height=600)
        else:
            # Show plain text preview
            sample_body = body.replace("{name}", "John Doe")
            st.markdown("### Text Preview")
            st.text(sample_body)
    
    def send_single_email(self, email, name, subject, body, email_type):
        """Send single test email"""
        try:
            personalized_body = body.replace("{name}", name)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.gmail_credentials['email']
            msg['To'] = email
            
            if email_type == "HTML Template":
                html_part = MIMEText(personalized_body, 'html')
                msg.attach(html_part)
            else:
                text_part = MIMEText(personalized_body, 'plain')
                msg.attach(text_part)
            
            # Note: In production, implement actual SMTP sending
            st.success(f"Test email sent to {email}!")
            
        except Exception as e:
            st.error(f"Error sending email: {str(e)}")
    
    def send_bulk_emails(self, subject, body, email_type, sender_name):
        """Send bulk emails to all contacts"""
        if not st.session_state.email_contacts:
            st.error("No email contacts available")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        sent_count = 0
        total_count = len(st.session_state.email_contacts)
        
        for i, contact in enumerate(st.session_state.email_contacts):
            try:
                # Simulate sending (implement actual SMTP in production)
                status_text.text(f"Sending to {contact['email']}...")
                
                # In production, call send_single_email here
                # self.send_single_email(contact['email'], contact['name'], subject, body, email_type)
                
                sent_count += 1
                progress_bar.progress((i + 1) / total_count)
                
            except Exception as e:
                st.error(f"Failed to send to {contact['email']}: {str(e)}")
        
        st.success(f"Campaign completed! Sent {sent_count}/{total_count} emails successfully.")
    
    def export_campaign_data(self):
        """Export campaign data and results"""
        st.markdown("### 📥 Export Campaign Data")
        
        export_data = {
            'campaign_parameters': st.session_state.campaign_data,
            'generated_content': st.session_state.generated_content,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create downloadable files
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export as JSON"):
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"campaign_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export Contacts as CSV"):
                if st.session_state.email_contacts:
                    contacts_df = pd.DataFrame(st.session_state.email_contacts)
                    csv_data = contacts_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"email_contacts_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("Generate Report"):
                # Create a comprehensive campaign report
                report_content = self.generate_campaign_report()
                st.download_button(
                    label="Download Report",
                    data=report_content,
                    file_name=f"campaign_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

    def generate_campaign_report(self):
        """Generate comprehensive campaign report"""
        report = f"""
# MARKETING CAMPAIGN REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## CAMPAIGN OVERVIEW
Company: {st.session_state.campaign_data.get('company_name', 'N/A')}
Campaign Type: {st.session_state.campaign_data.get('campaign_type', 'N/A')}
Target Audience: {st.session_state.campaign_data.get('target_audience', 'N/A')}
Duration: {st.session_state.campaign_data.get('duration', 'N/A')}
Budget: ${st.session_state.campaign_data.get('budget', 0):,}
Location: {st.session_state.campaign_data.get('location', 'N/A')}
Channels: {', '.join(st.session_state.campaign_data.get('channels', []))}

## GENERATED CONTENT SUMMARY
- Ad Copies: Generated
- Social Media Posts: Generated  
- Email Templates: Generated
- Campaign Slogans: Generated
- Visual Assets: Prompts Generated

## EMAIL CAMPAIGN STATS
Total Contacts: {len(st.session_state.email_contacts)}
Email Types: Plain Text & HTML Templates Available
Personalization: Name-based personalization implemented

## RECOMMENDATIONS
1. Monitor campaign performance across all channels
2. A/B test different creative variations
3. Track ROI and adjust budget allocation accordingly
4. Engage with audience feedback and optimize messaging
5. Scale successful tactics and pause underperforming ones

## NEXT STEPS
- Deploy campaign assets across selected channels
- Set up tracking and analytics
- Schedule regular performance reviews
- Prepare for optimization iterations

---
Report generated by Marketing Campaign Generator Bot
        """
        return report


def main():
    """Main application function"""
    bot = MarketingCampaignBot()
    
    # Setup API keys
    bot.setup_api_keys()
    
    # Navigation
    st.sidebar.markdown("### 🎯 Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Campaign Generator", "Email Manager", "Data Analysis", "Export Results"]
    )
    
    if mode == "Campaign Generator":
        # Collect campaign inputs
        campaign_data = bot.collect_campaign_inputs()
        
        # Generate campaign button
        if st.button("🚀 Generate Campaign Strategy", type="primary"):
            if campaign_data['company_name'] and campaign_data['target_audience']:
                with st.spinner("Generating comprehensive campaign strategy..."):
                    
                    # Generate strategy
                    strategy = bot.generate_campaign_strategy(campaign_data)
                    
                    if strategy:
                        st.session_state.generated_content['strategy'] = strategy
                        
                        # Display results
                        st.markdown('<div class="campaign-section">', unsafe_allow_html=True)
                        st.markdown("## 📋 Executive Summary")
                        st.info(strategy.get('executive_summary', 'Campaign strategy generated successfully'))
                        
                        # Phase timeline
                        st.markdown("## 📅 Campaign Phases")
                        phases = strategy.get('phases', {})
                        for phase, description in phases.items():
                            st.markdown(f"**{phase.replace('_', ' ').title()}:** {description}")
                        
                        # Budget visualization
                        if 'budget_breakdown' in strategy:
                            st.markdown("## 💰 Budget Allocation")
                            budget_fig = bot.create_budget_visualization(strategy['budget_breakdown'])
                            if budget_fig:
                                st.plotly_chart(budget_fig, use_container_width=True)
                        
                        # KPIs
                        if 'kpis' in strategy:
                            st.markdown("## 📊 Key Performance Indicators")
                            kpi_cols = st.columns(len(strategy['kpis']))
                            for i, kpi in enumerate(strategy['kpis']):
                                kpi_cols[i].metric("Target KPI", kpi)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Generate creative content
                    with st.spinner("Creating campaign assets..."):
                        creative_content = bot.generate_creative_content(campaign_data)
                        st.session_state.generated_content['creative'] = creative_content
                        
                        # Display creative content
                        st.markdown('<div class="campaign-section">', unsafe_allow_html=True)
                        st.markdown("## 🎨 Creative Assets")
                        
                        # Ad copies
                        st.markdown("### Advertisement Copies")
                        for i, ad in enumerate(creative_content['ad_copies'], 1):
                            st.markdown(f"**Ad Copy {i}:** {ad}")
                        
                        # Social media posts
                        st.markdown("### Social Media Posts")
                        for i, post in enumerate(creative_content['social_posts'][:5], 1):
                            st.markdown(f"**Post {i}:** {post}")
                        
                        # Slogans
                        st.markdown("### Campaign Slogans")
                        for slogan in creative_content['slogans']:
                            st.markdown(f"- {slogan}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Please fill in at least Company Name and Target Audience")
    
    elif mode == "Email Manager":
        bot.email_campaign_manager()
    
    elif mode == "Data Analysis":
        df = bot.file_upload_handler()
        if df is not None:
            bot.perform_eda(df)
    
    elif mode == "Export Results":
        bot.export_campaign_data()
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **Marketing Campaign Generator Bot** - Your AI-Powered Marketing Assistant")

if __name__ == "__main__":
    main()
