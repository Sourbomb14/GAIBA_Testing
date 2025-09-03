import streamlit as st
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

# Load environment variables
load_dotenv()

# Configuration - Updated based on your .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

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
            1. Include personalization placeholders: {{{{first_name}}}}, {{{{name}}}}, {{{{email}}}}
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
            <h1>Hello {{{{first_name}}}}! 👋</h1>
            <p>We have something special for you</p>
        </div>
        <div class="content">
            <p>Dear {{{{name}}}},</p>
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
            return f"""Subject: Exclusive {template_type} for {{{{first_name}}}}

Hello {{{{first_name}}}},

We're excited to share this exclusive {template_type.lower()} with you.

As a valued member of our community, you deserve the best we have to offer.

Here's what makes this special:
• Personalized just for you
• Exclusive member benefits
• Limited-time opportunity
• Premium experience

Ready to explore? Visit our website or reply to this email.

Thank you for being part of our journey, {{{{name}}}}!

Best regards,
The Marketing Team

---
You received this email because you're subscribed to our updates.
"""

# ================================
# ALTERNATIVE IMAGE GENERATOR USING EXTERNAL APIs
# ================================

class ExternalImageGenerator:
    """Generate images using external APIs (no diffusers dependency)"""
    
    def __init__(self):
        self.hugging_face_token = HUGGING_FACE_TOKEN
        
    def generate_campaign_image_via_api(self, campaign_description, style="professional"):
        """Generate campaign image using HuggingFace Inference API"""
        if not self.hugging_face_token:
            st.warning("⚠️ HuggingFace token not configured for image generation")
            return None
            
        try:
            # Enhanced prompt for marketing
            enhanced_prompt = f"Professional marketing campaign image for {campaign_description}, {style} style, high quality, vibrant colors, modern design, commercial photography, eye-catching, brand advertisement, 4K resolution, clean layout"
            
            # HuggingFace Inference API endpoint
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {self.hugging_face_token}"}
            
            payload = {
                "inputs": enhanced_prompt,
                "parameters": {
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
            }
            
            with st.spinner("🎨 Generating your campaign image via API..."):
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    # Convert response to image
                    image_bytes = response.content
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Store in session state
                    image_data = {
                        'prompt': enhanced_prompt,
                        'timestamp': datetime.now(),
                        'campaign': campaign_description,
                        'image': image
                    }
                    
                    st.session_state.generated_images.append(image_data)
                    
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
                else:
                    st.error(f"❌ Image generation failed: {response.status_code}")
                    st.info("💡 The model might be loading. Try again in a few minutes.")
                    return None
                    
        except requests.exceptions.Timeout:
            st.error("⏰ Image generation timed out. Please try again.")
            return None
        except Exception as e:
            st.error(f"❌ Error generating image: {str(e)}")
            return None
    
    def generate_placeholder_image(self, campaign_description):
        """Generate a placeholder image with campaign text"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a placeholder image
            width, height = 512, 512
            image = Image.new('RGB', (width, height), color='#f0f0f0')
            draw = ImageDraw.Draw(image)
            
            # Add campaign text
            text = f"Campaign Image\n{campaign_description[:50]}..."
            
            try:
                # Try to use a nice font
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            # Draw text
            draw.text((x, y), text, fill='#333333', font=font, anchor="la")
            
            # Store in session state
            image_data = {
                'prompt': f"Placeholder for: {campaign_description}",
                'timestamp': datetime.now(),
                'campaign': campaign_description,
                'image': image
            }
            
            st.session_state.generated_images.append(image_data)
            
            st.info("📷 Generated placeholder campaign image")
            st.image(image, caption=f"Placeholder for: {campaign_description}", use_column_width=True)
            
            return image
            
        except Exception as e:
            st.error(f"❌ Error creating placeholder: {str(e)}")
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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        sent_count = 0
        failed_count = 0
        invalid_count = 0
        
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
                
                for index, row in email_list.iterrows():
                    current_progress = (index + 1) / total_emails
                    
                    progress_bar.progress(current_progress)
                    status_text.info(f"📧 Sending {index + 1}/{total_emails}: {row['email']}")
                    
                    try:
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
                        
                        name = row.get('name', personalizer.extract_name_from_email(row['email']))
                        personal_subject = personalizer.personalize_template(subject, name, row['email'])
                        personal_body = personalizer.personalize_template(body_template, name, row['email'])
                        
                        msg = MIMEMultipart('alternative')
                        msg['Subject'] = personal_subject
                        msg['From'] = self.email
                        msg['To'] = row['email']
                        
                        if is_html:
                            msg.attach(MIMEText(personal_body, 'html'))
                        else:
                            msg.attach(MIMEText(personal_body, 'plain'))
                        
                        server.sendmail(self.email, row['email'], msg.as_string())
                        
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
                    
                    with metrics_container:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("✅ Sent", sent_count)
                        col2.metric("❌ Failed", failed_count)
                        col3.metric("⚠️ Invalid", invalid_count)
                        col4.metric("📊 Progress", f"{current_progress*100:.0f}%")
                    
                    time.sleep(1)
                
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
        df.columns = df.columns.str.lower()
        
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
        
        result_data = []
        
        for _, row in df.iterrows():
            email = row[email_col]
            if pd.isna(email) or str(email).strip() == '':
                continue
            
            email = str(email).strip().lower()
            
            if name_col and not pd.isna(row[name_col]):
                name = str(row[name_col]).strip()
            else:
                name = self.personalizer.extract_name_from_email(email)
            
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
            st.success("🎨 Image Generator: Connected (API)")
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
                "Lead Generation", "Event Promotion", "Sales Campaign", "Newsletter Campaign"
            ])
            
            target_audience = st.text_area("👥 Target Audience", 
                placeholder="Describe demographics, interests, pain points, behaviors...")
            
            duration = st.text_input("📅 Campaign Duration", placeholder="e.g., 6 weeks, 3 months")
        
        with col2:
            channels = st.multiselect("📢 Marketing Channels", [
                "Email Marketing", "Social Media Marketing", "Google Ads", "Facebook Ads", 
                "Content Marketing", "Influencer Marketing", "SEO/SEM", "TV/Radio", "Print Media"
            ])
            
            location = st.selectbox("🌍 Target Country", COUNTRIES)
            city_state = st.text_input("🏙️ City/State", placeholder="e.g., New York, NY")
            customer_segment = st.selectbox("💼 Customer Segment", 
                ["Mass Market", "Premium", "Luxury", "Niche", "Enterprise", "SMB"])
        
        budget_col1, budget_col2 = st.columns(2)
        with budget_col1:
            budget = st.text_input("💰 Budget Amount", placeholder="e.g., 50000")
        with budget_col2:
            currency = st.selectbox("💱 Currency", CURRENCIES)
        
        product_description = st.text_area("📦 Product/Service Description",
            placeholder="Describe what you're promoting: features, benefits, unique selling points...")
        
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
            
            st.session_state.current_campaign = campaign_data
            st.session_state.campaign_blueprint = strategy
            
            st.success("✨ Campaign strategy generated successfully!")
            st.balloons()
    
    # Handle image generation
    if generate_image and st.session_state.current_campaign:
        image_gen = ExternalImageGenerator()
        campaign_desc = f"{st.session_state.current_campaign['company_name']} {st.session_state.current_campaign['campaign_type']}"
        
        # Try API first, fallback to placeholder
        image = image_gen.generate_campaign_image_via_api(campaign_desc, "professional")
        if image is None:
            # Fallback to placeholder
            image_gen.generate_placeholder_image(campaign_desc)
    
    # Display existing campaign
    if st.session_state.campaign_blueprint:
        st.markdown("---")
        st.markdown("## 📋 Your AI-Generated Campaign Strategy")
        st.markdown(st.session_state.campaign_blueprint)
        
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
            "Newsletter", "Follow-up Email", "Event Invitation"
        ])
        tone = st.selectbox("🎭 Email Tone", [
            "Professional", "Friendly", "Casual", "Urgent", "Formal"
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
                                    help="Use {{first_name}}, {{name}}, and {{email}} for personalization")
        
        if edit_choice == "HTML Template":
            st.session_state.email_template_html = edited_content
        else:
            st.session_state.email_template_text = edited_content
        
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
            
            if not use_template or not single_body:
                single_body = st.text_area("📧 Email Content", 
                    value=single_body,
                    placeholder="Enter your email content here...",
                    height=200,
                    help="Use {{first_name}}, {{name}}, and {{email}} for personalization")
                single_is_html = st.checkbox("📄 Send as HTML", value=single_is_html)
            else:
                st.text_area("📧 Email Content Preview", 
                    value=single_body[:200] + "..." if len(single_body) > 200 else single_body,
                    height=100, disabled=True)
            
            send_single = st.form_submit_button("📧 Send Single Email", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
                    
                    final_name = single_name if single_name else personalizer.extract_name_from_email(single_email)
                    final_subject = personalizer.personalize_template(single_subject, final_name, single_email)
                    final_body = personalizer.personalize_template(single_body, final_name, single_email)
                    
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
    
    uploaded_file = st.file_uploader("📁 Upload Contact File (CSV/Excel)", 
        type=['csv', 'xlsx'], key="contact_upload",
        help="Upload a CSV or Excel file with 'email' and optionally 'name' columns")
    
    if uploaded_file:
        processor = FileProcessor()
        contacts = processor.process_file(uploaded_file)
        
        if contacts is not None:
            st.session_state.email_contacts = contacts
            st.success(f"✅ Successfully loaded {len(contacts)} valid contacts!")
            
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
            estimated_time = len(df) * 1.5 / 60
            st.metric("⏱️ Est. Time", f"{estimated_time:.0f}m")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            bulk_subject = st.text_input("📧 Campaign Subject Line", 
                value="Important
