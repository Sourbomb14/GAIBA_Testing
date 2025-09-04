import streamlit as st
import pandas as pd
import numpy as np
import yagmail
import time
import re
import json
import plotly.express as px
import plotly.graph_objects as go
from email_validator import validate_email, EmailNotValidError
import os
from datetime import datetime, timedelta
import io
import base64
from groq import Groq
import requests
from PIL import Image, ImageDraw, ImageFont
import textwrap
import csv

# Configuration using Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
HUGGING_FACE_TOKEN = st.secrets.get("HUGGING_FACE_TOKEN", "")

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
    if 'uploaded_dataset' not in st.session_state:
        st.session_state.uploaded_dataset = None
    if 'data_analysis_result' not in st.session_state:
        st.session_state.data_analysis_result = None

# ================================
# GROQ AI CAMPAIGN GENERATOR
# ================================

class GroqCampaignGenerator:
    """Generate campaigns using Groq API"""
    
    def __init__(self):
        self.client = None
        if GROQ_API_KEY:
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                st.error(f"Failed to initialize Groq: {e}")
    
    def generate_campaign_strategy(self, campaign_data):
        """Generate comprehensive campaign strategy"""
        if not self.client:
            return self._fallback_strategy(campaign_data)
        
        try:
            prompt = self._build_campaign_prompt(campaign_data)
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a world-class marketing strategist with 20+ years of experience. Create detailed, actionable, and comprehensive marketing campaigns that drive real results. Focus on practical implementation, specific tactics, and measurable outcomes. Include visual elements like emoji charts, tables, and structured layouts for better presentation."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL,
                temperature=0.7,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error generating campaign with Groq: {e}")
            return self._fallback_strategy(campaign_data)
    
    def generate_email_template(self, template_type, tone, format_type, campaign_context=None):
        """Generate clean email templates using Groq AI"""
        if not self.client:
            return self._fallback_email_template(template_type, tone, format_type)
        
        try:
            prompt = f"Create a professional {template_type.lower()} email template with a {tone.lower()} tone. Format: {format_type}. Campaign Context: {campaign_context if campaign_context else 'General marketing campaign'}. Requirements: 1. Include personalization placeholders: {{{{first_name}}}}, {{{{name}}}}, {{{{email}}}}. 2. Make it engaging and action-oriented. 3. Include a clear call-to-action. 4. Use modern, professional design. 5. Make it mobile-friendly. 6. IMPORTANT: Do NOT include any instructions, explanations, or meta-text in the output. 7. ONLY provide the clean template content that can be used directly. Generate only the template content without any explanations."
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert email marketing specialist. Create ONLY clean, ready-to-use email templates without any instructions or explanations. Provide ONLY the template content that can be used directly."
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
    
    def analyze_data(self, df_sample, file_info):
        """Analyze uploaded data using Groq AI"""
        if not self.client:
            return "Groq AI not available for data analysis"
        
        try:
            # Convert sample data to JSON-serializable format
            def safe_json_serializable(o):
                if isinstance(o, (pd.Timestamp, pd.Timedelta)):
                    return str(o)
                elif isinstance(o, (pd.NaType, type(pd.NaT))):
                    return None
                elif pd.isna(o):
                    return None
                raise TypeError(f"Type {type(o)} not serializable")
            
            sample_data = df_sample.head(10).to_dict(orient='records')
            
            prompt = f"You are a professional data analyst. Analyze the following dataset: FILE INFO: {file_info}. SAMPLE DATA (first 10 rows): {json.dumps(sample_data, indent=2, default=safe_json_serializable)}. DATASET SHAPE: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns. COLUMNS: {list(df_sample.columns)}. Please provide a comprehensive analysis including: 1. DATA SUMMARY & STATISTICS, 2. KEY INSIGHTS & TRENDS, 3. RECOMMENDED VISUALIZATIONS - Suggest specific chart types for the data - Recommend columns to visualize together, 4. DATA QUALITY ASSESSMENT, 5. BUSINESS RECOMMENDATIONS - Actionable insights for management - Strategic recommendations based on the data. Format your response with clear headings and bullet points for easy reading."
            
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data analyst with deep business acumen. Provide thorough, actionable analysis with clear visualizations recommendations and strategic insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=GROQ_MODEL,
                temperature=0.3,
                max_tokens=3000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error analyzing data: {e}")
            return f"Error analyzing data: {str(e)}"
    
    def _build_campaign_prompt(self, data):
        """Build comprehensive campaign prompt"""
        channels_str = ', '.join(data.get('channels', ['Email']))
        prompt_text = f"Create a comprehensive, actionable marketing campaign strategy for: CAMPAIGN DETAILS: - Company: {data.get('company_name', 'Company')} - Campaign Type: {data.get('campaign_type', 'Marketing Campaign')} - Target Audience: {data.get('target_audience', 'General audience')} - Geographic Focus: {data.get('location', 'Global')} {data.get('city_state', '')} - Marketing Channels: {channels_str} - Budget: {data.get('budget', 'TBD')} {data.get('currency', 'USD')} - Duration: {data.get('duration', 'TBD')} - Customer Segment: {data.get('customer_segment', 'Mass Market')} - Product/Service: {data.get('product_description', 'Product/Service')}. DELIVERABLES REQUIRED: Create a comprehensive strategy with visual elements including: 1. Executive Summary with key metrics visualization, 2. Market Analysis with audience breakdown tables, 3. Competitive Positioning with comparison charts, 4. Messaging Strategy with key themes, 5. Channel-Specific Tactics with implementation timeline, 6. Content Strategy with content calendar, 7. Timeline & Milestones with visual project roadmap, 8. Budget Allocation with spending breakdown charts, 9. Success Metrics & KPIs with measurement framework, 10. Risk Management with mitigation strategies, 11. Next Steps with priority action items. Use emojis, tables, charts descriptions, and structured layouts to make it visually appealing and easy to read. Make this practical, specific, and actionable with real tactics and numbers."
        
        return prompt_text
    
    def _fallback_strategy(self, data):
        """Enhanced fallback campaign strategy"""
        budget_val = data.get('budget', '10000')
        try:
            budget_num = int(budget_val) if str(budget_val).isdigit() else 10000
        except:
            budget_num = 10000
        
        company_name = data.get('company_name', 'Your Company')
        campaign_type = data.get('campaign_type', 'Marketing')
        location = data.get('location', 'Global')
        customer_segment = data.get('customer_segment', 'Mass Market')
        target_audience = data.get('target_audience', 'Primary target audience to be defined')
        channels = data.get('channels', ['Email Marketing'])
        channels_str = ', '.join(channels)
        duration = data.get('duration', '8 weeks')
        budget_display = data.get('budget', 'TBD')
        currency = data.get('currency', 'USD')
        current_date = datetime.now().strftime('%B %d, %Y')
        
        creative_budget = budget_num * 0.25
        media_budget = budget_num * 0.45
        tech_budget = budget_num * 0.20
        analytics_budget = budget_num * 0.10
        
        strategy_text = f"# 🚀 {company_name} - {campaign_type} Campaign Strategy\n\n"
        strategy_text += "## 📊 Executive Summary\n"
        strategy_text += "| Metric | Value |\n"
        strategy_text += "|-----------|-------|\n"
        strategy_text += f"| **Campaign Type** | {campaign_type} |\n"
        strategy_text += f"| **Target Market** | {location} - {customer_segment} |\n"
        strategy_text += f"| **Budget** | {budget_display} {currency} |\n"
        strategy_text += f"| **Duration** | {duration} |\n"
        strategy_text += f"| **Channels** | {channels_str} |\n\n"
        
        strategy_text += "## 👥 Target Audience Analysis\n"
        strategy_text += f"🎯 **Primary Audience:** {target_audience}\n\n"
        strategy_text += f"**📍 Geographic Focus:** {location}\n"
        strategy_text += f"**💼 Customer Segment:** {customer_segment}\n\n"
        
        strategy_text += "## 📢 Channel Strategy Dashboard\n"
        strategy_text += "```
        strategy_text += "📧 Email Marketing     ████████████ 85%\n"
        strategy_text += "📱 Social Media       ████████░░░░ 70%\n"
        strategy_text += "🎯 Google Ads         ██████░░░░░░ 60%\n"
        strategy_text += "📊 Content Marketing  ████████░░░░ 75%\n"
        strategy_text += "```\n\n"
        
        strategy_text += "### 📧 Email Marketing Strategy\n"
        strategy_text += "- 👋 Welcome series for new subscribers\n"
        strategy_text += "- 🚀 Promotional campaigns for product launches\n"
        strategy_text += "- 🔄 Re-engagement campaigns for inactive users\n"
        strategy_text += "- 🎯 Personalized product recommendations\n\n"
        
        strategy_text += "## 📅 Implementation Timeline\n"
        strategy_text += "```
        strategy_text += "Phase 1 (Weeks 1-2): 🎯 Strategy & Assets\n"
        strategy_text += "├── Strategy finalization\n"
        strategy_text += "├── Creative asset development\n"
        strategy_text += "└── Campaign setup\n\n"
        strategy_text += "Phase 2 (Weeks 3-4): 🚀 Launch & Optimize\n"
        strategy_text += "├── Campaign launch\n"
        strategy_text += "├── Performance monitoring\n"
        strategy_text += "└── Initial optimization\n\n"
        strategy_text += "Phase 3 (Weeks 5-6): 📈 Scale & Expand\n"
        strategy_text += "├── Performance scaling\n"
        strategy_text += "├── Additional channel activation\n"
        strategy_text += "└── Content expansion\n\n"
        strategy_text += "Phase 4 (Weeks 7-8): 📊 Analyze & Plan\n"
        strategy_text += "├── Comprehensive analysis\n"
        strategy_text += "├── ROI calculation\n"
        strategy_text += "└── Next campaign planning\n"
        strategy_text += "```\n\n"
        
        strategy_text += "## 💰 Budget Allocation Breakdown\n"
        strategy_text += "| Category | Percentage | Amount |\n"
        strategy_text += "|----------|------------|--------|\n"
        strategy_text += f"| 🎨 Creative Development | 25% | {creative_budget:.0f} |\n"
        strategy_text += f"| 📺 Media/Advertising | 45% | {media_budget:.0f} |\n"
        strategy_text += f"| 🔧 Technology & Tools | 20% | {tech_budget:.0f} |\n"
        strategy_text += f"| 📊 Analytics & Optimization | 10% | {analytics_budget:.0f} |\n\n"
        
        strategy_text += "## 📈 Success Metrics Dashboard\n"
        strategy_text += "- **👥 Reach:** Target audience exposure tracking\n"
        strategy_text += "- **💬 Engagement:** Click-through rates and interactions\n"
        strategy_text += "- **💰 Conversions:** Lead generation and sales metrics\n"
        strategy_text += "- **📊 ROI:** Return on advertising spend analysis\n\n"
        
        strategy_text += "## 🚀 Next Steps Checklist\n"
        strategy_text += "- [ ] ✅ Approve campaign strategy and budget\n"
        strategy_text += "- [ ] 🎨 Develop creative assets and content\n"
        strategy_text += "- [ ] 📊 Set up tracking and analytics systems\n"
        strategy_text += "- [ ] 🚀 Launch pilot campaign phase\n"
        strategy_text += "- [ ] 📈 Monitor performance and optimize continuously\n\n"
        
        strategy_text += "---\n"
        strategy_text += f"*🗓️ Campaign strategy generated on {current_date}*\n"
        strategy_text += "*🤖 Powered by AI Marketing Intelligence*"
        
        return strategy_text
    
    def _fallback_email_template(self, template_type, tone, format_type):
        """Clean fallback email template"""
        if format_type == "HTML":
            html_template = f'''<!DOCTYPE html>
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
</html>'''
            return html_template
        else:
            text_template = f'''Subject: Exclusive {template_type} for {{{{first_name}}}}

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
You received this email because you're subscribed to our updates.'''
            return text_template

# ================================
# FIXED IMAGE GENERATOR 
# ================================

class FixedImageGenerator:
    """Generate images using your specific model from secrets"""
    
    def __init__(self):
        self.hugging_face_token = HUGGING_FACE_TOKEN
        self.model = st.secrets.get("HF_MODEL", "black-forest-labs/FLUX.1-schnell")
        
    def generate_campaign_image_via_api(self, campaign_description, style="professional"):
        """Generate campaign image using your specific HF model from secrets"""
        if not self.hugging_face_token:
            st.warning("⚠️ HuggingFace token not configured for image generation")
            return self.generate_placeholder_image(campaign_description)
        
        try:
            enhanced_prompt = f"Professional marketing campaign image for {campaign_description}, {style} style, high quality, vibrant colors, modern design, commercial photography, eye-catching, brand advertisement, 4K resolution, clean layout"
            
            API_URL = f"https://api-inference.huggingface.co/models/{self.model}"
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
            
            with st.spinner(f"🎨 Generating image with {self.model}..."):
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    image_bytes = response.content
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Store in session state
                    image_data = {
                        'prompt': enhanced_prompt,
                        'timestamp': datetime.now(),
                        'campaign': campaign_description,
                        'image': image,
                        'model': self.model
                    }
                    
                    st.session_state.generated_images.append(image_data)
                    
                    st.success(f"✨ Campaign image generated successfully!")
                    st.image(image, caption=f"Generated for: {campaign_description}", use_container_width=True)
                    
                    # Download option
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
                    st.info("💡 Falling back to placeholder image...")
                    return self.generate_placeholder_image(campaign_description)
                    
        except Exception as e:
            st.error(f"❌ Error generating image: {str(e)}")
            return self.generate_placeholder_image(campaign_description)
    
    def generate_placeholder_image(self, campaign_description):
        """Generate a professional-looking placeholder image"""
        try:
            width, height = 512, 512
            image = Image.new('RGB', (width, height), color='#1e3a8a')
            draw = ImageDraw.Draw(image)
            
            # Create gradient effect
            for y in range(height):
                color_value = int(30 + (y / height) * 60)
                for x in range(width):
                    draw.point((x, y), fill=(color_value, color_value + 20, color_value + 60))
            
            title = "🚀 CAMPAIGN IMAGE"
            subtitle = campaign_description[:60] + "..." if len(campaign_description) > 60 else campaign_description
            
            try:
                title_font = ImageFont.truetype("arial.ttf", 32)
                subtitle_font = ImageFont.truetype("arial.ttf", 18)
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
            
            # Title
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (width - title_width) // 2
            draw.text((title_x, height//2 - 60), title, fill='white', font=title_font)
            
            # Subtitle
            wrapped_text = textwrap.fill(subtitle, width=40)
            subtitle_bbox = draw.textbbox((0, 0), wrapped_text, font=subtitle_font)
            subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
            subtitle_x = (width - subtitle_width) // 2
            draw.text((subtitle_x, height//2 + 20), wrapped_text, fill='#e0e7ff', font=subtitle_font)
            
            image_data = {
                'prompt': f"Professional placeholder for: {campaign_description}",
                'timestamp': datetime.now(),
                'campaign': campaign_description,
                'image': image,
                'model': 'placeholder'
            }
            
            st.session_state.generated_images.append(image_data)
            
            st.success("📷 Generated professional placeholder campaign image")
            st.image(image, caption=f"Placeholder for: {campaign_description}", use_container_width=True)
            
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            st.download_button(
                "📥 Download Placeholder Image",
                data=img_bytes.getvalue(),
                file_name=f"placeholder_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
            
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
# FIXED YAGMAIL EMAIL HANDLER
# ================================

class FixedYagmailHandler:
    """WORKING bulk and single email sending with yagmail"""
    
    def __init__(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.yag = None
        
    def initialize_connection(self):
        """Initialize yagmail connection"""
        try:
            self.yag = yagmail.SMTP(user=self.sender_email, password=self.sender_password)
            return True, "Connection successful"
        except Exception as e:
            return False, f"Failed to connect: {str(e)}"
    
    def validate_email_address(self, email):
        """Validate email address format"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    def send_single_email(self, to_email, subject, body):
        """Send a single email"""
        try:
            if not self.yag:
                success, msg = self.initialize_connection()
                if not success:
                    return False, msg
            
            self.yag.send(to=to_email, subject=subject, contents=body)
            return True, "Success"
        except Exception as e:
            return False, f"Failed to send: {str(e)}"
    
    def send_bulk_emails(self, email_list, subject, body_template, personalizer):
        """WORKING bulk email sending function with proper progress tracking"""
        try:
            if not self.yag:
                success, msg = self.initialize_connection()
                if not success:
                    st.error(f"❌ Connection failed: {msg}")
                    return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Email setup failed: {str(e)}")
            return pd.DataFrame()
        
        total_emails = len(email_list)
        results = []
        
        # Create progress tracking elements
        progress_container = st.empty()
        status_container = st.empty()
        metrics_container = st.empty()
        
        sent_count = 0
        failed_count = 0
        invalid_count = 0
        
        try:
            for index, row in email_list.iterrows():
                current_progress = (index + 1) / total_emails
                
                # Update progress
                with progress_container.container():
                    st.progress(current_progress)
                
                with status_container.container():
                    st.info(f"📧 Sending {index + 1}/{total_emails}: {row['email']}")
                
                try:
                    # Validate email
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
                    
                    # Personalize content
                    name = row.get('name', personalizer.extract_name_from_email(row['email']))
                    personal_subject = personalizer.personalize_template(subject, name, row['email'])
                    personal_body = personalizer.personalize_template(body_template, name, row['email'])
                    
                    # Send email using yagmail
                    self.yag.send(to=row['email'], subject=personal_subject, contents=personal_body)
                    
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
                with metrics_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("✅ Sent", sent_count)
                    col2.metric("❌ Failed", failed_count)
                    col3.metric("⚠️ Invalid", invalid_count)
                    col4.metric("📊 Progress", f"{current_progress*100:.0f}%")
                
                # Rate limiting
                time.sleep(1)
            
            # Final status
            with progress_container.container():
                st.progress(1.0)
            with status_container.container():
                st.success("🎉 Bulk email campaign completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Bulk email error: {str(e)}")
            return pd.DataFrame()
        
        return pd.DataFrame(results)

# ================================
# ENHANCED DATA PROCESSOR
# ================================

class EnhancedDataProcessor:
    """Process multiple file formats and data sources"""
    
    def __init__(self):
        self.personalizer = EmailPersonalizer()
    
    def process_uploaded_data(self, uploaded_files):
        """Process various file formats for data analysis"""
        all_data = []
        file_info = []
        
        for uploaded_file in uploaded_files:
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                file_info.append(f"File: {uploaded_file.name}, Size: {uploaded_file.size} bytes")
                
                if file_extension == 'csv':
                    df = pd.read_csv(uploaded_file)
                    all_data.append(('CSV', uploaded_file.name, df))
                
                elif file_extension in ['xlsx', 'xls']:
                    df = pd.read_excel(uploaded_file)
                    all_data.append(('Excel', uploaded_file.name, df))
                
                elif file_extension == 'json':
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                    all_data.append(('JSON', uploaded_file.name, df))
                
                else:
                    st.warning(f"Unsupported file format: {file_extension}")
                    continue
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        return all_data, file_info
    
    def process_google_sheets_data(self, sheet_url):
        """Process Google Sheets data for analysis"""
        try:
            sheet_id = self._extract_sheet_id(sheet_url)
            if not sheet_id:
                return None, "Invalid Google Sheets URL"
            
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            df = pd.read_csv(csv_url)
            
            return df, f"Google Sheets: {sheet_id}"
            
        except Exception as e:
            return None, f"Error accessing Google Sheets: {str(e)}"
    
    def process_file(self, uploaded_file):
        """Process uploaded contact files"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload CSV or Excel files only")
                return None
            
            return self._standardize_contacts(df)
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    
    def process_bulk_paste(self, bulk_text):
        """Process bulk pasted email data"""
        try:
            lines = bulk_text.strip().split('\n')
            contacts = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse different formats
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        email_part = parts[0] if '@' in parts[0] else parts[1]
                        name_part = parts[1] if '@' in parts[0] else parts[0]
                    else:
                        email_part = parts[0]
                        name_part = self.personalizer.extract_name_from_email(email_part)
                elif '\t' in line:
                    parts = [p.strip() for p in line.split('\t')]
                    email_part = parts[0] if '@' in parts[0] else parts[1] if len(parts) > 1 else parts[0]
                    name_part = parts[1] if '@' in parts[0] and len(parts) > 1 else self.personalizer.extract_name_from_email(email_part)
                else:
                    email_part = line
                    name_part = self.personalizer.extract_name_from_email(email_part)
                
                try:
                    validate_email(email_part)
                    contacts.append({'email': email_part.lower(), 'name': name_part})
                except EmailNotValidError:
                    continue
            
            if not contacts:
                st.error("No valid emails found in pasted text")
                return None
                
            return pd.DataFrame(contacts)
            
        except Exception as e:
            st.error(f"Error processing pasted data: {e}")
            return None
    
    def _extract_sheet_id(self, url):
        """Extract sheet ID from Google Sheets URL"""
        try:
            if '/spreadsheets/d/' in url:
                return url.split('/spreadsheets/d/')[1].split('/')[0]
            return None
        except:
            return None
    
    def _standardize_contacts(self, df):
        """Standardize contact data format"""
        df.columns = df.columns.str.lower()
        
        email_col = None
        name_col = None
        
        for col in df.columns:
            if any(keyword in col for keyword in ['email', 'mail', 'e-mail']):
                email_col = col
                break
        
        for col in df.columns:
            if any(keyword in col for keyword in ['name', 'first', 'last', 'full']):
                name_col = col
                break
        
        if email_col is None:
            st.error("❌ No email column found. Please ensure your data has an 'email' column.")
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
    
    .email-config-box {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# MAIN APPLICATION
# ================================

def main():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0;">🚀 Marketing Campaign War Room</h1>
        <p style="font-size: 1.3rem; color: #888; margin-top: 0;">AI-Powered Campaign Generation, Email Marketing & Data Analytics Platform</p>
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
        
        if HUGGING_FACE_TOKEN:
            st.success(f"🎨 Image Generator: Connected ({st.secrets.get('HF_MODEL', 'Default')})")
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
    """Enhanced campaign dashboard"""
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
            generate_campaign = st.form_submit_button("🚀 Generate Enhanced AI Campaign Strategy", use_container_width=True)
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
        
        with st.spinner(f"🤖 Groq AI ({GROQ_MODEL}) is generating your enhanced campaign strategy..."):
            generator = GroqCampaignGenerator()
            strategy = generator.generate_campaign_strategy(campaign_data)
            
            st.session_state.current_campaign = campaign_data
            st.session_state.campaign_blueprint = strategy
            
            st.success("✨ Enhanced campaign strategy generated successfully!")
            st.balloons()
    
    # Handle image generation
    if generate_image and st.session_state.current_campaign:
        image_gen = FixedImageGenerator()
        campaign_desc = f"{st.session_state.current_campaign['company_name']} {st.session_state.current_campaign['campaign_type']}"
        image_gen.generate_campaign_image_via_api(campaign_desc, "professional")
    
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
                    file_name=f"{st.session_state.current_campaign['company_name']}_enhanced_strategy.md",
                    mime="text/markdown",
                    use_container_width=True)

def show_email_marketing():
    """FIXED email marketing page"""
    st.header("📧 Comprehensive Email Marketing Center")
    
    if st.session_state.current_campaign:
        st.success(f"🎯 Active Campaign: **{st.session_state.current_campaign['company_name']}** - {st.session_state.current_campaign['campaign_type']}")
    
    # Email template generation
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
        
        if st.button("🚀 Generate Clean AI Email Template", use_container_width=True):
            generator = GroqCampaignGenerator()
            campaign_context = f"{st.session_state.current_campaign['company_name']} {st.session_state.current_campaign['campaign_type']}" if st.session_state.current_campaign else None
            
            with st.spinner(f"🤖 Generating clean {content_format.lower()}..."):
                template = generator.generate_email_template(email_type, tone, content_format, campaign_context)
                
                if content_format == "HTML Template":
                    st.session_state.email_template_html = template
                else:
                    st.session_state.email_template_text = template
                
                st.success("✨ Clean email template generated successfully!")
    
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
    
    # Email Configuration
    st.subheader("📧 Email Configuration")
    st.markdown('<div class="email-config-box">', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        sender_email = st.text_input("📧 Company Gmail Address", 
                                   placeholder="your-company@gmail.com",
                                   help="Use your company Gmail address")
    with config_col2:
        sender_password = st.text_input("🔑 Gmail App Password", 
                                      type="password", 
                                      help="Generate app password from Gmail settings > Security > App passwords")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if sender_email and sender_password:
        if st.button("🔍 Test Email Configuration"):
            try:
                test_handler = FixedYagmailHandler(sender_email, sender_password)
                success, msg = test_handler.initialize_connection()
                if success:
                    st.success("✅ Email configuration successful!")
                else:
                    st.error(f"❌ Email configuration failed: {msg}")
            except Exception as e:
                st.error(f"❌ Configuration error: {str(e)}")
    
    st.markdown("---")
    
    # Contact Data Management
    st.subheader("👥 Contact Data Management")
    
    contact_method = st.radio("📥 Choose Contact Input Method:", 
                             ["📁 Upload File (CSV/Excel)", "📋 Bulk Paste", "🌐 Google Forms/Sheets"])
    
    if contact_method == "📁 Upload File (CSV/Excel)":
        uploaded_file = st.file_uploader("Upload Contact File", 
                                       type=['csv', 'xlsx'], 
                                       help="Upload a CSV or Excel file with 'email' and 'name' columns")
        
        if uploaded_file:
            processor = EnhancedDataProcessor()
            contacts = processor.process_file(uploaded_file)
            
            if contacts is not None:
                st.session_state.email_contacts = contacts
                st.success(f"✅ Successfully loaded {len(contacts)} valid contacts!")
    
    elif contact_method == "📋 Bulk Paste":
        st.info("💡 Paste email addresses or email,name pairs (one per line)")
        bulk_text = st.text_area("Paste Contact Data:", 
                                height=200,
                                placeholder="""john.doe@example.com, John Doe
jane.smith@example.com, Jane Smith
mark.wilson@example.com""")
        
        if st.button("🔄 Process Pasted Data") and bulk_text:
            processor = EnhancedDataProcessor()
            contacts = processor.process_bulk_paste(bulk_text)
            
            if contacts is not None:
                st.session_state.email_contacts = contacts
                st.success(f"✅ Successfully processed {len(contacts)} valid contacts!")
    
    elif contact_method == "🌐 Google Forms/Sheets":
        st.info("💡 Make sure your Google Sheet is publicly accessible (Anyone with link can view)")
        sheet_url = st.text_input("Google Sheets URL:", 
                                 placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit")
        
        if st.button("📊 Load from Google Sheets") and sheet_url:
            processor = EnhancedDataProcessor()
            df, msg = processor.process_google_sheets_data(sheet_url)
            
            if df is not None:
                contacts_df = processor._standardize_contacts(df)
                if contacts_df is not None:
                    st.session_state.email_contacts = contacts_df
                    st.success(f"✅ Successfully loaded {len(contacts_df)} contacts from Google Sheets!")
            else:
                st.error(f"❌ {msg}")
    
    # Show and edit contacts
    if st.session_state.email_contacts is not None:
        st.markdown("---")
        st.subheader("📋 Contact List Editor")
        
        edited_contacts = st.data_editor(
            st.session_state.email_contacts,
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
    
    # FIXED Bulk Email Campaign
    if (st.session_state.email_contacts is not None and 
        (st.session_state.email_template_html or st.session_state.email_template_text) and
        sender_email and sender_password):
        
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
                value="Important message for {{first_name}}")
            test_email = st.text_input("🧪 Test Email Address", placeholder="test@email.com")
        
        with config_col2:
            if st.session_state.email_template_html and st.session_state.email_template_text:
                send_format = st.radio("📝 Send As:", ["HTML", "Plain Text"])
                template_to_use = st.session_state.email_template_html if send_format == "HTML" else st.session_state.email_template_text
            elif st.session_state.email_template_html:
                template_to_use = st.session_state.email_template_html
                st.info("✅ HTML template ready")
            else:
                template_to_use = st.session_state.email_template_text
                st.info("✅ Plain text template ready")
        
        # Test email
        if test_email and st.button("🧪 Send Test Email"):
            try:
                test_handler = FixedYagmailHandler(sender_email, sender_password)
                personalizer = EmailPersonalizer()
                
                test_content = personalizer.personalize_template(template_to_use, "Test User", test_email)
                test_subject = personalizer.personalize_template(bulk_subject, "Test User", test_email)
                
                with st.spinner(f"🧪 Sending test email to {test_email}..."):
                    success, error_msg = test_handler.send_single_email(test_email, test_subject, test_content)
                
                if success:
                    st.success("✅ Test email sent successfully!")
                else:
                    st.error(f"❌ Test failed: {error_msg}")
                    
            except Exception as e:
                st.error(f"❌ Test error: {str(e)}")
        
        # WORKING Bulk Campaign Launch
        st.markdown("### 🎯 Campaign Launch")
        
        if st.button("🚀 LAUNCH BULK EMAIL CAMPAIGN", type="primary", use_container_width=True):
            st.warning(f"⚠️ About to send {len(df)} personalized emails using yagmail. This action cannot be undone!")
            
            if st.button("✅ CONFIRM & SEND ALL EMAILS", key="confirm_bulk_send"):
                st.info("🚀 Starting bulk email campaign with fixed yagmail...")
                
                try:
                    email_handler = FixedYagmailHandler(sender_email, sender_password)
                    personalizer = EmailPersonalizer()
                    
                    results = email_handler.send_bulk_emails(
                        df, bulk_subject, template_to_use, personalizer
                    )
                    
                    if not results.empty:
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
                        
                        st.session_state.campaign_results = results
                        
                        csv_data = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Campaign Results",
                            data=csv_data,
                            file_name=f"fixed_campaign_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        with st.expander("📋 View Detailed Campaign Results"):
                            st.dataframe(results, use_container_width=True)
                        
                        if success_count > 0:
                            st.balloons()
                    
                    else:
                        st.error("❌ Campaign failed - no results generated")
                        
                except Exception as e:
                    st.error(f"❌ Campaign error: {str(e)}")
    
    elif st.session_state.email_contacts is not None and (st.session_state.email_template_html or st.session_state.email_template_text):
        st.warning("⚠️ Please configure your email settings above to launch campaigns")
    
    elif sender_email and sender_password:
        st.info("💡 Load some contacts and generate an email template to start your campaign!")
    
    else:
        st.info("📧 **Getting Started with Fixed Email Marketing:** 1. **Configure Email Settings** - Enter your company Gmail and app password, 2. **Generate Clean Email Template** - Use AI to create templates without instructions, 3. **Load Contacts** - Upload files, paste data, or connect to Google Forms, 4. **Launch WORKING Campaign** - Send personalized bulk emails with real-time tracking. All emails sent using **fixed yagmail** for reliable delivery!")

def show_analytics_reports():
    """Enhanced analytics with data upload and AI analysis"""
    st.header("📊 Campaign Analytics & Data Intelligence Platform")
    
    # Data Upload and Analysis Section
    st.subheader("📁 Upload Data for AI Analysis")
    
    analysis_method = st.radio("📈 Choose Data Source:", [
        "📁 Upload Files (CSV, Excel, JSON)", 
        "🌐 Google Sheets URL",
        "🔗 Direct Data URL"
    ])
    
    if analysis_method == "📁 Upload Files (CSV, Excel, JSON)":
        uploaded_files = st.file_uploader(
            "Upload Data Files for Analysis", 
            type=['csv', 'xlsx', 'xls', 'json'],
            accept_multiple_files=True,
            help="Upload multiple files for comprehensive analysis"
        )
        
        if uploaded_files and st.button("🤖 Analyze Data with Groq AI"):
            with st.spinner("🔍 Processing files and generating AI analysis..."):
                processor = EnhancedDataProcessor()
                all_data, file_info = processor.process_uploaded_data(uploaded_files)
                
                if all_data:
                    st.success(f"✅ Successfully processed {len(all_data)} files!")
                    
                    # Analyze each dataset
                    generator = GroqCampaignGenerator()
                    
                    for file_type, filename, df in all_data:
                        st.markdown(f"---")
                        st.markdown(f"### 📊 Analysis of {filename} ({file_type})")
                        
                        # Show data preview
                        with st.expander(f"📋 Data Preview - {filename}"):
                            st.dataframe(df.head(10), use_container_width=True)
                            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        # AI Analysis
                        analysis = generator.analyze_data(df, f"{file_type}: {filename}")
                        st.markdown(analysis)
                        
                        # Generate visualizations based on data
                        if not df.empty:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                            
                            if len(numeric_cols) > 0:
                                st.markdown("#### 📈 Automated Visualizations")
                                
                                vis_col1, vis_col2 = st.columns(2)
                                
                                with vis_col1:
                                    if len(numeric_cols) >= 1:
                                        fig = px.histogram(df, x=numeric_cols[0], 
                                                         title=f"Distribution of {numeric_cols[0]}")
                                        fig.update_layout(template="plotly_dark")
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with vis_col2:
                                    if len(numeric_cols) >= 2:
                                        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                                       title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                                        fig.update_layout(template="plotly_dark")
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                                    fig = px.box(df, x=categorical_cols[0], y=numeric_cols[0],
                                               title=f"{numeric_cols[0]} by {categorical_cols[0]}")
                                    fig.update_layout(template="plotly_dark")
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store results
                    st.session_state.uploaded_dataset = all_data
                    st.session_state.data_analysis_result = "Analysis completed"
                
                else:
                    st.error("❌ No data could be processed from uploaded files")
    
    elif analysis_method == "🌐 Google Sheets URL":
        sheet_url = st.text_input("Google Sheets URL:", 
                                 placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit")
        
        if sheet_url and st.button("🤖 Analyze Google Sheets Data"):
            with st.spinner("📊 Loading and analyzing Google Sheets data..."):
                processor = EnhancedDataProcessor()
                df, msg = processor.process_google_sheets_data(sheet_url)
                
                if df is not None:
                    st.success("✅ Google Sheets data loaded successfully!")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                    
                    generator = GroqCampaignGenerator()
                    analysis = generator.analyze_data(df, msg)
                    st.markdown("### 🤖 AI Analysis Results")
                    st.markdown(analysis)
                    
                else:
                    st.error(f"❌ {msg}")
    
    elif analysis_method == "🔗 Direct Data URL":
        data_url = st.text_input("Direct Data URL:", 
                                placeholder="https://example.com/data.csv")
        
        if data_url and st.button("🤖 Analyze URL Data"):
            try:
                with st.spinner("🌐 Loading data from URL..."):
                    if data_url.endswith('.csv'):
                        df = pd.read_csv(data_url)
                    elif data_url.endswith('.json'):
                        df = pd.read_json(data_url)
                    else:
                        st.error("❌ Unsupported URL format. Please use CSV or JSON URLs.")
                        return
                    
                    st.success("✅ Data loaded from URL successfully!")
                    
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    generator = GroqCampaignGenerator()
                    analysis = generator.analyze_data(df, f"URL Data: {data_url}")
                    st.markdown("### 🤖 AI Analysis Results")
                    st.markdown(analysis)
                    
            except Exception as e:
                st.error(f"❌ Error loading data from URL: {str(e)}")
    
    st.markdown("---")
    
    # Campaign Geographic Analysis
    if st.session_state.current_campaign:
        st.subheader("🗺️ Campaign Geographic Analysis")
        
        campaign = st.session_state.current_campaign
        location = campaign['location']
        
        if location in COUNTRIES_DATA:
            coords = COUNTRIES_DATA[location]['coords']
            
            map_data = pd.DataFrame({
                'lat': [coords[0]],
                'lon': [coords[1]], 
                'location': [location],
                'campaign': [campaign['campaign_type']],
                'company': [campaign['company_name']]
            })
            
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
            
            estimated_reach = budget * 30
            estimated_clicks = int(estimated_reach * 0.04)
            estimated_conversions = int(estimated_clicks * 0.03)
            estimated_revenue = estimated_conversions * 65
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
                    st.image(img_data['image'], caption=f"Campaign Image {i+1}: {img_data['campaign']}", use_container_width=True)
                
                with col2:
                    st.write(f"**Generated:** {img_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Campaign:** {img_data['campaign']}")
                    st.write(f"**Model:** {img_data.get('model', 'Unknown')}")
                    
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
        
        total_sent = len(results_df[results_df['status'] == 'sent'])
        total_failed = len(results_df[results_df['status'] == 'failed'])
        total_invalid = len(results_df[results_df['status'] == 'invalid'])
        success_rate = (total_sent / len(results_df)) * 100 if len(results_df) > 0 else 0
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("📧 Total Emails", len(results_df))
        with perf_col2:
            st.metric("✅ Successfully Delivered", total_sent, delta=f"{success_rate:.1f}%")
        with perf_col3:
            st.metric("❌ Failed Deliveries", total_failed)
        with perf_col4:
            st.metric("⚠️ Invalid Addresses", total_invalid)
        
        col1, col2 = st.columns(2)
        
        with col1:
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
    
    else:
        st.info("📊 **Enhanced Analytics Dashboard** **🤖 AI-Powered Data Analysis:** - Upload CSV, Excel, JSON files for analysis - Connect to Google Sheets for real-time data - Get AI-powered insights and visualizations from Groq - Automated data quality checks and recommendations **🗺️ Geographic Campaign Analysis:** - Interactive campaign targeting maps - Location-based performance insights **📈 Performance Projections:** - ROI calculations and forecasts - Estimated reach and conversion metrics **📧 Email Campaign Analytics:** - Real-time delivery tracking with fixed yagmail - Success rate analysis and domain breakdown **🎨 Creative Asset Management:** - Generated campaign images with your HF model - Asset download and management tools Upload data or create campaigns to unlock powerful AI analytics insights!")

if __name__ == "__main__":
    main()
