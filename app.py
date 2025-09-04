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
from datetime import datetime, timedelta
import io
import base64
from groq import Groq
import requests
from PIL import Image, ImageDraw, ImageFont
import textwrap

# ================================
# CONFIGURATION (EXACT AS SPECIFIED)
# ================================

# Groq Configuration
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"

# Gmail SMTP Configuration  
GMAIL_EMAIL = "rushil.cat23@gmail.com"
GMAIL_APP_PASSWORD = st.secrets.get("GMAIL_APP_PASSWORD", "")

# Hugging Face Configuration
HUGGING_FACE_TOKEN = st.secrets.get("HUGGING_FACE_TOKEN", "")
HF_MODEL = "black-forest-labs/FLUX.1-dev"

# App Configuration
st.set_page_config(
    page_title="Marketing Campaign War Room",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
COUNTRIES = ["Global", "United States", "Canada", "United Kingdom", "Germany", "France", "Spain", "Italy", "Netherlands", "Australia", "Japan", "India", "China", "Brazil", "Mexico"]
CURRENCIES = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "INR", "BRL", "MXN", "CNY"]

# ================================
# SESSION STATE INITIALIZATION
# ================================

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = [
        'current_page', 'campaign_data', 'campaign_strategy', 'email_template_html', 
        'email_template_text', 'email_contacts', 'campaign_results', 'generated_images', 
        'data_analysis_results', 'sender_email', 'sender_password'
    ]
    
    defaults = {
        'current_page': "Campaign Dashboard",
        'generated_images': [],
        'sender_email': GMAIL_EMAIL,
        'sender_password': ""
    }
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = defaults.get(var, None)

# ================================
# GROQ AI FUNCTIONS
# ================================

def generate_campaign_strategy_with_groq(campaign_data):
    """Generate comprehensive campaign strategy using Groq API"""
    if not GROQ_API_KEY:
        return generate_fallback_strategy(campaign_data)
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        channels_str = ', '.join(campaign_data.get('channels', ['Email Marketing']))
        
        prompt = f"""Create a comprehensive, professional marketing campaign strategy for:

CAMPAIGN DETAILS:
- Company: {campaign_data.get('company_name', 'Company')}
- Campaign Type: {campaign_data.get('campaign_type', 'Marketing Campaign')}
- Target Audience: {campaign_data.get('target_audience', 'General audience')}
- Geographic Focus: {campaign_data.get('location', 'Global')}
- Marketing Channels: {channels_str}
- Budget: {campaign_data.get('budget', 'TBD')} {campaign_data.get('currency', 'USD')}
- Duration: {campaign_data.get('duration', 'TBD')}
- Customer Segment: {campaign_data.get('customer_segment', 'Mass Market')}
- Product/Service: {campaign_data.get('product_description', 'Product/Service')}

Please provide a detailed, actionable strategy with:
1. Executive Summary with key metrics
2. Target Audience Analysis with demographics
3. Competitive Positioning
4. Messaging Strategy with key themes
5. Channel-Specific Tactics with implementation details
6. Content Strategy and calendar
7. Timeline & Milestones with specific dates
8. Budget Allocation with detailed breakdown
9. Success Metrics & KPIs
10. Risk Management strategies
11. Next Steps with priority actions

Use emojis, tables, and clear formatting for better readability. Make this practical and actionable."""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class marketing strategist with 20+ years of experience. Create detailed, actionable marketing campaigns with specific tactics and measurable outcomes."
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
        return generate_fallback_strategy(campaign_data)

def generate_email_template_with_groq(template_type, tone, format_type, campaign_context=None):
    """Generate clean email templates using Groq AI"""
    if not GROQ_API_KEY:
        return generate_fallback_email_template(template_type, tone, format_type)
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        context_info = f"Campaign Context: {campaign_context}" if campaign_context else "General marketing campaign"
        
        prompt = f"""Create a professional {template_type.lower()} email template with a {tone.lower()} tone.
Format: {format_type}
{context_info}

Requirements:
1. Include personalization placeholders: {{{{first_name}}}}, {{{{name}}}}, {{{{email}}}}
2. Make it engaging and action-oriented
3. Include a clear call-to-action
4. Use modern, professional design
5. Make it mobile-friendly
6. IMPORTANT: Provide ONLY the clean template content without any explanations or instructions

{"Generate HTML email with inline CSS styling - ONLY the HTML code" if format_type == "HTML" else "Generate plain text email with proper formatting - ONLY the email text"}"""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert email marketing specialist. Create ONLY clean, ready-to-use email templates without any explanations. Provide ONLY the template content."
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
        return generate_fallback_email_template(template_type, tone, format_type)

def analyze_data_with_groq(df, file_info):
    """Analyze uploaded data using Groq AI"""
    if not GROQ_API_KEY:
        return "Groq AI not available for data analysis. Please configure API key."
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Safe JSON serialization
        def safe_json_serializable(o):
            if isinstance(o, (pd.Timestamp, pd.Timedelta)):
                return str(o)
            elif pd.isna(o):
                return None
            raise TypeError(f"Type {type(o)} not serializable")
        
        sample_data = df.head(5).to_dict(orient='records')
        
        prompt = f"""You are a professional data analyst. Analyze this dataset:

FILE INFO: {file_info}
DATASET SHAPE: {df.shape[0]} rows, {df.shape[1]} columns
COLUMNS: {list(df.columns)}
SAMPLE DATA (first 5 rows):
{json.dumps(sample_data, indent=2, default=safe_json_serializable)}

Please provide a comprehensive analysis including:

1. **DATA SUMMARY & STATISTICS**
2. **KEY INSIGHTS & TRENDS** 
3. **RECOMMENDED VISUALIZATIONS**
   - Suggest specific chart types
   - Recommend columns to visualize together
4. **DATA QUALITY ASSESSMENT**
5. **BUSINESS RECOMMENDATIONS**
   - Actionable insights for management
   - Strategic recommendations

Format with clear headings and bullet points."""

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data analyst. Provide thorough, actionable analysis with clear insights and strategic recommendations."
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

def generate_fallback_strategy(campaign_data):
    """Fallback strategy when Groq AI is not available"""
    company = campaign_data.get('company_name', 'Your Company')
    campaign_type = campaign_data.get('campaign_type', 'Marketing Campaign')
    budget = campaign_data.get('budget', '10000')
    
    try:
        budget_num = int(budget) if budget.isdigit() else 10000
    except:
        budget_num = 10000
    
    return f"""# 🚀 {company} - {campaign_type} Strategy

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Campaign Type** | {campaign_type} |
| **Target Market** | {campaign_data.get('location', 'Global')} |
| **Budget** | {budget} {campaign_data.get('currency', 'USD')} |
| **Duration** | {campaign_data.get('duration', '8 weeks')} |
| **Channels** | {', '.join(campaign_data.get('channels', ['Email Marketing']))} |

## 👥 Target Audience Analysis
🎯 **Primary Audience:** {campaign_data.get('target_audience', 'Target audience to be defined')}

**📍 Geographic Focus:** {campaign_data.get('location', 'Global')}
**💼 Customer Segment:** {campaign_data.get('customer_segment', 'Mass Market')}

## 📢 Channel Strategy
Selected Channels: {', '.join(campaign_data.get('channels', ['Email Marketing']))}

### Email Marketing Strategy
- 👋 Welcome series for new subscribers  
- 🚀 Promotional campaigns for product launches
- 🔄 Re-engagement campaigns for inactive users
- 🎯 Personalized product recommendations

## 💰 Budget Allocation Breakdown

| Category | Percentage | Amount |
|----------|------------|--------|
| 🎨 Creative Development | 25% | ${budget_num * 0.25:,.0f} |
| 📺 Media/Advertising | 45% | ${budget_num * 0.45:,.0f} |
| 🔧 Technology & Tools | 20% | ${budget_num * 0.20:,.0f} |
| 📊 Analytics & Optimization | 10% | ${budget_num * 0.10:,.0f} |

## 📈 Success Metrics Dashboard
- **👥 Reach:** Target audience exposure tracking
- **💬 Engagement:** Click-through rates and interactions
- **💰 Conversions:** Lead generation and sales metrics  
- **📊 ROI:** Return on advertising spend analysis

## 🚀 Next Steps Checklist
- [ ] ✅ Approve campaign strategy and budget
- [ ] 🎨 Develop creative assets and content
- [ ] 📊 Set up tracking and analytics systems
- [ ] 🚀 Launch pilot campaign phase
- [ ] 📈 Monitor performance and optimize continuously

---
*🗓️ Campaign strategy generated on {datetime.now().strftime('%B %d, %Y')}*
*🤖 Powered by AI Marketing Intelligence*"""

def generate_fallback_email_template(template_type, tone, format_type):
    """Fallback email template when Groq AI is not available"""
    if format_type == "HTML":
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{template_type}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #f5f5f5; }}
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
    else:
        return f'''Subject: Exclusive {template_type} for {{{{first_name}}}}

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

# ================================
# IMAGE GENERATION FUNCTIONS
# ================================

def generate_campaign_image_hf(campaign_description):
    """Generate campaign image using HuggingFace FLUX.1-dev model"""
    if not HUGGING_FACE_TOKEN:
        st.warning("⚠️ HuggingFace token not configured")
        return generate_placeholder_image(campaign_description)
    
    try:
        enhanced_prompt = f"Professional marketing campaign image for {campaign_description}, high quality, vibrant colors, modern design, commercial photography, eye-catching, brand advertisement, 4K resolution, clean layout"
        
        API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }
        }
        
        with st.spinner(f"🎨 Generating image with {HF_MODEL}..."):
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
                    'model': HF_MODEL
                }
                
                st.session_state.generated_images.append(image_data)
                
                st.success(f"✨ Campaign image generated successfully!")
                return image
                
            else:
                st.warning(f"Image generation failed: {response.status_code}")
                return generate_placeholder_image(campaign_description)
                
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return generate_placeholder_image(campaign_description)

def generate_placeholder_image(campaign_description):
    """Generate professional placeholder image"""
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
        subtitle = campaign_description[:50] + "..." if len(campaign_description) > 50 else campaign_description
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 28)
            subtitle_font = ImageFont.truetype("arial.ttf", 16)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, height//2 - 60), title, fill='white', font=title_font)
        
        # Draw subtitle
        wrapped_text = textwrap.fill(subtitle, width=35)
        subtitle_bbox = draw.textbbox((0, 0), wrapped_text, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        draw.text((subtitle_x, height//2 + 20), wrapped_text, fill='#e0e7ff', font=subtitle_font)
        
        # Store in session state
        image_data = {
            'prompt': f"Placeholder for: {campaign_description}",
            'timestamp': datetime.now(),
            'campaign': campaign_description,
            'image': image,
            'model': 'placeholder'
        }
        
        st.session_state.generated_images.append(image_data)
        
        st.success("📷 Generated professional placeholder image")
        return image
        
    except Exception as e:
        st.error(f"Error creating placeholder: {e}")
        return None

# ================================
# EMAIL FUNCTIONS
# ================================

def personalize_email_content(template, name, email):
    """Personalize email template with user data"""
    first_name = name.split()[0] if name and ' ' in name else name
    
    personalized = template.replace('{name}', name or 'Valued Customer')
    personalized = personalized.replace('{{name}}', name or 'Valued Customer')
    personalized = personalized.replace('{first_name}', first_name or 'Valued Customer')
    personalized = personalized.replace('{{first_name}}', first_name or 'Valued Customer')
    personalized = personalized.replace('{email}', email or '')
    personalized = personalized.replace('{{email}}', email or '')
    
    return personalized

def extract_name_from_email_address(email):
    """Extract potential name from email address"""
    try:
        local_part = email.split('@')[0]
        name_part = re.sub(r'[0-9._-]', ' ', local_part)
        name_parts = [part.capitalize() for part in name_part.split() if len(part) > 1]
        return ' '.join(name_parts) if name_parts else 'Valued Customer'
    except:
        return 'Valued Customer'

def send_bulk_emails_yagmail(sender_email, sender_password, email_list, subject, body_template):
    """Send bulk emails using yagmail with the specified Gmail configuration"""
    try:
        # Initialize yagmail connection
        yag = yagmail.SMTP(user=sender_email, password=sender_password)
        
        results = []
        total_emails = len(email_list)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        
        sent_count = 0
        failed_count = 0
        invalid_count = 0
        
        for index, row in email_list.iterrows():
            current_progress = (index + 1) / total_emails
            progress_bar.progress(current_progress)
            status_text.info(f"📧 Sending {index + 1}/{total_emails}: {row['email']}")
            
            try:
                # Validate email format
                if not validate_email_format(row['email']):
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
                name = row.get('name', extract_name_from_email_address(row['email']))
                personal_subject = personalize_email_content(subject, name, row['email'])
                personal_body = personalize_email_content(body_template, name, row['email'])
                
                # Send email using yagmail
                yag.send(to=row['email'], subject=personal_subject, contents=personal_body)
                
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
        progress_bar.progress(1.0)
        status_text.success("🎉 Bulk email campaign completed successfully!")
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Email setup failed: {e}")
        return pd.DataFrame()

def validate_email_format(email):
    """Validate email address format"""
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

# ================================
# DATA PROCESSING FUNCTIONS
# ================================

def process_uploaded_data_file(uploaded_file):
    """Process various file formats for data analysis"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            data = json.load(uploaded_file)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.json_normalize(data)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON files.")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def process_contacts_data_file(uploaded_file):
    """Process uploaded contacts file"""
    df = process_uploaded_data_file(uploaded_file)
    if df is None:
        return None
    
    # Standardize contact data format
    df.columns = df.columns.str.lower()
    
    email_col = None
    name_col = None
    
    # Find email column
    for col in df.columns:
        if any(keyword in col for keyword in ['email', 'mail', 'e-mail']):
            email_col = col
            break
    
    # Find name column
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
            name = extract_name_from_email_address(email)
        
        try:
            validate_email(email)
            result_data.append({'email': email, 'name': name})
        except EmailNotValidError:
            continue
    
    if not result_data:
        st.error("❌ No valid emails found")
        return None
    
    return pd.DataFrame(result_data)

def process_bulk_paste_contacts(bulk_text):
    """Process bulk pasted contact data"""
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
                    name_part = extract_name_from_email_address(email_part)
            elif '\t' in line:
                parts = [p.strip() for p in line.split('\t')]
                email_part = parts[0] if '@' in parts[0] else parts[1] if len(parts) > 1 else parts[0]
                name_part = parts[1] if '@' in parts[0] and len(parts) > 1 else extract_name_from_email_address(email_part)
            else:
                email_part = line
                name_part = extract_name_from_email_address(email_part)
            
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

def extract_google_sheet_id(url):
    """Extract Google Sheets ID from URL"""
    try:
        if '/spreadsheets/d/' in url:
            return url.split('/spreadsheets/d/')[1].split('/')[0]
        return None
    except:
        return None

# ================================
# MAIN APP FUNCTIONS
# ================================

def main():
    """Main application function"""
    initialize_session_state()
    
    # Custom CSS styling
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
    
    # Header
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
            st.error("🤖 Groq AI: Configure API key")
        
        if HUGGING_FACE_TOKEN:
            st.success(f"🎨 Image Generator: Connected ({HF_MODEL})")
        else:
            st.warning("🎨 Image Generator: Configure token")
        
        st.markdown("---")
        
        # Current campaign info
        if st.session_state.campaign_data:
            st.markdown("### 🎯 Active Campaign")
            st.info(f"**{st.session_state.campaign_data['company_name']}**")
            st.caption(f"Type: {st.session_state.campaign_data['campaign_type']}")
            st.caption(f"Location: {st.session_state.campaign_data['location']}")
        
        if st.session_state.email_contacts is not None:
            st.markdown("### 📊 Contact Stats")
            st.info(f"📧 Loaded: {len(st.session_state.email_contacts)} contacts")
    
    # Show current page content
    if st.session_state.current_page == "Campaign Dashboard":
        show_campaign_dashboard_page()
    elif st.session_state.current_page == "Email Marketing":
        show_email_marketing_page()
    elif st.session_state.current_page == "Analytics & Reports":
        show_analytics_reports_page()

def show_campaign_dashboard_page():
    """Campaign dashboard page"""
    st.header("🎯 AI Campaign Strategy Generator")
    st.write("Create comprehensive marketing campaigns powered by Groq AI")
    
    with st.form("campaign_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("🏢 Company Name", 
                value=st.session_state.campaign_data['company_name'] if st.session_state.campaign_data else "")
            
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
            strategy = generate_campaign_strategy_with_groq(campaign_data)
            st.session_state.campaign_data = campaign_data
            st.session_state.campaign_strategy = strategy
            
            st.success("✨ Campaign strategy generated successfully!")
            st.balloons()
    
    # Handle image generation
    if generate_image and st.session_state.campaign_data:
        campaign_desc = f"{st.session_state.campaign_data['company_name']} {st.session_state.campaign_data['campaign_type']}"
        image = generate_campaign_image_hf(campaign_desc)
        
        if image:
            st.image(image, caption=f"Generated for: {campaign_desc}", use_container_width=True)
            
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
    
    # Display existing campaign strategy
    if st.session_state.campaign_strategy:
        st.markdown("---")
        st.markdown("## 📋 Your AI-Generated Campaign Strategy")
        st.markdown(st.session_state.campaign_strategy)
        
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
            if st.session_state.campaign_data:
                st.download_button("📄 Download Strategy", 
                    data=st.session_state.campaign_strategy,
                    file_name=f"{st.session_state.campaign_data['company_name']}_campaign_strategy.md",
                    mime="text/markdown",
                    use_container_width=True)

def show_email_marketing_page():
    """Email marketing page"""
    st.header("📧 Comprehensive Email Marketing Center")
    
    if st.session_state.campaign_data:
        st.success(f"🎯 Active Campaign: **{st.session_state.campaign_data['company_name']}** - {st.session_state.campaign_data['campaign_type']}")
    
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
        content_format = st.radio("📝 Template Format", ["HTML", "Plain Text"])
        
        if st.button("🚀 Generate Clean AI Email Template", use_container_width=True):
            campaign_context = f"{st.session_state.campaign_data['company_name']} {st.session_state.campaign_data['campaign_type']}" if st.session_state.campaign_data else None
            
            with st.spinner(f"🤖 Groq AI ({GROQ_MODEL}) is generating clean email template..."):
                template = generate_email_template_with_groq(email_type, tone, content_format, campaign_context)
                
                if content_format == "HTML":
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
            preview = personalize_email_content(edited_content, "John Smith", "john@example.com")
            st.components.v1.html(preview, height=600, scrolling=True)
    
    st.markdown("---")
    
    # Email Configuration Section using specified Gmail settings
    st.subheader("📧 Email Configuration")
    st.markdown('<div class="email-config-box">', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        sender_email = st.text_input("📧 Gmail Address", 
                                   value=st.session_state.sender_email,
                                   help="Using configured Gmail address")
    with config_col2:
        sender_password = st.text_input("🔑 Gmail App Password", 
                                      type="password",
                                      value=st.session_state.sender_password,
                                      help="Generate app password from Gmail settings > Security > App passwords")
    
    st.session_state.sender_email = sender_email
    st.session_state.sender_password = sender_password
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Test email configuration
    if sender_email and sender_password:
        if st.button("🔍 Test Email Configuration"):
            try:
                yag = yagmail.SMTP(user=sender_email, password=sender_password)
                st.success("✅ Email configuration successful!")
            except Exception as e:
                st.error(f"❌ Email configuration failed: {e}")
    
    st.markdown("---")
    
    # Contact Data Management Section
    st.subheader("👥 Contact Data Management")
    
    # Multiple ways to add contacts
    contact_method = st.radio("📥 Choose Contact Input Method:", 
                             ["📁 Upload File (CSV/Excel)", "📋 Bulk Paste", "🌐 Google Forms/Sheets"])
    
    if contact_method == "📁 Upload File (CSV/Excel)":
        uploaded_file = st.file_uploader("Upload Contact File", 
                                       type=['csv', 'xlsx'], 
                                       help="Upload a CSV or Excel file with 'email' and 'name' columns")
        
        if uploaded_file:
            contacts = process_contacts_data_file(uploaded_file)
            
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
            contacts = process_bulk_paste_contacts(bulk_text)
            
            if contacts is not None:
                st.session_state.email_contacts = contacts
                st.success(f"✅ Successfully processed {len(contacts)} valid contacts!")
    
    elif contact_method == "🌐 Google Forms/Sheets":
        st.info("💡 Make sure your Google Sheet is publicly accessible (Anyone with link can view)")
        sheet_url = st.text_input("Google Sheets URL:", 
                                 placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit#gid=0")
        
        if st.button("📊 Load from Google Sheets") and sheet_url:
            try:
                sheet_id = extract_google_sheet_id(sheet_url)
                if sheet_id:
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                    df = pd.read_csv(csv_url)
                    
                    # Create a temporary file-like object for processing
                    temp_csv = io.StringIO()
                    df.to_csv(temp_csv, index=False)
                    temp_csv.seek(0)
                    
                    # Process as contacts file
                    class MockFile:
                        def __init__(self, content, name):
                            self.content = content
                            self.name = name
                        
                        def read(self):
                            return self.content.getvalue()
                    
                    mock_file = MockFile(temp_csv, "google_sheets.csv")
                    contacts = process_contacts_data_file(mock_file)
                    
                    if contacts is not None:
                        st.session_state.email_contacts = contacts
                        st.success(f"✅ Successfully loaded {len(contacts)} contacts from Google Sheets!")
                else:
                    st.error("❌ Invalid Google Sheets URL")
            except Exception as e:
                st.error(f"❌ Error loading Google Sheets: {e}")
    
    # Show and edit contacts if available
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
    
    # Bulk Email Campaign Section
    if (st.session_state.email_contacts is not None and 
        (st.session_state.email_template_html or st.session_state.email_template_text) and
        sender_email and sender_password):
        
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
            estimated_time = len(df) * 1.5 / 60
            st.metric("⏱️ Est. Time", f"{estimated_time:.0f}m")
        
        # Campaign configuration
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
        
        # Test email functionality
        if test_email and st.button("🧪 Send Test Email"):
            try:
                yag = yagmail.SMTP(user=sender_email, password=sender_password)
                
                test_content = personalize_email_content(template_to_use, "Test User", test_email)
                test_subject = personalize_email_content(bulk_subject, "Test User", test_email)
                
                with st.spinner(f"🧪 Sending test email to {test_email}..."):
                    yag.send(to=test_email, subject=test_subject, contents=test_content)
                
                st.success("✅ Test email sent successfully!")
                    
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
        
        # Bulk campaign launch
        st.markdown("### 🎯 Campaign Launch")
        
        if st.button("🚀 LAUNCH BULK EMAIL CAMPAIGN", type="primary", use_container_width=True):
            st.warning(f"⚠️ About to send {len(df)} personalized emails using yagmail. This action cannot be undone!")
            
            if st.button("✅ CONFIRM & SEND ALL EMAILS", key="confirm_bulk_send"):
                st.info("🚀 Starting bulk email campaign with yagmail...")
                
                try:
                    results = send_bulk_emails_yagmail(
                        sender_email, sender_password, df, bulk_subject, template_to_use
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
                        
                        # Download results
                        csv_data = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Campaign Results",
                            data=csv_data,
                            file_name=f"campaign_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
        st.info("""
        📧 **Getting Started with Email Marketing:**
        
        1. **Configure Email Settings** - Enter your Gmail app password
        2. **Generate Clean Email Template** - Use AI to create templates
        3. **Load Contacts** - Upload files, paste data, or connect to Google Forms
        4. **Launch Campaign** - Send personalized bulk emails with tracking
        
        All emails sent using **yagmail** for reliable delivery!
        """)

def show_analytics_reports_page():
    """Analytics and reports page with enhanced features"""
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
                for uploaded_file in uploaded_files:
                    df = process_uploaded_data_file(uploaded_file)
                    
                    if df is not None:
                        st.markdown(f"---")
                        st.markdown(f"### 📊 Analysis of {uploaded_file.name}")
                        
                        # Show data preview
                        with st.expander(f"📋 Data Preview - {uploaded_file.name}"):
                            st.dataframe(df.head(10), use_container_width=True)
                            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        # AI Analysis using Groq
                        analysis = analyze_data_with_groq(df, f"File: {uploaded_file.name}")
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
    
    elif analysis_method == "🌐 Google Sheets URL":
        sheet_url = st.text_input("Google Sheets URL:", 
                                 placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit")
        
        if sheet_url and st.button("🤖 Analyze Google Sheets Data"):
            with st.spinner("📊 Loading and analyzing Google Sheets data..."):
                try:
                    sheet_id = extract_google_sheet_id(sheet_url)
                    if sheet_id:
                        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                        df = pd.read_csv(csv_url)
                        
                        st.success("✅ Google Sheets data loaded successfully!")
                        
                        with st.expander("📋 Data Preview"):
                            st.dataframe(df.head(10), use_container_width=True)
                            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        # AI Analysis
                        analysis = analyze_data_with_groq(df, f"Google Sheets: {sheet_id}")
                        st.markdown("### 🤖 AI Analysis Results")
                        st.markdown(analysis)
                    else:
                        st.error("❌ Invalid Google Sheets URL")
                        
                except Exception as e:
                    st.error(f"❌ Error loading Google Sheets: {e}")
    
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
                    
                    # AI Analysis
                    analysis = analyze_data_with_groq(df, f"URL Data: {data_url}")
                    st.markdown("### 🤖 AI Analysis Results")
                    st.markdown(analysis)
                    
            except Exception as e:
                st.error(f"❌ Error loading data from URL: {str(e)}")
    
    st.markdown("---")
    
    # Display generated images
    if st.session_state.generated_images:
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
    
    # Email campaign results analysis
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
        st.info("""
        📊 **Enhanced Analytics Dashboard**
        
        **🤖 AI-Powered Data Analysis:**
        - Upload CSV, Excel, JSON files for analysis
        - Connect to Google Sheets for real-time data  
        - Get AI-powered insights from Groq ({})
        - Automated data quality checks and recommendations
        
        **📧 Email Campaign Analytics:**
        - Real-time delivery tracking with yagmail
        - Success rate analysis and domain breakdown
        - Comprehensive performance metrics
        
        **🎨 Creative Asset Management:**
        - Generated campaign images with {}
        - Asset download and management tools
        
        Upload data or create campaigns to unlock powerful AI analytics insights!
        """.format(GROQ_MODEL, HF_MODEL))

if __name__ == "__main__":
    main()
