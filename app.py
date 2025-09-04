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
import PyPDF2
import docx

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
    """Generate campaigns using Groq API with openai/gpt-oss-120b"""
    
    def __init__(self):
        self.client = None
        if GROQ_API_KEY:
            try:
                self.client = Groq(api_key=GROQ_API_KEY)
            except Exception as e:
                st.error(f"Failed to initialize Groq: {e}")
    
    def generate_campaign_strategy(self, campaign_data):
        """Generate comprehensive campaign strategy with visuals using Groq"""
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
            6. IMPORTANT: Do NOT include any instructions, explanations, or meta-text in the output
            7. ONLY provide the clean template content that can be used directly
            
            {"Generate HTML email with inline CSS styling - ONLY the HTML code" if format_type == "HTML" else "Generate plain text email with proper formatting - ONLY the email text"}
            """
            
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
            
            prompt = f"""
            You are a professional data analyst. Analyze the following dataset:
            
            FILE INFO: {file_info}
            SAMPLE DATA (first 10 rows):
            {json.dumps(sample_data, indent=2, default=safe_json_serializable)}
            
            DATASET SHAPE: {df_sample.shape[0]} rows, {df_sample.shape[1]} columns
            COLUMNS: {list(df_sample.columns)}
            
            Please provide a comprehensive analysis including:
            
            1. **DATA SUMMARY & STATISTICS**
            2. **KEY INSIGHTS & TRENDS**
            3. **RECOMMENDED VISUALIZATIONS**
               - Suggest specific chart types for the data
               - Recommend columns to visualize together
            4. **DATA QUALITY ASSESSMENT**
            5. **BUSINESS RECOMMENDATIONS**
               - Actionable insights for management
               - Strategic recommendations based on the data
            
            Format your response with clear headings and bullet points for easy reading.
            """
            
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
        """Build comprehensive campaign prompt with visual elements"""
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
        Create a comprehensive strategy with visual elements including:
        1. **📊 Executive Summary** with key metrics visualization
        2. **👥 Market Analysis** with audience breakdown tables
        3. **🎯 Competitive Positioning** with comparison charts
        4. **💬 Messaging Strategy** with key themes
        5. **📱 Channel-Specific Tactics** with implementation timeline
        6. **📝 Content Strategy** with content calendar
        7. **📅 Timeline & Milestones** with visual project roadmap
        8. **💰 Budget Allocation** with spending breakdown charts
        9. **📈 Success Metrics & KPIs** with measurement framework
        10. **⚠️ Risk Management** with mitigation strategies
        11. **🚀 Next Steps** with priority action items
        
        Use emojis, tables, charts descriptions, and structured layouts to make it visually appealing and easy to read.
        Make this practical, specific, and actionable with real tactics and numbers.
        """
    
    def _fallback_strategy(self, data):
        """Enhanced fallback campaign strategy with visual elements"""
        return f"""
# 🚀 {data.get('company_name', 'Your Company')} - {data.get('campaign_type', 'Marketing')} Campaign Strategy

## 📊 Executive Summary
| Metric | Value |
|--------|-------|
| **Campaign Type** | {data.get('campaign_type', 'Marketing Campaign')} |
| **Target Market** | {data.get('location', 'Global')} - {data.get('customer_segment', 'Mass Market')} |
| **Budget** | {data.get('budget', 'TBD')} {data.get('currency', 'USD')} |
| **Duration** | {data.get('duration', '8 weeks')} |
| **Channels** | {', '.join(data.get('channels', ['Email Marketing']))} |

## 👥 Target Audience Analysis
🎯 **Primary Audience:** {data.get('target_audience', 'Primary target audience to be defined')}

**📍 Geographic Focus:** {data.get('location', 'Global')}
**💼 Customer Segment:** {data.get('customer_segment', 'Mass Market')}

## 📢 Channel Strategy Dashboard
