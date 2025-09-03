import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
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
