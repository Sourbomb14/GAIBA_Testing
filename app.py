import streamlit as st
import pandas as pd
import numpy as np
import yagmail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import re
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from email_validator import validate_email, EmailNotValidError
from datetime import datetime, timedelta
import io
import base64
from groq import Groq
import requests
from PIL import Image, ImageDraw, ImageFont
import textwrap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, classification_report, mean_squared_error, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
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

# Enhanced Color Schemes for Visualizations
COLOR_SCHEMES = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'gradient_blue': ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff'],
    'gradient_green': ['#00441b', '#006d2c', '#238b45', '#41ab5d', '#74c476', '#a1d99b', '#c7e9c0', '#e5f5e0', '#f7fcf5'],
    'gradient_red': ['#67000d', '#a50f15', '#cb181d', '#ef3b2c', '#fb6a4a', '#fc9272', '#fcbba1', '#fee0d2', '#fff5f0'],
    'campaign': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
    'analytics': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#560319', '#0B6623', '#FF6B35']
}

# ================================
# SESSION STATE INITIALIZATION
# ================================

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = [
        'current_page', 'campaign_data', 'campaign_strategy', 'email_template_html', 
        'email_template_text', 'email_contacts', 'campaign_results', 'generated_images', 
        'data_analysis_results', 'sender_email', 'sender_password', 'uploaded_datasets',
        'clustering_results', 'feature_importance_results', 'prediction_results'
    ]
    
    defaults = {
        'current_page': "Campaign Dashboard",
        'generated_images': [],
        'sender_email': GMAIL_EMAIL,
        'sender_password': "",
        'uploaded_datasets': [],
        'clustering_results': None,
        'feature_importance_results': None,
        'prediction_results': None
    }
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = defaults.get(var, None)

# ================================
# ENHANCED BULK EMAIL SENDER CLASS
# ================================

class EnhancedBulkEmailSender:
    """Enhanced bulk email sender with SMTP and yagmail support"""
    
    def __init__(self, gmail_address, app_password):
        self.gmail_address = gmail_address
        self.app_password = app_password
        self.smtp_server = None
        self.yag = None
    
    def setup_smtp_connection(self):
        """Setup SMTP connection to Gmail"""
        try:
            self.smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
            self.smtp_server.starttls()
            self.smtp_server.login(self.gmail_address, self.app_password)
            return True
        except Exception as e:
            st.error(f"SMTP connection failed: {e}")
            return False
    
    def setup_yagmail_connection(self):
        """Setup yagmail connection"""
        try:
            self.yag = yagmail.SMTP(user=self.gmail_address, password=self.app_password)
            return True
        except Exception as e:
            st.error(f"Yagmail connection failed: {e}")
            return False
    
    def create_personalized_email(self, template, recipient_data):
        """Create personalized email from template"""
        try:
            personalized_content = template
            for key, value in recipient_data.items():
                placeholder = f"{{{key}}}"
                personalized_content = personalized_content.replace(placeholder, str(value))
            return personalized_content
        except Exception:
            return template
    
    def send_bulk_emails_enhanced(self, df, subject, template, method="yagmail", delay_seconds=1):
        """Send bulk emails with enhanced tracking and progress"""
        results = []
        total_emails = len(df)
        
        # Setup connection based on method
        if method == "yagmail":
            if not self.setup_yagmail_connection():
                return pd.DataFrame()
        else:
            if not self.setup_smtp_connection():
                return pd.DataFrame()
        
        # Progress tracking with simplified UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        sent_count = 0
        failed_count = 0
        invalid_count = 0
        
        for index, row in df.iterrows():
            current_progress = (index + 1) / total_emails
            progress_bar.progress(current_progress)
            
            status_text.text(f"📧 Processing {index + 1}/{total_emails}: {row.get('email', 'Unknown')}")
            
            try:
                recipient_email = row.get('email', '')
                if not recipient_email or not self.validate_email(recipient_email):
                    invalid_count += 1
                    results.append({
                        "email": recipient_email,
                        "name": row.get('name', 'Unknown'),
                        "status": "invalid",
                        "error": "Invalid email format",
                        "timestamp": datetime.now().strftime('%H:%M:%S'),
                        "method": method
                    })
                    continue
                
                # Personalize content
                personalized_content = self.create_personalized_email(template, row.to_dict())
                personalized_subject = self.create_personalized_email(subject, row.to_dict())
                
                # Send email based on method
                if method == "yagmail":
                    self.yag.send(to=recipient_email, subject=personalized_subject, contents=personalized_content)
                else:
                    msg = MIMEMultipart()
                    msg['From'] = self.gmail_address
                    msg['To'] = recipient_email
                    msg['Subject'] = personalized_subject
                    msg.attach(MIMEText(personalized_content, 'html'))
                    text = msg.as_string()
                    self.smtp_server.sendmail(self.gmail_address, recipient_email, text)
                
                sent_count += 1
                results.append({
                    "email": recipient_email,
                    "name": row.get('name', 'Unknown'),
                    "status": "sent",
                    "error": "",
                    "timestamp": datetime.now().strftime('%H:%M:%S'),
                    "method": method
                })
                
            except Exception as e:
                failed_count += 1
                results.append({
                    "email": row.get('email', 'Unknown'),
                    "name": row.get('name', 'Unknown'),
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().strftime('%H:%M:%S'),
                    "method": method
                })
            
            time.sleep(delay_seconds)
        
        # Cleanup connections
        try:
            if method == "smtp" and self.smtp_server:
                self.smtp_server.quit()
        except:
            pass
        
        # Final status
        progress_bar.progress(1.0)
        status_text.success(f"🎉 Campaign completed! ✅ {sent_count} sent, ❌ {failed_count} failed, ⚠️ {invalid_count} invalid")
        
        return pd.DataFrame(results)
    
    def validate_email(self, email):
        """Validate email address"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

# ================================
# ADVANCED ANALYTICS CLASS
# ================================

class AdvancedAnalytics:
    """Advanced analytics with clustering, feature importance, and predictions"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        """Preprocess data for machine learning"""
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            if not df_processed[col].mode().empty:
                df_processed[col].fillna(df_processed[col].mode().iloc[0], inplace=True)
            else:
                df_processed[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col + '_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                try:
                    df_processed[col + '_encoded'] = self.label_encoders[col].transform(df_processed[col].astype(str))
                except:
                    # Handle unseen labels
                    df_processed[col + '_encoded'] = 0
        
        return df_processed
    
    def perform_clustering(self, df, n_clusters=None):
        """Perform K-means clustering analysis"""
        try:
            df_processed = self.preprocess_data(df)
            
            # Select only numeric columns for clustering
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None, "Need at least 2 numeric columns for clustering"
            
            X = df_processed[numeric_cols]
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine optimal number of clusters if not provided
            if n_clusters is None:
                silhouette_scores = []
                K = range(2, min(11, len(df)//2))
                for k in K:
                    if k < len(df):
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
                
                if silhouette_scores:
                    n_clusters = K[np.argmax(silhouette_scores)]
                else:
                    n_clusters = 3
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to original dataframe
            df_clustered = df.copy()
            df_clustered['Cluster'] = cluster_labels
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            
            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            results = {
                'clustered_data': df_clustered,
                'cluster_centers': kmeans.cluster_centers_,
                'silhouette_score': silhouette_avg,
                'pca_data': X_pca,
                'pca_variance_ratio': pca.explained_variance_ratio_,
                'numeric_columns': list(numeric_cols),
                'n_clusters': n_clusters
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Clustering error: {str(e)}"
    
    def calculate_feature_importance(self, df, target_column=None):
        """Calculate feature importance using Random Forest"""
        try:
            df_processed = self.preprocess_data(df)
            
            # Select features (numeric columns)
            feature_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(feature_cols) < 2:
                return None, "Need at least 2 numeric columns for feature importance"
            
            # If no target specified, use the first numeric column
            if target_column is None or target_column not in feature_cols:
                target_column = feature_cols[0]
            
            feature_cols = [col for col in feature_cols if col != target_column]
            
            if len(feature_cols) == 0:
                return None, "Need at least one feature column"
            
            X = df_processed[feature_cols]
            y = df_processed[target_column]
            
            # Determine if regression or classification
            if y.nunique() <= 10:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Make predictions for evaluation
            y_pred = model.predict(X)
            
            results = {
                'importance_data': importance_df,
                'model': model,
                'target_column': target_column,
                'feature_columns': feature_cols,
                'predictions': y_pred,
                'actual_values': y.values,
                'model_type': 'classification' if y.nunique() <= 10 else 'regression'
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Feature importance error: {str(e)}"
    
    def create_predictions(self, df, target_column, forecast_periods=30):
        """Create future predictions"""
        try:
            df_processed = self.preprocess_data(df)
            
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if target_column not in numeric_cols:
                return None, f"Target column '{target_column}' not found in numeric columns"
            
            feature_cols = [col for col in numeric_cols if col != target_column]
            
            if len(feature_cols) == 0:
                return None, "Need at least one feature column"
            
            X = df_processed[feature_cols]
            y = df_processed[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            if y.nunique() <= 10:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Create future predictions (simple trend-based)
            last_values = X.tail(1)
            future_predictions = []
            
            for i in range(forecast_periods):
                pred = model.predict(last_values)[0]
                future_predictions.append(pred)
                # Simple trend continuation
                last_values = last_values * 1.001  # 0.1% growth assumption
            
            results = {
                'model': model,
                'target_column': target_column,
                'train_predictions': y_pred_train,
                'test_predictions': y_pred_test,
                'train_actual': y_train.values,
                'test_actual': y_test.values,
                'future_predictions': future_predictions,
                'forecast_periods': forecast_periods,
                'model_type': 'classification' if y.nunique() <= 10 else 'regression'
            }
            
            return results, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

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

def extract_name_from_email_address(email):
    """Extract potential name from email address"""
    try:
        local_part = email.split('@')[0]
        name_part = re.sub(r'[0-9._-]', ' ', local_part)
        name_parts = [part.capitalize() for part in name_part.split() if len(part) > 1]
        return ' '.join(name_parts) if name_parts else 'Valued Customer'
    except:
        return 'Valued Customer'

def extract_google_sheet_id(url):
    """Extract Google Sheets ID from URL"""
    try:
        if '/spreadsheets/d/' in url:
            return url.split('/spreadsheets/d/')[1].split('/')[0]
        return None
    except:
        return None

def validate_email_format(email):
    """Validate email format"""
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_enhanced_clustering_viz(clustering_results):
    """Create enhanced clustering visualizations"""
    if not clustering_results:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PCA Cluster Visualization', 'Cluster Distribution', 
                       'Silhouette Analysis', 'Cluster Statistics'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "table"}]]
    )
    
    # PCA Cluster Visualization
    pca_data = clustering_results['pca_data']
    clusters = clustering_results['clustered_data']['Cluster']
    colors = COLOR_SCHEMES['campaign'][:clustering_results['n_clusters']]
    
    for i in range(clustering_results['n_clusters']):
        mask = clusters == i
        fig.add_trace(
            go.Scatter(
                x=pca_data[mask, 0], 
                y=pca_data[mask, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(color=colors[i % len(colors)], size=8),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Cluster Distribution
    cluster_counts = clustering_results['clustered_data']['Cluster'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=[f'Cluster {i}' for i in cluster_counts.index],
            y=cluster_counts.values,
            marker_color=colors[:len(cluster_counts)],
            name='Count',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Silhouette Score
    fig.add_trace(
        go.Bar(
            x=['Silhouette Score'],
            y=[clustering_results['silhouette_score']],
            marker_color='lightblue',
            name='Silhouette',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Cluster Statistics Table
    cluster_stats = []
    for i in range(clustering_results['n_clusters']):
        cluster_data = clustering_results['clustered_data'][clustering_results['clustered_data']['Cluster'] == i]
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        avg_values = cluster_data[numeric_cols].mean()
        cluster_stats.append([f'Cluster {i}', len(cluster_data)] + [f"{val:.2f}" for val in avg_values.head(3)])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Cluster', 'Size'] + list(clustering_results['numeric_columns'][:3])),
            cells=dict(values=list(zip(*cluster_stats)) if cluster_stats else [[], [], []])
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Advanced Clustering Analysis",
        height=800,
        template="plotly_dark"
    )
    
    return fig

def create_feature_importance_viz(importance_results):
    """Create enhanced feature importance visualizations"""
    if not importance_results:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Feature Importance Ranking', 'Top 10 Features', 
                       'Model Performance', 'Prediction vs Actual'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "indicator"}, {"type": "scatter"}]]
    )
    
    importance_data = importance_results['importance_data']
    
    # Feature Importance Ranking
    fig.add_trace(
        go.Bar(
            y=importance_data['Feature'],
            x=importance_data['Importance'],
            orientation='h',
            marker_color=COLOR_SCHEMES['analytics'][:len(importance_data)],
            name='Importance',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Top 10 Features
    top_10 = importance_data.head(10)
    fig.add_trace(
        go.Bar(
            x=top_10['Feature'],
            y=top_10['Importance'],
            marker_color=COLOR_SCHEMES['gradient_blue'][:len(top_10)],
            name='Top 10',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Model Performance Indicator
    if importance_results['model_type'] == 'regression':
        mse = mean_squared_error(importance_results['actual_values'], importance_results['predictions'])
        score = max(0, 100 - mse)  # Convert to percentage-like score
        title = "MSE Score"
    else:
        score = accuracy_score(importance_results['actual_values'], importance_results['predictions']) * 100
        title = "Accuracy Score"
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ),
        row=2, col=1
    )
    
    # Prediction vs Actual
    fig.add_trace(
        go.Scatter(
            x=importance_results['actual_values'],
            y=importance_results['predictions'],
            mode='markers',
            marker=dict(color=COLOR_SCHEMES['campaign'][0], size=6),
            name='Predictions',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add diagonal line for perfect predictions
    min_val = min(importance_results['actual_values'].min(), importance_results['predictions'].min())
    max_val = max(importance_results['actual_values'].max(), importance_results['predictions'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Feature Importance Analysis",
        height=800,
        template="plotly_dark"
    )
    
    return fig

def create_prediction_viz(prediction_results):
    """Create enhanced prediction visualizations"""
    if not prediction_results:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Performance', 'Test Performance', 
                       'Future Predictions', 'Model Metrics'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Training Performance
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prediction_results['train_actual']))),
            y=prediction_results['train_actual'],
            mode='lines+markers',
            name='Actual (Train)',
            line=dict(color=COLOR_SCHEMES['campaign'][0]),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prediction_results['train_predictions']))),
            y=prediction_results['train_predictions'],
            mode='lines+markers',
            name='Predicted (Train)',
            line=dict(color=COLOR_SCHEMES['campaign'][1], dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Test Performance
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prediction_results['test_actual']))),
            y=prediction_results['test_actual'],
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color=COLOR_SCHEMES['campaign'][2]),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(prediction_results['test_predictions']))),
            y=prediction_results['test_predictions'],
            mode='lines+markers',
            name='Predicted (Test)',
            line=dict(color=COLOR_SCHEMES['campaign'][3], dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Future Predictions
    future_dates = list(range(len(prediction_results['future_predictions'])))
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=prediction_results['future_predictions'],
            mode='lines+markers',
            name='Future Predictions',
            line=dict(color=COLOR_SCHEMES['campaign'][4]),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Model Metrics
    if prediction_results['model_type'] == 'regression':
        train_mse = mean_squared_error(prediction_results['train_actual'], prediction_results['train_predictions'])
        test_mse = mean_squared_error(prediction_results['test_actual'], prediction_results['test_predictions'])
        metrics = ['Train MSE', 'Test MSE']
        values = [train_mse, test_mse]
    else:
        train_acc = accuracy_score(prediction_results['train_actual'], prediction_results['train_predictions'])
        test_acc = accuracy_score(prediction_results['test_actual'], prediction_results['test_predictions'])
        metrics = ['Train Accuracy', 'Test Accuracy']
        values = [train_acc * 100, test_acc * 100]
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=COLOR_SCHEMES['analytics'][:len(metrics)],
            name='Metrics',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Predictive Analysis Results",
        height=800,
        template="plotly_dark"
    )
    
    return fig

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application function"""
    initialize_session_state()
    
    # Enhanced Custom CSS styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533a71 100%);
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        h1, h2, h3 {
            color: #00d4ff !important;
            font-weight: 600 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #00d4ff, #0099cc, #667eea);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.6);
        }
        
        .success-metric {
            background: linear-gradient(135deg, #28a745 0%, #20c997 50%, #17a2b8 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
        }
        
        .email-config-box {
            background: rgba(255,255,255,0.08);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            margin: 15px 0;
        }
        
        .analytics-card {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            margin: 10px 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-connected { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1 style="font-size: 4rem; margin-bottom: 0; color: white !important;">🚀 Marketing Campaign War Room</h1>
        <p style="font-size: 1.4rem; color: rgba(255,255,255,0.9); margin-top: 0;">AI-Powered Campaign Generation, Advanced Analytics & Email Marketing Platform</p>
        <p style="font-size: 1rem; color: rgba(255,255,255,0.7);">Powered by Groq AI • FLUX.1-dev • Advanced ML Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Navigation Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        if st.button("🎯 Campaign Dashboard", use_container_width=True):
            st.session_state.current_page = "Campaign Dashboard"
            st.rerun()
        
        if st.button("📧 Email Marketing", use_container_width=True):
            st.session_state.current_page = "Email Marketing"
            st.rerun()
        
        if st.button("📊 Advanced Analytics", use_container_width=True):
            st.session_state.current_page = "Advanced Analytics"
            st.rerun()
        
        if st.button("🤖 ML Insights", use_container_width=True):
            st.session_state.current_page = "ML Insights"
            st.rerun()
        
        st.markdown("---")
        
        # Enhanced System status
        st.markdown("### 🔧 System Status")
        
        if GROQ_API_KEY:
            st.markdown('<span class="status-indicator status-connected"></span>🤖 **Groq AI**: Connected', unsafe_allow_html=True)
            st.caption(f"Model: {GROQ_MODEL}")
        else:
            st.markdown('<span class="status-indicator status-error"></span>🤖 **Groq AI**: Configure API key', unsafe_allow_html=True)
        
        if HUGGING_FACE_TOKEN:
            st.markdown('<span class="status-indicator status-connected"></span>🎨 **Image Gen**: Connected', unsafe_allow_html=True)
            st.caption(f"Model: {HF_MODEL}")
        else:
            st.markdown('<span class="status-indicator status-warning"></span>🎨 **Image Gen**: Configure token', unsafe_allow_html=True)
        
        if GMAIL_APP_PASSWORD:
            st.markdown('<span class="status-indicator status-connected"></span>📧 **Email**: Configured', unsafe_allow_html=True)
            st.caption(f"Address: {GMAIL_EMAIL}")
        else:
            st.markdown('<span class="status-indicator status-error"></span>📧 **Email**: Configure password', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced Campaign info
        if st.session_state.campaign_data:
            st.markdown("### 🎯 Active Campaign")
            st.markdown(f'<div class="metric-card"><strong>{st.session_state.campaign_data["company_name"]}</strong><br><small>{st.session_state.campaign_data["campaign_type"]} • {st.session_state.campaign_data["location"]}</small></div>', unsafe_allow_html=True)
        
        if st.session_state.email_contacts is not None:
            st.markdown("### 📊 Contact Stats")
            st.markdown(f'<div class="metric-card">📧 <strong>{len(st.session_state.email_contacts)}</strong> contacts loaded</div>', unsafe_allow_html=True)
        
        # Analytics Status
        if st.session_state.clustering_results or st.session_state.feature_importance_results:
            st.markdown("### 🤖 ML Status")
            if st.session_state.clustering_results:
                st.markdown('<div class="metric-card">🔍 Clustering: ✅ Complete</div>', unsafe_allow_html=True)
            if st.session_state.feature_importance_results:
                st.markdown('<div class="metric-card">📈 Feature Analysis: ✅ Complete</div>', unsafe_allow_html=True)
    
    # Show current page content
    if st.session_state.current_page == "Campaign Dashboard":
        show_campaign_dashboard_page()
    elif st.session_state.current_page == "Email Marketing":
        show_email_marketing_page()
    elif st.session_state.current_page == "Advanced Analytics":
        show_advanced_analytics_page()
    elif st.session_state.current_page == "ML Insights":
        show_ml_insights_page()

# ================================
# PAGE FUNCTIONS
# ================================

def show_campaign_dashboard_page():
    """Enhanced campaign dashboard page"""
    st.header("🎯 AI Campaign Strategy Generator")
    st.markdown('<div class="analytics-card">Create comprehensive marketing campaigns powered by Groq AI with advanced analytics integration</div>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown(st.session_state.campaign_strategy)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📧 Create Email Campaign", use_container_width=True):
                st.session_state.current_page = "Email Marketing"
                st.rerun()
        with col2:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.current_page = "Advanced Analytics"
                st.rerun()
        with col3:
            if st.session_state.campaign_data:
                st.download_button("📄 Download Strategy", 
                    data=st.session_state.campaign_strategy,
                    file_name=f"{st.session_state.campaign_data['company_name']}_campaign_strategy.md",
                    mime="text/markdown",
                    use_container_width=True)

def show_email_marketing_page():
    """Enhanced email marketing page"""
    st.header("📧 Comprehensive Email Marketing Center")
    
    if st.session_state.campaign_data:
        st.success(f"🎯 Active Campaign: **{st.session_state.campaign_data['company_name']}** - {st.session_state.campaign_data['campaign_type']}")
    
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
                                    help="Use {first_name}, {name}, and {email} for personalization")
        
        if edit_choice == "HTML Template":
            st.session_state.email_template_html = edited_content
        else:
            st.session_state.email_template_text = edited_content
        
        if edit_choice == "HTML Template" and st.button("👀 Preview Email Template"):
            preview = edited_content.replace("{first_name}", "John").replace("{name}", "John Smith").replace("{email}", "john@example.com")
            st.components.v1.html(preview, height=600, scrolling=True)
    
    st.markdown("---")
    
    # Email Configuration Section
    st.subheader("📧 Email Configuration & Method Selection")
    st.markdown('<div class="email-config-box">', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        sender_email = st.text_input("📧 Gmail Address", 
                                   value=st.session_state.sender_email)
        email_method = st.selectbox("📮 Email Sending Method", ["YagMail (Recommended)", "SMTP Direct"])
        
    with config_col2:
        sender_password = st.text_input("🔑 Gmail App Password", 
                                      type="password",
                                      value=st.session_state.sender_password)
        delay_seconds = st.slider("⏱️ Delay Between Emails (seconds)", 1, 10, 2)
    
    st.session_state.sender_email = sender_email
    st.session_state.sender_password = sender_password
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Test email configuration
    if sender_email and sender_password:
        if st.button("🔍 Test Email Configuration"):
            try:
                test_sender = EnhancedBulkEmailSender(sender_email, sender_password)
                if email_method == "YagMail (Recommended)":
                    success = test_sender.setup_yagmail_connection()
                else:
                    success = test_sender.setup_smtp_connection()
                
                if success:
                    st.success("✅ Email configuration successful!")
                else:
                    st.error("❌ Email configuration failed!")
            except Exception as e:
                st.error(f"❌ Configuration error: {e}")
    
    st.markdown("---")
    
    # Contact Data Management
    st.subheader("👥 Contact Data Management")
    
    contact_method = st.radio("📥 Choose Contact Input Method:", [
        "📁 Upload File (CSV/Excel)", "📋 Bulk Paste", "🌐 Google Forms/Sheets"
    ])
    
    if contact_method == "📁 Upload File (CSV/Excel)":
        uploaded_file = st.file_uploader("Upload Contact File", type=['csv', 'xlsx'])
        
        if uploaded_file:
            contacts = process_contacts_data_file(uploaded_file)
            if contacts is not None:
                st.session_state.email_contacts = contacts
                st.success(f"✅ Successfully loaded {len(contacts)} valid contacts!")
                with st.expander("👀 Preview Contact Data"):
                    st.dataframe(contacts.head(10))
    
    elif contact_method == "📋 Bulk Paste":
        st.info("💡 Paste email addresses or email,name pairs (one per line)")
        bulk_text = st.text_area("Paste Contact Data:", height=200)
        
        if st.button("🔄 Process Pasted Data") and bulk_text:
            contacts = process_bulk_paste_contacts(bulk_text)
            if contacts is not None:
                st.session_state.email_contacts = contacts
                st.success(f"✅ Successfully processed {len(contacts)} valid contacts!")
    
    elif contact_method == "🌐 Google Forms/Sheets":
        sheet_url = st.text_input("Google Sheets URL:")
        
        if st.button("📊 Load from Google Sheets") and sheet_url:
            try:
                sheet_id = extract_google_sheet_id(sheet_url)
                if sheet_id:
                    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                    df = pd.read_csv(csv_url)
                    contacts = process_contacts_data_file(io.StringIO(df.to_csv(index=False)))
                    if contacts is not None:
                        st.session_state.email_contacts = contacts
                        st.success(f"✅ Successfully loaded {len(contacts)} contacts!")
                else:
                    st.error("❌ Invalid Google Sheets URL")
            except Exception as e:
                st.error(f"❌ Error loading Google Sheets: {e}")
    
    # Enhanced Contact List Management
    if st.session_state.email_contacts is not None:
        st.markdown("---")
        st.subheader("📋 Contact List Management")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Contacts", len(st.session_state.email_contacts))
        with col2:
            domains = st.session_state.email_contacts['email'].str.split('@').str[1].nunique()
            st.metric("🏢 Unique Domains", domains)
        with col3:
            avg_name_length = st.session_state.email_contacts['name'].str.len().mean()
            st.metric("📝 Avg Name Length", f"{avg_name_length:.0f}")
        with col4:
            duplicate_emails = st.session_state.email_contacts['email'].duplicated().sum()
            st.metric("🔄 Duplicates", duplicate_emails)
        
        with st.expander("✏️ Edit Contact List"):
            edited_contacts = st.data_editor(st.session_state.email_contacts, num_rows="dynamic")
            if st.button("💾 Save Contact Changes"):
                st.session_state.email_contacts = edited_contacts
                st.success("Contact list updated!")
    
    # Enhanced Bulk Email Campaign
    if (st.session_state.email_contacts is not None and 
        (st.session_state.email_template_html or st.session_state.email_template_text) and
        sender_email and sender_password):
        
        st.markdown("---")
        st.subheader("🚀 Launch Enhanced Bulk Email Campaign")
        
        df = st.session_state.email_contacts
        
        # Campaign overview
        st.markdown("### 📊 Campaign Overview")
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.markdown(f'<div class="metric-card">👥 Recipients<br><strong>{len(df)}</strong></div>', unsafe_allow_html=True)
        with overview_col2:
            domains = df['email'].str.split('@').str[1].nunique()
            st.markdown(f'<div class="metric-card">🏢 Domains<br><strong>{domains}</strong></div>', unsafe_allow_html=True)
        with overview_col3:
            template_status = "HTML" if st.session_state.email_template_html else "Plain Text"
            st.markdown(f'<div class="metric-card">📧 Template<br><strong>{template_status}</strong></div>', unsafe_allow_html=True)
        with overview_col4:
            estimated_time = (len(df) * delay_seconds) / 60
            st.markdown(f'<div class="metric-card">⏱️ Est. Time<br><strong>{estimated_time:.0f}m</strong></div>', unsafe_allow_html=True)
        
        # Campaign configuration
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            bulk_subject = st.text_input("📧 Campaign Subject Line", value="Important message for {name}")
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
                test_sender = EnhancedBulkEmailSender(sender_email, sender_password)
                test_content = test_sender.create_personalized_email(template_to_use, {"name": "Test User", "first_name": "Test", "email": test_email})
                test_subject = test_sender.create_personalized_email(bulk_subject, {"name": "Test User", "first_name": "Test", "email": test_email})
                
                if email_method == "YagMail (Recommended)":
                    if test_sender.setup_yagmail_connection():
                        test_sender.yag.send(to=test_email, subject=test_subject, contents=test_content)
                        st.success("✅ Test email sent successfully!")
                else:
                    if test_sender.setup_smtp_connection():
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = test_email
                        msg['Subject'] = test_subject
                        msg.attach(MIMEText(test_content, 'html'))
                        test_sender.smtp_server.sendmail(sender_email, test_email, msg.as_string())
                        test_sender.smtp_server.quit()
                        st.success("✅ Test email sent successfully!")
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
        
        # Bulk campaign launch
        if st.button("🚀 LAUNCH ENHANCED BULK EMAIL CAMPAIGN", type="primary", use_container_width=True):
            st.warning(f"⚠️ About to send {len(df)} personalized emails. This action cannot be undone!")
            
            if st.button("✅ YES, SEND ALL EMAILS NOW", key="final_confirm", type="primary"):
                st.info("🚀 Starting enhanced bulk email campaign...")
                
                try:
                    email_sender = EnhancedBulkEmailSender(sender_email, sender_password)
                    method = "yagmail" if email_method == "YagMail (Recommended)" else "smtp"
                    
                    results = email_sender.send_bulk_emails_enhanced(df, bulk_subject, template_to_use, method, delay_seconds)
                    
                    if not results.empty:
                        success_count = len(results[results['status'] == 'sent'])
                        failed_count = len(results[results['status'] == 'failed'])
                        invalid_count = len(results[results['status'] == 'invalid'])
                        success_rate = (success_count / len(results)) * 100
                        
                        st.markdown("### 🎉 Enhanced Campaign Results")
                        
                        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                        
                        with result_col1:
                            st.markdown(f'<div class="success-metric">✅ Successfully Sent<br><h2>{success_count}</h2></div>', unsafe_allow_html=True)
                        with result_col2:
                            st.markdown(f'<div class="metric-card">❌ Failed<br><strong>{failed_count}</strong></div>', unsafe_allow_html=True)
                        with result_col3:
                            st.markdown(f'<div class="metric-card">⚠️ Invalid<br><strong>{invalid_count}</strong></div>', unsafe_allow_html=True)
                        with result_col4:
                            st.markdown(f'<div class="metric-card">📊 Success Rate<br><strong>{success_rate:.1f}%</strong></div>', unsafe_allow_html=True)
                        
                        st.session_state.campaign_results = results
                        
                        # Enhanced results analysis
                        if success_count > 0:
                            st.markdown("### 📈 Campaign Analysis")
                            
                            # Domain analysis
                            sent_results = results[results['status'] == 'sent']
                            domain_analysis = sent_results['email'].str.split('@').str[1].value_counts().head(10)
                            
                            fig = px.bar(
                                x=domain_analysis.index,
                                y=domain_analysis.values,
                                title="Top Email Domains Reached",
                                color_discrete_sequence=['#28a745']
                            )
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv_data = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Campaign Results",
                            data=csv_data,
                            file_name=f"enhanced_campaign_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        with st.expander("📋 View Detailed Campaign Results"):
                            st.dataframe(results, use_container_width=True)
                        
                        if success_count > 0:
                            st.balloons()
                except Exception as e:
                    st.error(f"❌ Campaign error: {str(e)}")
    
    else:
        st.info("📧 **Getting Started:** Configure email settings, generate templates, load contacts, then launch campaigns!")

def show_advanced_analytics_page():
    """Enhanced analytics page with data upload and AI analysis"""
    st.header("📊 Campaign Analytics & Data Intelligence Platform")
    
    # Data Upload and Analysis Section
    st.subheader("📁 Upload Data for AI Analysis")
    
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
                    
                    # Store dataset for ML analysis
                    if 'uploaded_datasets' not in st.session_state:
                        st.session_state.uploaded_datasets = []
                    st.session_state.uploaded_datasets.append((uploaded_file.name, df))
    
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
            st.dataframe(results_df, use_container_width=True)

def show_ml_insights_page():
    """Advanced ML insights page with clustering, feature importance, and predictions"""
    st.header("🤖 Machine Learning Insights & Predictions")
    st.markdown('<div class="analytics-card">Advanced ML analytics with clustering, feature importance, and predictive modeling</div>', unsafe_allow_html=True)
    
    # Check if we have datasets
    if not hasattr(st.session_state, 'uploaded_datasets') or not st.session_state.uploaded_datasets:
        st.info("📊 **Upload data in the Advanced Analytics section first to unlock ML insights!**")
        if st.button("🔗 Go to Advanced Analytics"):
            st.session_state.current_page = "Advanced Analytics"
            st.rerun()
        return
    
    # Dataset selection
    st.subheader("📊 Select Dataset for ML Analysis")
    
    dataset_names = [name for name, df in st.session_state.uploaded_datasets]
    selected_dataset_name = st.selectbox("Choose Dataset:", dataset_names)
    
    selected_df = None
    for name, df in st.session_state.uploaded_datasets:
        if name == selected_dataset_name:
            selected_df = df
            break
    
    if selected_df is None:
        st.error("No dataset selected")
        return
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Rows", selected_df.shape[0])
    with col2:
        st.metric("📊 Columns", selected_df.shape[1])
    with col3:
        numeric_cols = selected_df.select_dtypes(include=[np.number]).columns
        st.metric("🔢 Numeric Cols", len(numeric_cols))
    with col4:
        categorical_cols = selected_df.select_dtypes(include=['object']).columns
        st.metric("📝 Text Cols", len(categorical_cols))
    
    # ML Analysis Options
    st.markdown("---")
    
    analysis_type = st.radio("🔬 Choose ML Analysis Type:", [
        "🔍 Clustering Analysis", "📈 Feature Importance", "🔮 Predictive Modeling"
    ])
    
    analytics = AdvancedAnalytics()
    
    if analysis_type == "🔍 Clustering Analysis":
        st.subheader("🔍 Advanced Clustering Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        with col2:
            if st.button("🚀 Perform Clustering Analysis"):
                with st.spinner("🔍 Performing clustering analysis..."):
                    results, error = analytics.perform_clustering(selected_df, n_clusters)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.session_state.clustering_results = results
                        st.success("✅ Clustering analysis completed!")
        
        # Display clustering results
        if st.session_state.clustering_results:
            st.markdown("### 📊 Clustering Results")
            
            results = st.session_state.clustering_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Clusters Found", results['n_clusters'])
            with col2:
                st.metric("📊 Silhouette Score", f"{results['silhouette_score']:.3f}")
            with col3:
                st.metric("📈 PCA Variance", f"{sum(results['pca_variance_ratio']):.3f}")
            
            # Create and display visualization
            fig = create_enhanced_clustering_viz(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Download clustered data
            clustered_csv = results['clustered_data'].to_csv(index=False)
            st.download_button(
                "📥 Download Clustered Data",
                data=clustered_csv,
                file_name=f"clustered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    elif analysis_type == "📈 Feature Importance":
        st.subheader("📈 Feature Importance Analysis")
        
        numeric_columns = selected_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.error("❌ Need at least 2 numeric columns for feature importance analysis")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox("🎯 Target Column", numeric_columns)
        with col2:
            if st.button("🚀 Calculate Feature Importance"):
                with st.spinner("📈 Calculating feature importance..."):
                    results, error = analytics.calculate_feature_importance(selected_df, target_column)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.session_state.feature_importance_results = results
                        st.success("✅ Feature importance analysis completed!")
        
        # Display feature importance results
        if st.session_state.feature_importance_results:
            st.markdown("### 📊 Feature Importance Results")
            
            results = st.session_state.feature_importance_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Target Column", results['target_column'])
            with col2:
                st.metric("📊 Features Analyzed", len(results['feature_columns']))
            with col3:
                model_type = results['model_type'].title()
                st.metric("🤖 Model Type", model_type)
            
            # Create and display visualization
            fig = create_feature_importance_viz(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show feature importance table
            st.markdown("#### 📋 Feature Importance Rankings")
            st.dataframe(results['importance_data'], use_container_width=True)
            
            # Download results
            importance_csv = results['importance_data'].to_csv(index=False)
            st.download_button(
                "📥 Download Feature Importance",
                data=importance_csv,
                file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    elif analysis_type == "🔮 Predictive Modeling":
        st.subheader("🔮 Predictive Modeling")
        
        numeric_columns = selected_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.error("❌ Need at least 2 numeric columns for predictions")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_column = st.selectbox("🎯 Target Column", numeric_columns)
        with col2:
            forecast_periods = st.slider("📅 Forecast Periods", 10, 100, 30)
        with col3:
            if st.button("🚀 Generate Predictions"):
                with st.spinner("🔮 Generating predictions..."):
                    results, error = analytics.create_predictions(selected_df, target_column, forecast_periods)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        st.session_state.prediction_results = results
                        st.success("✅ Predictions generated successfully!")
        
        # Display prediction results
        if st.session_state.prediction_results:
            st.markdown("### 📊 Prediction Results")
            
            results = st.session_state.prediction_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Target Column", results['target_column'])
            with col2:
                st.metric("🔮 Forecast Periods", results['forecast_periods'])
            with col3:
                model_type = results['model_type'].title()
                st.metric("🤖 Model Type", model_type)
            
            # Create and display visualization
            fig = create_prediction_viz(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction summary
            st.markdown("#### 📋 Prediction Summary")
            
            if results['model_type'] == 'regression':
                train_mse = mean_squared_error(results['train_actual'], results['train_predictions'])
                test_mse = mean_squared_error(results['test_actual'], results['test_predictions'])
                
                summary_data = {
                    'Metric': ['Train MSE', 'Test MSE', 'Avg Future Prediction'],
                    'Value': [f"{train_mse:.4f}", f"{test_mse:.4f}", f"{np.mean(results['future_predictions']):.4f}"]
                }
            else:
                train_acc = accuracy_score(results['train_actual'], results['train_predictions'])
                test_acc = accuracy_score(results['test_actual'], results['test_predictions'])
                
                summary_data = {
                    'Metric': ['Train Accuracy', 'Test Accuracy', 'Most Frequent Prediction'],
                    'Value': [f"{train_acc:.4f}", f"{test_acc:.4f}", f"{max(set(results['future_predictions']), key=results['future_predictions'].count)}"]
                }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download predictions
            predictions_df = pd.DataFrame({
                'Period': range(1, len(results['future_predictions']) + 1),
                'Predicted_Value': results['future_predictions']
            })
            
            predictions_csv = predictions_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions",
                data=predictions_csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    st.markdown("### 🎯 ML Insights Summary")
    
    insights_completed = []
    if st.session_state.clustering_results:
        insights_completed.append("🔍 Clustering Analysis")
    if st.session_state.feature_importance_results:
        insights_completed.append("📈 Feature Importance")
    if st.session_state.prediction_results:
        insights_completed.append("🔮 Predictive Modeling")
    
    if insights_completed:
        st.success(f"✅ Completed: {', '.join(insights_completed)}")
    else:
        st.info("💡 Select an analysis type above to get started with ML insights!")

def extract_name_from_email_address(email):
    """Extract potential name from email address"""
    try:
        local_part = email.split('@')[0]
        name_part = re.sub(r'[0-9._-]', ' ', local_part)
        name_parts = [part.capitalize() for part in name_part.split() if len(part) > 1]
        return ' '.join(name_parts) if name_parts else 'Valued Customer'
    except:
        return 'Valued Customer'

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

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    main()
