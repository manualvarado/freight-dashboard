import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'tms_database'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Email configuration for future automation
EMAIL_CONFIG = {
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': os.getenv('SMTP_PORT', '587'),
    'email_user': os.getenv('EMAIL_USER', ''),
    'email_password': os.getenv('EMAIL_PASSWORD', ''),
    'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
}

# Report configuration
REPORT_CONFIG = {
    'company_name': os.getenv('COMPANY_NAME', 'Freight Company'),
    'report_title': 'Weekly Operations Report',
    'output_format': ['pdf', 'excel'],  # Future: support multiple formats
    'schedule_time': '23:00',  # 11 PM EST
    'schedule_day': 'sunday',
} 