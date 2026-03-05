"""
Email service utility for sending emails via SMTP
"""
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import logging
import ssl

load_dotenv()

logger = logging.getLogger("email_service")

# Email configuration from environment variables
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"  # Use SSL instead of TLS


def send_password_reset_email(to_email: str, reset_token: str) -> bool:
    """
    Send password reset link email to the user.
    
    Args:
        to_email: Recipient email address
        reset_token: Password reset token to include in the link
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Validate configuration
        if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
            logger.error("Email configuration incomplete. Missing SMTP_HOST, SMTP_USER, or SMTP_PASSWORD")
            return False
        
        # Get frontend URL from environment
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"
        
        # Create message with HTML button
        body = f"""Hello,

You have requested to reset your password for your ZaneAI account.

Click the button below to reset your password:

{reset_link}

Or copy and paste this link into your browser: {reset_link}

This link will expire in 60 minutes.

If you did not request this password reset, please ignore this email.

Best regards,
ZaneAI Team"""
        
        # Create HTML email with button
        html_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        .button {{
            display: inline-block;
            padding: 12px 24px;
            background-color: #007bff;
            color: #ffffff;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .button:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <p>Hello,</p>
    <p>You have requested to reset your password for your ZaneAI account.</p>
    <p>
        <a href="{reset_link}" class="button">Reset Password</a>
    </p>
    <p>Or copy and paste this link into your browser:</p>
    <p><a href="{reset_link}">{reset_link}</a></p>
    <p>This link will expire in 60 minutes.</p>
    <p>If you did not request this password reset, please ignore this email.</p>
    <p>Best regards,<br>ZaneAI Team</p>
</body>
</html>"""
        
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText as HTMLMIMEText
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Password Reset - ZaneAI"
        msg['From'] = SMTP_FROM_EMAIL
        msg['To'] = to_email
        
        # Add both plain text and HTML versions
        part1 = MIMEText(body, 'plain')
        part2 = HTMLMIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        try:
            if SMTP_USE_SSL:
                # Use SSL (typically port 465)
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            else:
                # Use TLS (typically port 587)
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls()  # Enable TLS encryption
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            
            logger.info(f"Password reset email sent successfully to {to_email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed for {to_email}: {str(e)}")
            logger.error("Please check your SMTP_USER and SMTP_PASSWORD")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP connection failed to {SMTP_HOST}:{SMTP_PORT}: {str(e)}")
            logger.error("Please check your SMTP_HOST and SMTP_PORT")
            return False
        
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error while sending email to {to_email}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while sending email to {to_email}: {str(e)}", exc_info=True)
        return False


def send_welcome_email(to_email: str, username: str) -> bool:
    """
    Send welcome email to new user.
    
    Args:
        to_email: Recipient email address
        username: Username of the new user
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Validate configuration
        if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
            logger.error("Email configuration incomplete. Missing SMTP_HOST, SMTP_USER, or SMTP_PASSWORD")
            return False
        
        # Create message
        body = f"""Hello {username},

Welcome to ZaneAI! Your account has been created.

You will receive a separate email with a link to set your password.

If you did not expect this email, please contact your administrator.

Best regards,
ZaneAI Team"""
        
        msg = MIMEText(body)
        msg['Subject'] = "Welcome to ZaneAI"
        msg['From'] = SMTP_FROM_EMAIL
        msg['To'] = to_email
        
        # Send email
        try:
            if SMTP_USE_SSL:
                # Use SSL (typically port 465)
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            else:
                # Use TLS (typically port 587)
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls()  # Enable TLS encryption
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            
            logger.info(f"Welcome email sent successfully to {to_email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed for {to_email}: {str(e)}")
            logger.error("Please check your SMTP_USER and SMTP_PASSWORD")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP connection failed to {SMTP_HOST}:{SMTP_PORT}: {str(e)}")
            logger.error("Please check your SMTP_HOST and SMTP_PORT")
            return False
        
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error while sending welcome email to {to_email}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while sending welcome email to {to_email}: {str(e)}", exc_info=True)
        return False


def send_password_setup_email(to_email: str, username: str, reset_token: str) -> bool:
    """
    Send password setup link email to new user created by admin.
    
    Args:
        to_email: Recipient email address
        username: Username of the new user
        reset_token: Password reset token to include in the link
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Validate configuration
        if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
            logger.error("Email configuration incomplete. Missing SMTP_HOST, SMTP_USER, or SMTP_PASSWORD")
            return False
        
        # Get frontend URL from environment
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        reset_link = f"{frontend_url}/reset-password?token={reset_token}"
        
        # Create message with HTML button
        body = f"""Hello {username},

Your ZaneAI account has been created. To complete your account setup, please set your password by clicking the link below:

{reset_link}

Or copy and paste this link into your browser: {reset_link}

This link will expire in 60 minutes.

If you did not expect this email, please contact your administrator.

Best regards,
ZaneAI Team"""
        
        # Create HTML email with button
        html_body = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        .button {{
            display: inline-block;
            padding: 12px 24px;
            background-color: #007bff;
            color: #ffffff;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .button:hover {{
            background-color: #0056b3;
        }}
    </style>
</head>
<body>
    <p>Hello {username},</p>
    <p>Your ZaneAI account has been created. To complete your account setup, please set your password by clicking the button below:</p>
    <p>
        <a href="{reset_link}" class="button">Set Password</a>
    </p>
    <p>Or copy and paste this link into your browser:</p>
    <p><a href="{reset_link}">{reset_link}</a></p>
    <p>This link will expire in 60 minutes.</p>
    <p>If you did not expect this email, please contact your administrator.</p>
    <p>Best regards,<br>ZaneAI Team</p>
</body>
</html>"""
        
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText as HTMLMIMEText
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Set Your Password - ZaneAI"
        msg['From'] = SMTP_FROM_EMAIL
        msg['To'] = to_email
        
        # Add both plain text and HTML versions
        part1 = MIMEText(body, 'plain')
        part2 = HTMLMIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        try:
            if SMTP_USE_SSL:
                # Use SSL (typically port 465)
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            else:
                # Use TLS (typically port 587)
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    server.starttls()  # Enable TLS encryption
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            
            logger.info(f"Password setup email sent successfully to {to_email}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed for {to_email}: {str(e)}")
            logger.error("Please check your SMTP_USER and SMTP_PASSWORD")
            return False
        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP connection failed to {SMTP_HOST}:{SMTP_PORT}: {str(e)}")
            logger.error("Please check your SMTP_HOST and SMTP_PORT")
            return False
        
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error while sending password setup email to {to_email}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while sending password setup email to {to_email}: {str(e)}", exc_info=True)
        return False

