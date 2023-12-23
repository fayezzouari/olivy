import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(subject, message, to_email):
    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = 'info.Olivy@gmail.com'
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the message to the MIME
    msg.attach(MIMEText(message, 'plain'))

    # Connect to the Yahoo SMTP server
    with smtplib.SMTP('smtp.mail.yahoo.com', 587) as server:
        # Start the TLS session
        server.starttls()

        # Log in to the Yahoo email account
        server.login('info.Olivy@gmail.com', 'j/UrEZK0AfKmvU6fY9xpYDr9fC5wlK5dBLJQ/9+1dvc=')

        # Send the email
        server.sendmail('your_yahoo_email@yahoo.com', to_email, msg.as_string())

# Example usage
send_email('Test Subject', 'This is a test message.', 'ellabouhawel@gmail.com')
print('DOne')