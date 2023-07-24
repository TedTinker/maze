import smtplib
import os
import zipfile
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

os.chdir("maze/saved")

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

dir_path = "thesis_pics"
zip_file = "thesis_pics.zip"

zipf = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED)
zipdir(dir_path, zipf)
zipf.close()

smtp_user = 'tedjtinker@gmail.com'
smtp_pass = '1TedTinkerGoogle!'
to_address = 'tedjtinker@gmail.com'
subject = 'subject'
body = 'text'

msg = MIMEMultipart()
msg['From'] = smtp_user
msg['To'] = to_address
msg['Subject'] = subject

msg.attach(MIMEText(body, 'plain'))
binary_zip = open(zip_file, "rb")

part = MIMEBase('application', 'octet-stream')
part.set_payload(binary_zip.read())
encoders.encode_base64(part)

part.add_header('Content-Disposition', 'attachment; filename= ' + os.path.basename(zip_file))
msg.attach(part)

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(smtp_user, smtp_pass)
s.sendmail(smtp_user, to_address, msg.as_string())
s.quit()