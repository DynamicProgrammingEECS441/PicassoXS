"""
flask_app development configuration.
"""

import pathlib

# Root of this application, useful if it doesn't occupy an entire domain
APPLICATION_ROOT = '/'

# Secret key for encrypting cookies
SECRET_KEY = b'\xca\x12\xa8\xcf\x08\xa7\xef"\x8f\xd9\x8e\x18\xf8\xa9\x84\x15\xb9\xd4\x19\x13e]\xda\x9f'
SESSION_COOKIE_NAME = 'login'

# File Upload to var/uploads/
FLASK_APP_ROOT = pathlib.Path(__file__).resolve().parent.parent
# UPLOAD_FOLDER = FLASK_APP_ROOT/'var'/'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Database file is var/insta485.sqlite3
# DATABASE_FILENAME = FLASK_APP_ROOT/'var'/'insta485.sqlite3'