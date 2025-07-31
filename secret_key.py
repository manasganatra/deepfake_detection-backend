from flask import Flask
import os

app = Flask(__name__)
app.secret_key = os.urandom(24) # Generates a random 24-byte key
print(app.secret_key) # For debugging purposes, remove in production