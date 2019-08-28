import firebase_admin
from firebase_admin import firestore

from .credentials import CREDENTIALS

credentials = firebase_admin.credentials.Certificate(CREDENTIALS)
app = firebase_admin.initialize_app(credentials, name='Firestore Client')
firestore_client = firestore.client(app=app)
