"""
auth.py  — JWT auth + Google OAuth helpers
"""
from functools import wraps
from flask import request, jsonify
import jwt, bcrypt, os
from datetime import datetime, timedelta

SECRET_KEY     = os.getenv('SECRET_KEY', 'dev-secret-key')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_token(user_id: str, email: str) -> str:
    payload = {
        'user_id': user_id,
        'email':   email,
        'exp':     datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth  = request.headers.get('Authorization', '')
        if auth.startswith('Bearer '):
            token = auth.split(' ', 1)[1]
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            data         = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
            current_user = {'id': data['user_id'], 'email': data['email']}
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated
