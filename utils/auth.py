import json
import os
import streamlit as st
from hashlib import sha256

def initialize_users():
    """Initialize user data file if it doesn't exist"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    users_file = os.path.join(data_dir, "users.json")
    
    # Create users.json if it doesn't exist
    if not os.path.exists(users_file):
        with open(users_file, 'w') as f:
            json.dump({}, f)
    
    return users_file

def hash_password(password):
    """Simple password hashing"""
    return sha256(password.encode()).hexdigest()

def load_users():
    """Load all users from the users.json file"""
    users_file = initialize_users()
    try:
        with open(users_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is empty or corrupt, return empty dict
        return {}

def save_users(users):
    """Save users to the users.json file"""
    users_file = initialize_users()
    with open(users_file, 'w') as f:
        json.dump(users, f)

def register_user(username, password):
    """Register a new user"""
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists."
    
    # Add new user
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registration successful!"

def login_user(username, password):
    """Login user with username and password"""
    users = load_users()
    
    # Check if username exists
    if username not in users:
        return False, "Username does not exist."
    
    # Check if password is correct
    if users[username] != hash_password(password):
        return False, "Incorrect password."
    
    return True, "Login successful!"

def logout_user():
    """Logout the current user"""
    for key in ['logged_in', 'username']:
        if key in st.session_state:
            del st.session_state[key]