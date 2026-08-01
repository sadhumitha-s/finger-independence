import streamlit as st
import hashlib
from finger_independence.db_client import db

def hash_password(password: str) -> str:
    """Hashes a password using SHA-256 for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password() -> bool:
    """
    Handles user authentication via Streamlit session state and Supabase UI.
    Returns True if the user is authenticated, otherwise False (rendering the login UI).
    """
    if "user_logged_in" not in st.session_state:
        st.session_state["user_logged_in"] = False
        st.session_state["current_user"] = None

    if st.session_state["user_logged_in"]:
        with st.sidebar:
            st.write(f"Logged in as: **{st.session_state['current_user']}**")
            if st.button("Logout"):
                st.session_state["user_logged_in"] = False
                st.session_state["current_user"] = None
                st.rerun()
        return True

    st.title("Welcome to Finger Independence Tracker")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user_record = db.get_user(login_user)
            if user_record and user_record["password_hash"] == hash_password(login_pass):
                st.session_state["user_logged_in"] = True
                st.session_state["current_user"] = login_user
                st.rerun()
            else:
                st.error("Invalid username or password")
                
    with tab2:
        st.subheader("Sign Up")
        signup_user = st.text_input("Choose Username", key="signup_user")
        signup_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        signup_pass2 = st.text_input("Confirm Password", type="password", key="signup_pass2")
        if st.button("Sign Up"):
            if signup_pass != signup_pass2:
                st.error("Passwords do not match!")
            elif len(signup_user) < 3:
                st.error("Username must be at least 3 characters")
            elif len(signup_pass) < 5:
                st.error("Password must be at least 5 characters")
            else:
                if db.create_user(signup_user, hash_password(signup_pass)):
                    st.success("Account created successfully! You can now log in.")
                else:
                    st.error("Username already exists or database error.")

    return False
