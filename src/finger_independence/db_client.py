import os
import json
from typing import Dict, Any, List, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    """Client for interacting with the Supabase database for telemetry and history."""
    
    def __init__(self):
        url: str = os.environ.get("SUPABASE_URL", "")
        key: str = os.environ.get("SUPABASE_KEY", "")
        
        if not url or not key:
            print("Warning: SUPABASE_URL or SUPABASE_KEY is missing. Database operations will be mocked or fail.")
            self.supabase: Optional[Client] = None
        else:
            self.supabase: Client = create_client(url, key)

    def insert_session(self, user_id: str, final_independence_score: float = None, enslavement_matrix: List[List[float]] = None) -> Optional[str]:
        """Creates a new session record and returns its ID."""
        if not self.supabase:
            import uuid
            return str(uuid.uuid4())
            
        try:
            response = self.supabase.table("sessions").insert({
                "user_id": user_id,
                "final_independence_score": final_independence_score,
                "enslavement_matrix_json": enslavement_matrix
            }).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0].get("id")
            return None
        except Exception as e:
            print(f"Error inserting session: {e}")
            return None

    def update_session(self, session_id: str, final_independence_score: float, enslavement_matrix: List[List[float]]):
        """Updates a session with the final results."""
        if not self.supabase:
            return
            
        try:
            self.supabase.table("sessions").update({
                "final_independence_score": final_independence_score,
                "enslavement_matrix_json": enslavement_matrix
            }).eq("id", session_id).execute()
        except Exception as e:
            print(f"Error updating session {session_id}: {e}")

    def insert_telemetry(self, session_id: str, finger_id: int, frame_data: List[List[float]]):
        """Inserts batched frame data for a specific finger in a session."""
        if not self.supabase:
            return
            
        try:
            # We use jsonb to store the array of frame angles/metrics efficiently
            self.supabase.table("session_telemetry").insert({
                "session_id": session_id,
                "finger_id": finger_id,
                "frame_data_jsonb": frame_data
            }).execute()
        except Exception as e:
            print(f"Error inserting telemetry for finger {finger_id}: {e}")

    def get_recent_sessions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetches the user's recent sessions for historical trend analysis (AI Diagnostics)."""
        if not self.supabase:
            return []
            
        try:
            response = self.supabase.table("sessions") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
                
            if response.data:
                # Return sorted chronologically (oldest to newest) for easier trend analysis
                sessions = response.data
                sessions.reverse() 
                return sessions
            return []
        except Exception as e:
            print(f"Error fetching recent sessions: {e}")
            return []

    def create_user(self, username: str, password_hash: str) -> bool:
        """Creates a new user with the given username and password hash. Returns True on success, False if username exists or error."""
        if not self.supabase:
            return False
            
        try:
            response = self.supabase.table("users").insert({
                "username": username,
                "password_hash": password_hash
            }).execute()
            return True
        except Exception as e:
            print(f"Error creating user (might already exist): {e}")
            return False

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Fetches the user record by username."""
        if not self.supabase:
            return None
            
        try:
            response = self.supabase.table("users").select("*").eq("username", username).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            print(f"Error fetching user: {e}")
            return None

# Singleton instance for easy import
db = SupabaseClient()
