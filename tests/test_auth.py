import pytest
from finger_independence.auth import hash_password

def test_hash_password():
    # Test that hashing is deterministic
    pass1 = hash_password("mypassword")
    pass2 = hash_password("mypassword")
    assert pass1 == pass2
    
    # Test that different passwords yield different hashes
    pass3 = hash_password("different")
    assert pass1 != pass3
    
    # Test length of sha256 hex digest
    assert len(pass1) == 64
