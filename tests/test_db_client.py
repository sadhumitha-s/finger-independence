import pytest
from unittest.mock import MagicMock, patch
from finger_independence.db_client import SupabaseClient

@pytest.fixture
def mock_supabase():
    with patch('finger_independence.db_client.create_client') as mock_create_client:
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        
        # Override environment variables for tests
        with patch.dict('os.environ', {'SUPABASE_URL': 'http://mock.supabase', 'SUPABASE_KEY': 'mock-key'}):
            db = SupabaseClient()
            yield db, mock_client

def create_mock_response(data_list):
    # supabase APIResponse behaves like a pydantic model, iter() yields ("data", data_list) and ("count", None)
    mock_response = MagicMock()
    mock_response.__iter__.return_value = iter([("data", data_list), ("count", None)])
    return mock_response

def test_create_user(mock_supabase):
    db, mock_client = mock_supabase
    
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_table.insert.return_value.execute.return_value = create_mock_response([{"username": "testuser"}])
    
    result = db.create_user("testuser", "hashed_pwd")
    
    assert result is True
    mock_client.table.assert_called_with("users")
    mock_table.insert.assert_called_with({
        "username": "testuser",
        "password_hash": "hashed_pwd"
    })

def test_get_user(mock_supabase):
    db, mock_client = mock_supabase
    
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    
    mock_table.select.return_value.eq.return_value.execute.return_value = create_mock_response([{"username": "testuser", "password_hash": "hash"}])
    
    user = db.get_user("testuser")
    
    assert user is not None
    assert user["username"] == "testuser"

def test_get_user_not_found(mock_supabase):
    db, mock_client = mock_supabase
    
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    
    mock_table.select.return_value.eq.return_value.execute.return_value = create_mock_response([])
    
    user = db.get_user("nonexistent")
    
    assert user is None
