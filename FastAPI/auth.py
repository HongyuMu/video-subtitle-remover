import secrets
from sqlalchemy.orm import Session
from FastAPI.user_models import User, get_db
from fastapi import HTTPException

# Function to generate a random API key
def generate_api_key():
    return secrets.token_hex(32)  # Generates a secure random 64-character key

# Create a new user and store the API key
def create_user(db: Session, username: str):
    api_key = generate_api_key()  # Generate a unique API key for the user
    user = User(username=username, api_key=api_key)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# Example: Register a new user with a unique API key
def register_user(username: str):
    db = next(get_db())
    try:
        user = create_user(db, username)
        return {"username": user.username, "api_key": user.api_key}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")
    finally:
        db.close()

def validate_api_key(api_key: str) -> bool:
    db = next(get_db())
    user = db.query(User).filter(User.api_key == api_key).first()
    if user:
        return True
    return False
