#auth.py
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
import uuid

from sqlalchemy.orm import Session
from database_v2 import get_db, User

SECRET_KEY = os.environ.get("JWT_SECRET", "super-secret-key-change-me-in-production-123456789")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

class LocalUser(BaseModel):
    id: str  # UUID as string
    username: str
    email: Optional[str] = None

import bcrypt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not bool(plain_password) or not bool(hashed_password):
        return False
    # If the hash doesn't look like bcrypt, try direct string comparison for old test users
    if not hashed_password.startswith("$2b$") and not hashed_password.startswith("$2a$"):
        return plain_password == hashed_password
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> LocalUser:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise credentials_exception from exc

    username = payload.get("sub")
    uid = payload.get("uid")
    if not username or not uid:
        raise credentials_exception

    return LocalUser(
        id=str(uid),
        username=username,
        email=payload.get("email"),
    )

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> LocalUser:
    user_data = decode_access_token(token)
    user = db.query(User).filter(User.id == user_data.id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return LocalUser(id=str(user.id), username=user.username, email=user.email)
