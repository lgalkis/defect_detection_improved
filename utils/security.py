#!/usr/bin/env python3
"""
Security Utilities for Defect Detection System
Provides secure authentication and encryption functionality.
"""

import os
import hashlib
import binascii
import time
import re
from datetime import datetime

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("security_utils")

def verify_password(password):
    """
    Verify a password against the stored hash.
    
    Args:
        password: Password to verify
        
    Returns:
        Boolean indicating if password is valid
    """
    # Get stored hash and salt from config
    stored_hash = config.SECURITY["UI_PASSWORD_HASH"]
    salt = config.SECURITY["PASSWORD_SALT"]
    
    try:
        # Add salt if provided
        password_with_salt = password + salt
        
        # Hash the password
        password_hash = hashlib.sha256(password_with_salt.encode()).hexdigest()
        
        # Compare with stored hash
        is_valid = password_hash == stored_hash
        
        # Log the attempt (without the actual password)
        if is_valid:
            logger.info("Successful authentication attempt")
        else:
            logger.warning("Failed authentication attempt")
        
        return is_valid
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

def hash_password(password, salt=None):
    """
    Generate a secure hash for a password.
    
    Args:
        password: Password to hash
        salt: Optional salt string (generated if not provided)
        
    Returns:
        Tuple of (hash_string, salt)
    """
    try:
        # Generate salt if not provided
        if salt is None:
            salt = binascii.hexlify(os.urandom(16)).decode()
        
        # Add salt to password
        password_with_salt = password + salt
        
        # Hash the password
        password_hash = hashlib.sha256(password_with_salt.encode()).hexdigest()
        
        return password_hash, salt
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        return None, None

def generate_secure_password(length=12):
    """
    Generate a secure random password.
    
    Args:
        length: Length of the password to generate
        
    Returns:
        Randomly generated password string
    """
    import random
    import string
    
    # Characters to use in password
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    numbers = string.digits
    symbols = "!@#$%^&*()-_=+[]{}|;:,.<>?"
    
    # Ensure password has at least one of each character type
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(numbers),
        random.choice(symbols)
    ]
    
    # Fill the rest of the password length with random characters
    all_chars = lowercase + uppercase + numbers + symbols
    password.extend(random.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password characters
    random.shuffle(password)
    
    # Return as string
    return ''.join(password)

def change_password(current_password, new_password):
    """
    Change the application password.
    
    Args:
        current_password: Current password for verification
        new_password: New password to set
        
    Returns:
        Tuple of (success, message)
    """
    # First verify the current password
    if not verify_password(current_password):
        return False, "Current password is incorrect"
    
    # Validate the new password
    if not is_password_strong(new_password):
        return False, "New password does not meet complexity requirements"
    
    try:
        # Hash the new password
        password_hash, salt = hash_password(new_password)
        
        # Update the configuration
        # Note: This would require a proper configuration update mechanism
        # that's outside the scope of this example
        logger.info("Password changed successfully")
        
        # Return the new hash and salt for manual configuration update
        return True, {
            "password_hash": password_hash,
            "salt": salt
        }
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return False, f"Error changing password: {e}"

def is_password_strong(password):
    """
    Check if a password meets complexity requirements.
    
    Args:
        password: Password to check
        
    Returns:
        Boolean indicating if password is strong
    """
    # Password must be at least 8 characters
    if len(password) < 8:
        return False
    
    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        return False
    
    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        return False
    
    # Check for at least one digit
    if not re.search(r'\d', password):
        return False
    
    # Check for at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    
    return True

class LoginRateLimiter:
    """
    Rate limiter for login attempts to prevent brute force attacks.
    """
    
    def __init__(self, max_attempts=5, lockout_period=300):
        """
        Initialize the rate limiter.
        
        Args:
            max_attempts: Maximum number of failed attempts before lockout
            lockout_period: Lockout period in seconds
        """
        self.max_attempts = max_attempts
        self.lockout_period = lockout_period
        self.attempt_count = 0
        self.lockout_until = 0
    
    def is_locked_out(self):
        """
        Check if login is currently locked out.
        
        Returns:
            Tuple of (is_locked, seconds_remaining)
        """
        if self.lockout_until > time.time():
            seconds_remaining = int(self.lockout_until - time.time())
            return True, seconds_remaining
        return False, 0
    
    def record_attempt(self, success):
        """
        Record a login attempt.
        
        Args:
            success: Whether the attempt was successful
            
        Returns:
            Tuple of (is_locked, seconds_remaining)
        """
        # If currently locked out, don't process attempts
        locked, seconds = self.is_locked_out()
        if locked:
            return locked, seconds
        
        if success:
            # Reset counter on successful login
            self.attempt_count = 0
            return False, 0
        else:
            # Increment counter on failed login
            self.attempt_count += 1
            
            # Check if we've reached the maximum attempts
            if self.attempt_count >= self.max_attempts:
                self.lockout_until = time.time() + self.lockout_period
                logger.warning(f"Account locked for {self.lockout_period} seconds due to {self.attempt_count} failed login attempts")
                return True, self.lockout_period
            
            return False, 0

class SecurityAuditLog:
    """
    Security audit logging for sensitive operations.
    """
    
    def __init__(self, log_file=None):
        """
        Initialize the security audit log.
        
        Args:
            log_file: Path to the log file (default: based on config)
        """
        self.log_file = log_file or os.path.join(config.BASE_DIR, "security_audit.log")
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def log_event(self, event_type, details, username=None, success=True):
        """
        Log a security event.
        
        Args:
            event_type: Type of event (login, logout, etc.)
            details: Details about the event
            username: Username associated with the event
            success: Whether the event was successful
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = "SUCCESS" if success else "FAILURE"
            username = username or "unknown"
            
            log_line = f"{timestamp} | {status} | {event_type} | {username} | {details}\n"
            
            with open(self.log_file, "a") as f:
                f.write(log_line)
        except Exception as e:
            logger.error(f"Error writing to security audit log: {e}")

# Initialize security audit log
audit_log = SecurityAuditLog()

# Initialize login rate limiter
rate_limiter = LoginRateLimiter()

# For testing
if __name__ == "__main__":
    print("Testing Security Utilities")
    
    # Test password hashing
    print("\nTesting password hashing:")
    test_password = "P@ssw0rd123"
    password_hash, salt = hash_password(test_password)
    print(f"Password: {test_password}")
    print(f"Hash: {password_hash}")
    print(f"Salt: {salt}")
    
    # Test password verification
    print("\nTesting password verification:")
    is_valid = verify_password(test_password)
    print(f"Is valid password: {is_valid}")
    
    # Test password strength check
    print("\nTesting password strength check:")
    strong_password = "StrongP@ss123"
    weak_password = "password"
    print(f"Is '{strong_password}' strong: {is_password_strong(strong_password)}")
    print(f"Is '{weak_password}' strong: {is_password_strong(weak_password)}")
    
    # Test password generator
    print("\nTesting password generator:")
    generated_password = generate_secure_password()
    print(f"Generated password: {generated_password}")
    print(f"Is generated password strong: {is_password_strong(generated_password)}")
    
    # Test rate limiter
    print("\nTesting login rate limiter:")
    limiter = LoginRateLimiter(max_attempts=3, lockout_period=10)
    
    # Simulate failed login attempts
    for i in range(5):
        locked, time_remaining = limiter.record_attempt(success=False)
        if locked:
            print(f"Locked out after {i+1} attempts. Time remaining: {time_remaining} seconds")
            break
        else:
            print(f"Failed attempt {i+1}. Not locked out yet.")
    
    # Test security audit log
    print("\nTesting security audit log:")
    audit = SecurityAuditLog()
    audit.log_event("LOGIN", "User login via UI", username="admin", success=True)
    audit.log_event("SETTINGS_CHANGE", "Threshold values updated", username="admin", success=True)
    print(f"Events logged to {audit.log_file}")
    
    print("\nSecurity Utilities tests completed.")