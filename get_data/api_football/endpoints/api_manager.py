import time
import json
from datetime import timedelta
from typing import Dict, Optional, Tuple, List
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the special "unlimited" key
UNLIMITED_API_KEY = "dca41d4edemshe469d9d1754cd7ap1c7e06jsn7c5425d89bef"

class APIManager:
    """Manages multiple API keys and their usage with rotation."""
    
    def __init__(self):
        """Initialize API Manager without loading keys."""
        self.api_keys = []
        self.limited_keys = [] # Store limited keys separately
        self.unlimited_key = None # Store the unlimited key if found
        self._request_counts = {}
        self._last_reset = {}
        self._current_key_index = 0 # Index within self.api_keys (includes unlimited)
        self._state_file = Path('data/.api_manager_state.json')
        
        # Constants
        self.DAILY_LIMIT = 99  # Reduced API daily request limit
        self.RESET_INTERVAL = 24 * 60 * 60  # 24 hours in seconds
        self.RATE_LIMIT_WAIT = 60  # Wait time in seconds when rate limit is hit
        self.SAFETY_THRESHOLD = 5  # Number of requests to keep as safety buffer for limited keys
        self.MIN_KEY_INTERVAL = 1.5  # Minimum seconds between uses of same key
        
        # Track consecutive failures and usage per key
        self._consecutive_failures = {}
        self._last_use_time = {}
        self._load_state()
        
        logger.info("Created API Manager instance")
        
    def _load_state(self):
        """Load saved state from file if it exists."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    state = json.load(f)
                self._request_counts = state.get('request_counts', {})
                self._last_reset = state.get('last_reset', {})
                self._last_use_time = state.get('last_use_time', {})
                self._consecutive_failures = state.get('consecutive_failures', {})
                logger.info("Loaded existing API manager state")
        except Exception as e:
            logger.warning(f"Could not load API manager state: {e}")
            
    def _save_state(self):
        """Save current state to file."""
        try:
            state = {
                'request_counts': self._request_counts,
                'last_reset': self._last_reset,
                'last_use_time': self._last_use_time,
                'consecutive_failures': self._consecutive_failures
            }
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"Could not save API manager state: {e}")

    def initialize(self, api_keys: Optional[List[str]] = None):
        """Initialize or reinitialize the API manager with keys."""
        all_keys = []
        if api_keys:
            all_keys = api_keys
        else:
            # Try to get API keys from environment variables
            env_keys = [
                os.getenv("RAPID_API_KEY_1"),
                os.getenv("RAPID_API_KEY_2"),
                os.getenv("RAPID_API_KEY_3"),
                os.getenv("RAPID_API_KEY_4")
            ]
            
            # Filter out None values
            env_keys_filtered = [key for key in env_keys if key]

            # Use hardcoded backup keys if no env keys found
            if not env_keys_filtered:
                 all_keys = [
                     os.getenv("RAPID_API_KEY_1", "312836304amsh55d4ccaf5ca371dp13a3f6jsn5d6b72f5d91a"),  # Primary
                     os.getenv("RAPID_API_KEY_2", "efd8a9c220msh948a00c77b1dfa9p189680jsn5a01311680c0"),  # Secondary
                     os.getenv("RAPID_API_KEY_3", "59eafe2452msh5cac1e68bf1bd35p105bb6jsn45a1a524807c"),  # Third
                     os.getenv("RAPID_API_KEY_4", UNLIMITED_API_KEY),  # Fourth
                     # Add the unlimited key here if using hardcoded defaults
                     # UNLIMITED_API_KEY
                 ]
            else:
                 all_keys = env_keys_filtered

            # Explicitly add the unlimited key if it's not already present from env vars
            if UNLIMITED_API_KEY not in all_keys:
                 all_keys.append(UNLIMITED_API_KEY)

        # Separate keys
        self.api_keys = []
        self.limited_keys = []
        self.unlimited_key = None

        for key in all_keys:
             if key == UNLIMITED_API_KEY:
                 self.unlimited_key = key
                 self.api_keys.append(key) # Keep it in the main list for indexing
                 logger.info(f"Identified unlimited key: ...{key[-4:]}")
             elif key: # Ensure key is not empty/None
                 self.limited_keys.append(key)
                 self.api_keys.append(key)

        # Ensure unlimited key is last in the general api_keys list for rotation simplicity if needed
        if self.unlimited_key and self.api_keys[-1] != self.unlimited_key:
             self.api_keys.remove(self.unlimited_key)
             self.api_keys.append(self.unlimited_key)

        # Initialize tracking dictionaries for all keys found
        for key in self.api_keys:
            if key not in self._request_counts:
                self._request_counts[key] = 0
                self._last_reset[key] = time.time()
                self._last_use_time[key] = 0
                self._consecutive_failures[key] = 0
        
        self._current_key_index = 0
        self._save_state()
        
        logger.info(f"Initialized API Manager with {len(self.limited_keys)} limited keys and {1 if self.unlimited_key else 0} unlimited key.")

    def _should_reset_counter(self, api_key: str) -> bool:
        """Check if the counter should be reset based on time elapsed."""
        current_time = time.time()
        elapsed = current_time - self._last_reset[api_key]
        return elapsed >= self.RESET_INTERVAL
        
    def _reset_counter_if_needed(self, api_key: str):
        """Reset the counter if enough time has passed."""
        if self._should_reset_counter(api_key):
            self._request_counts[api_key] = 0
            self._last_reset[api_key] = time.time()
            self._consecutive_failures[api_key] = 0
            self._save_state()
            logger.info(f"Reset counter for API key ending in ...{api_key[-4:]}")

    def _find_best_available_key(self) -> Optional[str]:
        """Find the best available API key, prioritizing limited keys."""
        current_time = time.time()
        best_limited_key = None
        best_limited_score = float('-inf')

        # 1. Try to find the best LIMITED key
        for key in self.limited_keys:
            self._reset_counter_if_needed(key)

            # Skip if over limit (only for limited keys)
            if self._request_counts[key] >= (self.DAILY_LIMIT - self.SAFETY_THRESHOLD):
                continue

            time_since_last_use = current_time - self._last_use_time.get(key, 0)
            if time_since_last_use < self.MIN_KEY_INTERVAL:
                continue

            remaining_requests = self.DAILY_LIMIT - self._request_counts[key]
            failure_penalty = self._consecutive_failures.get(key, 0) * 10
            score = (remaining_requests * 100) + (time_since_last_use * 10) - failure_penalty

            if score > best_limited_score:
                best_limited_score = score
                best_limited_key = key

        if best_limited_key:
            logger.debug(f"Found best limited key: ...{best_limited_key[-4:]}")
            return best_limited_key

        # 2. If no limited key is suitable, check the UNLIMITED key
        if self.unlimited_key:
             key = self.unlimited_key
             self._reset_counter_if_needed(key) # Reset needed for failure count reset primarily
             time_since_last_use = current_time - self._last_use_time.get(key, 0)

             # Only check cooldown for unlimited key
             if time_since_last_use >= self.MIN_KEY_INTERVAL:
                 logger.debug(f"Falling back to unlimited key: ...{key[-4:]}")
                 return key
             else:
                 logger.debug(f"Unlimited key ...{key[-4:]} is on cooldown.")
                 # Fall through to return None, triggering wait logic in _rotate_to_next_key

        # 3. If neither limited nor unlimited keys are ready
        logger.debug("No suitable key found immediately (limited exhausted, unlimited on cooldown or not present).")
        return None

    def _rotate_to_next_key(self) -> str:
        """Rotate to the next available API key, prioritizing limited then unlimited."""
        if not self.api_keys:
            raise ValueError("No API keys available. Call initialize() first.")

        # Try to find the best available key (limited first, then unlimited)
        best_key = self._find_best_available_key()

        if best_key:
            self._current_key_index = self.api_keys.index(best_key)
            usage_info = f"(Used: {self._request_counts[best_key]})"
            if best_key in self.limited_keys:
                 usage_info = f"(Used: {self._request_counts[best_key]}/{self.DAILY_LIMIT})"
            logger.info(
                f"Rotated to API key ...{best_key[-4:]} {usage_info}"
            )
            return best_key

        # If no key is immediately available (all limited exhausted, unlimited on cooldown)
        logger.warning("All limited API keys exhausted or need cooldown, and unlimited key is on cooldown!")

        # Find the key that will be available soonest (consider limited key resets and cooldowns for all keys)
        current_time = time.time()
        best_wait_time = float('inf')
        # best_wait_key = None # Keep track if needed for logging

        # Check limited keys for reset time
        for key in self.limited_keys:
            if self._request_counts[key] >= (self.DAILY_LIMIT - self.SAFETY_THRESHOLD):
                time_until_reset = (
                    self._last_reset[key] + self.RESET_INTERVAL - current_time
                )
                if time_until_reset < best_wait_time:
                    best_wait_time = time_until_reset
                    # best_wait_key = key

        # Check all keys (including unlimited) for cooldown
        for key in self.api_keys:
             time_since_last_use = current_time - self._last_use_time.get(key, 0)
             if time_since_last_use < self.MIN_KEY_INTERVAL:
                 cooldown_wait = self.MIN_KEY_INTERVAL - time_since_last_use
                 if cooldown_wait < best_wait_time:
                     best_wait_time = cooldown_wait
                     # best_wait_key = key

        # Wait for the shortest time plus a small buffer
        wait_time = max(best_wait_time + 0.1, 1.0) # Ensure minimum wait of 1s
        logger.info(f"Waiting {wait_time:.1f} seconds for next available key (cooldown or reset)")
        time.sleep(wait_time)

        # Try again after waiting
        return self._rotate_to_next_key()

    def get_active_api_key(self) -> Tuple[str, Dict[str, str]]:
        """Get the current active API key and headers with improved management."""
        if not self.api_keys:
            self.initialize()
            if not self.api_keys:
                raise ValueError("No API keys available. Check environment variables or hardcoded keys.")

        current_key = self.api_keys[self._current_key_index]
        current_time = time.time()

        self._reset_counter_if_needed(current_key)

        # Check if rotation is needed
        needs_rotation = False
        # Condition 1: Key is limited AND near its usage limit
        if current_key in self.limited_keys and self._request_counts[current_key] >= (self.DAILY_LIMIT - self.SAFETY_THRESHOLD):
             logger.debug(f"Limited key ...{current_key[-4:]} is near limit.")
             needs_rotation = True
        # Condition 2: Key (any key) was used too recently (cooldown)
        if current_time - self._last_use_time.get(current_key, 0) < self.MIN_KEY_INTERVAL:
             logger.debug(f"Key ...{current_key[-4:]} is on cooldown.")
             needs_rotation = True

        if needs_rotation:
            current_key = self._rotate_to_next_key() # This finds the best next key (limited first)

        # --- Update tracking for the key we are *actually* using ---
        self._request_counts[current_key] += 1
        self._last_use_time[current_key] = current_time
        # Reset consecutive failures on successful use
        self._consecutive_failures[current_key] = 0
        self._save_state()

        # Log usage with more detail
        if current_key == self.unlimited_key:
             logger.info(
                 f"Using Unlimited API key ...{current_key[-4:]} "
                 f"(Used: {self._request_counts[current_key]})"
             )
        else: # It's a limited key
             remaining = self.DAILY_LIMIT - self._request_counts[current_key]
             logger.info(
                 f"Using Limited API key ...{current_key[-4:]} "
                 f"(Used: {self._request_counts[current_key]}/{self.DAILY_LIMIT}, "
                 f"Remaining: {remaining})"
             )

        # Create headers
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": current_key
        }

        return current_key, headers

    def handle_rate_limit(self, api_key: str):
        """Handle rate limit with improved failure tracking."""
        if not self.api_keys:
            raise ValueError("No API keys available. Call initialize() first.")
            
        # Increment consecutive failures
        self._consecutive_failures[api_key] = self._consecutive_failures.get(api_key, 0) + 1
        self._save_state()
        
        logger.warning(
            f"Rate limit hit for key ...{api_key[-4:]} "
            f"(Failure #{self._consecutive_failures[api_key]})"
        )
        
        # Calculate wait time based on failure count
        if self._consecutive_failures[api_key] >= 3:
            wait_time = self.RATE_LIMIT_WAIT * (2 ** (self._consecutive_failures[api_key] - 2))
            logger.warning(f"Multiple failures detected, increasing wait time to {wait_time}s")
        else:
            wait_time = self.RATE_LIMIT_WAIT
            
        # Force rotation to a different key
        new_key = self._rotate_to_next_key()
        if new_key == api_key:  # If we're back to the same key
            logger.info(f"Waiting {wait_time} seconds due to rate limit")
            time.sleep(wait_time)

    def get_request_counts(self) -> Dict[str, int]:
        """Get current request counts for all APIs."""
        if not self.api_keys:
            raise ValueError("No API keys available. Call initialize() first.")
        return self._request_counts.copy()
        
    def get_time_until_reset(self, api_key: str) -> timedelta:
        """Get time remaining until the counter resets for an API key."""
        if not self.api_keys:
            raise ValueError("No API keys available. Call initialize() first.")
            
        current_time = time.time()
        elapsed = current_time - self._last_reset[api_key]
        remaining = self.RESET_INTERVAL - elapsed
        return timedelta(seconds=max(0, remaining))
    
    def get_usage_stats(self) -> Dict[str, Dict]:
        """Get detailed usage statistics for all API keys."""
        if not self.api_keys:
            # Try initializing, but return empty if still no keys
            self.initialize()
            if not self.api_keys:
                 logger.warning("Cannot get usage stats: No API keys available.")
                 return {}


        stats = {}
        current_time = time.time()

        for key in self.api_keys:
            self._reset_counter_if_needed(key)  # Ensure counts are current

            time_since_last_use = current_time - self._last_use_time.get(key, 0)
            cooldown_remaining = max(0, self.MIN_KEY_INTERVAL - time_since_last_use)

            key_label = f"...{key[-4:]}"
            key_stats = {
                "requests_made": self._request_counts.get(key, 0),
                "time_until_reset": str(self.get_time_until_reset(key)),
                "cooldown_remaining": f"{cooldown_remaining:.1f}s",
                "consecutive_failures": self._consecutive_failures.get(key, 0),
                "is_current": self.api_keys[self._current_key_index] == key,
                "type": "Unlimited" if key == self.unlimited_key else "Limited"
            }

            if key in self.limited_keys:
                key_stats["requests_remaining"] = self.DAILY_LIMIT - self._request_counts.get(key, 0)
            else:
                key_stats["requests_remaining"] = "N/A (Unlimited)"


            stats[key_label] = key_stats
        return stats

    def initialize_scraper(self, scraper) -> None:
        """Initialize a scraper with API configuration."""
        try:
            # Check if scraper has the required methods
            if hasattr(scraper, 'set_api_key'):
                current_key, headers = self.get_active_api_key()
                scraper.set_api_key(current_key)
                logger.info("Initialized scraper with API key")
                return
                
            # Alternative: Try setting headers directly
            if hasattr(scraper, 'set_headers'):
                _, headers = self.get_active_api_key()
                scraper.set_headers(headers)
                logger.info("Initialized scraper with API headers")
                return
                
            # Last resort: Try setting attributes directly
            current_key, headers = self.get_active_api_key()
            if not hasattr(scraper, 'api_key'):
                scraper.api_key = current_key
            if not hasattr(scraper, 'headers'):
                scraper.headers = headers
            logger.info("Set API attributes on scraper directly")
            
        except Exception as e:
            logger.error(f"Failed to initialize scraper: {str(e)}")
            raise ValueError(f"Could not initialize scraper: {str(e)}")
            
    def set_api_manager(self, scraper) -> None:
        """Set this API manager instance on a scraper."""
        try:
            if hasattr(scraper, 'set_api_manager'):
                scraper.set_api_manager(self)
                logger.info("Set API manager on scraper")
            else:
                scraper.api_manager = self
                logger.info("Set API manager attribute directly")
        except Exception as e:
            logger.error(f"Failed to set API manager: {str(e)}")
            raise ValueError(f"Could not set API manager: {str(e)}")

    def ensure_initialized(self):
        """Ensure the API manager is initialized with keys."""
        if not self.api_keys:
            self.initialize()
            
# Create a singleton instance
api_manager = APIManager() 