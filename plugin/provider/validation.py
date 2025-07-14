from sqlalchemy.orm import Session
from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from backend.FastAPI.models import User, get_db

class SubRemoverProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, str]) -> None:
        api_key = credentials.get("api_key")
        if not api_key:
            raise ToolProviderCredentialValidationError("API key is missing.")
        
        try:
            db = next(get_db())  # Get a database session
            user = db.query(User).filter(User.api_key == api_key).first()
            
            if not user:
                raise ToolProviderCredentialValidationError("Invalid API key.")
            
            print(f"API key for {user.username} validated.")

        except Exception as e:
            raise ToolProviderCredentialValidationError(f"Error validating API key: {str(e)}")
        
        finally:
            db.close()
