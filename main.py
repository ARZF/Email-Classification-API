from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Email Classification API", version="1.0.0")

# Create templates directory
templates = Jinja2Templates(directory="templates")

# Initialize the email classifier
from email_classifier import EmailClassifier
classifier = EmailClassifier()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with file upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify")
async def classify_email(
    request: Request,
    file: UploadFile = File(...),
    email_text: str = Form(None)
):
    """Classify email category"""
    try:
        # Get email content
        if file and file.filename:
            # Read from uploaded file
            content = await file.read()
            email_content = content.decode('utf-8')
        elif email_text:
            # Use text input
            email_content = email_text
        else:
            raise HTTPException(status_code=400, detail="No email content provided")
        
        # Classify the email
        result = classifier.classify_email(email_content)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result,
            "email_content": email_content[:500] + "..." if len(email_content) > 500 else email_content
        })
        
    except Exception as e:
        logger.error(f"Error classifying email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/api/classify")
async def classify_email_api(email_text: str):
    """API endpoint for email classification"""
    try:
        result = classifier.classify_email(email_text)
        return result
    except Exception as e:
        logger.error(f"Error classifying email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
