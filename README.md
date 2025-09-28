# Email Classification API

A FastAPI application that classifies emails into different categories (Support, Marketing, Corporate, Spam) with support for both English and Persian languages.

## Features

- **Multilingual Support**: English and Persian language detection
- **Email Categorization**: Classifies emails into Support, Marketing, Corporate, or Spam
- **Hybrid Approach**: Combines rule-based keyword matching with ML sentiment analysis
- **Web Interface**: Beautiful, responsive web interface for easy use
- **API Endpoints**: RESTful API for programmatic access
- **File Upload**: Support for uploading email files (.txt, .eml)
- **Text Input**: Direct text input for email content
- **Spam Detection**: Identifies potential spam emails
- **Auto Device Detection**: Automatically uses GPU or CPU

## Screenshots

### Home Page
![Home Page](screenshots\Home_page.JPG)

### Results Page
![Results Page](screenshots/results.png)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ARZF/email-classification-api.git
   cd email-classification-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8000`

## Usage

### Web Interface

1. Open `http://localhost:8000` in your browser
2. Either:
   - Upload an email file (.txt or .eml format)
   - Paste email text directly into the text area
3. Click "Classify Email" to get results
4. View detailed analysis including category, confidence scores, and spam detection

### API Endpoints

#### Classify Email (Form)
- **POST** `/classify`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Email file (optional)
  - `email_text`: Email text content (optional)

#### Classify Email (API)
- **GET** `/api/classify?email_text=your_email_content`
- **Response**: JSON with classification results

#### API Documentation
- **GET** `/docs` - Interactive API documentation (Swagger UI)
- **GET** `/redoc` - Alternative API documentation

## Classification Categories

### Support
- **English**: help, support, issue, problem, bug, error, fix, resolve, troubleshoot, assistance
- **Persian**: کمک، پشتیبانی، مشکل، خطا، خرابی، تعمیر، حل، خدمات، مشتری، شکایت

### Marketing
- **English**: promotion, offer, discount, sale, deal, special, newsletter, campaign
- **Persian**: تبلیغات، پیشنهاد، تخفیف، فروش، معامله، ویژه، عضویت، خبرنامه

### Corporate
- **English**: meeting, conference, business, partnership, contract, proposal, budget, report
- **Persian**: جلسه، کنفرانس، کسب‌وکار، شراکت، همکاری، قرارداد، بودجه، مالی

### Spam
- **English**: urgent, act now, free money, congratulations, lottery, inheritance
- **Persian**: فوری، الان اقدام کن، پول رایگان، تبریک، برنده شدی، قرعه‌کشی

## Example API Response

```json
{
  "category": {
    "predicted": "support",
    "confidence": 0.75,
    "all_scores": {
      "support": 0.75,
      "marketing": 0.1,
      "corporate": 0.1,
      "spam": 0.05
    }
  },
  "is_spam": false,
  "spam_score": 0.05,
  "text_length": 245,
  "method": "hybrid"
}
```

## Technical Details

- **Framework**: FastAPI
- **ML Models**: 
  - Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`
  - Classification: Rule-based with keyword matching
- **Frontend**: Bootstrap 5 with custom styling
- **Text Processing**: Email header removal, URL/email cleanup
- **Device Support**: Automatic GPU/CPU detection
- **Languages**: English + Persian (easily extensible)

## File Structure

```
email-classification-api/
├── main.py                    # FastAPI application
├── email_classifier.py        # Core classification logic
├── run.py                     # Application runner
├── requirements.txt           # Python dependencies
├── templates/
│   ├── index.html            # Home page
│   └── result.html           # Results page
├── example_persian_email.txt  # Persian test email
├── README.md                 # This file
└── .gitignore               # Git ignore file
```

## Configuration

### Adding New Languages

To add support for additional languages, update the `category_keywords` dictionary in `email_classifier.py`:

```python
self.category_keywords = {
    "support": [
        "help", "support", "issue",  # English
        "کمک", "پشتیبانی", "مشکل",  # Persian
    ],
    # ... other categories
}
```

### Customizing Keywords

Modify the keyword lists in `email_classifier.py` to improve classification accuracy for your specific use case.

## Performance

- **Startup Time**: ~10-15 seconds (first run downloads ML model)
- **Classification Speed**: ~100-500ms per email
- **Memory Usage**: ~500MB-1GB (depending on model)
- **Model Size**: ~250MB (DistilBERT)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) for ML models
- [Bootstrap](https://getbootstrap.com/) for the UI framework

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ for multilingual email classification**
