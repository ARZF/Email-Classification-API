import re
import torch
from transformers import pipeline
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmailClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float32,
                return_all_scores=True
            )
            self.use_ml = True
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}. Using rule-based fallback.")
            self.use_ml = False
        
        self.category_keywords = {
            "support": [
                "help", "support", "issue", "problem", "bug", "error", "fix", "resolve",
                "troubleshoot", "assistance", "service", "customer", "complaint", "broken",
                "not working", "can't", "unable", "failed", "technical", "maintenance",
                "login", "password", "account", "access", "ticket", "urgent",
                "کمک", "پشتیبانی", "مشکل", "خطا", "خرابی", "تعمیر", "حل", "کمک",
                "خدمات", "مشتری", "شکایت", "خراب", "کار نمی‌کند", "نمی‌توانم",
                "ناموفق", "فنی", "نگهداری", "ورود", "رمز عبور", "حساب", "دسترسی",
                "تیکت", "فوری", "کمک فوری", "مشکل فنی", "خطای سیستم"
            ],
            "marketing": [
                "promotion", "offer", "discount", "sale", "deal", "special", "limited time",
                "subscribe", "newsletter", "marketing", "advertisement", "campaign",
                "product launch", "new product", "upgrade", "premium", "trial", "free",
                "exclusive", "bonus", "save", "win", "contest", "sweepstakes",
                "تبلیغات", "پیشنهاد", "تخفیف", "فروش", "معامله", "ویژه", "زمان محدود", "قیمت",
                "چنده" , "چند",
                "عضویت", "خبرنامه", "بازاریابی", "آگهی", "کمپین", "راه‌اندازی محصول",
                "محصول جدید", "ارتقا", "پریمیوم", "آزمایشی", "رایگان", "انحصاری",
                "پاداش", "صرفه‌جویی", "برنده", "مسابقه", "قرعه‌کشی", "فروش ویژه"
            ],
            "corporate": [
                "meeting", "conference", "business", "partnership", "collaboration",
                "contract", "agreement", "proposal", "budget", "finance", "report",
                "quarterly", "annual", "board", "executive", "management", "strategy",
                "project", "deadline", "milestone", "review", "presentation", "invoice",
                "payment", "legal", "compliance", "policy", "procedure",
                "جلسه", "کنفرانس", "کسب‌وکار", "شراکت", "همکاری", "قرارداد",
                "توافق", "پیشنهاد", "بودجه", "مالی", "گزارش", "فصلی", "سالانه",
                "هیئت مدیره", "اجرایی", "مدیریت", "استراتژی", "پروژه", "مهلت",
                "نقطه عطف", "بررسی", "ارائه", "فاکتور", "پرداخت", "قانونی",
                "انطباق", "سیاست", "روش", "جلسه کاری", "گزارش مالی"
            ],
            "spam": [
                "urgent", "act now", "limited time", "click here", "free money",
                "congratulations", "you've won", "lottery", "inheritance", "viagra",
                "casino", "pills", "weight loss", "make money", "work from home",
                "nigerian prince", "wire transfer", "bank account", "social security",
                "guaranteed", "no risk", "instant", "secret", "exclusive offer",
                "فوری", "الان اقدام کن", "زمان محدود", "اینجا کلیک کن", "پول رایگان",
                "تبریک", "برنده شدی", "قرعه‌کشی", "ارث", "ویاگرا", "کازینو",
                "قرص", "کاهش وزن", "پول درآوردن", "کار از خانه", "شاهزاده نیجریه",
                "انتقال پول", "حساب بانکی", "تأمین اجتماعی", "ضمانت شده",
                "بدون ریسک", "فوری", "مخفی", "پیشنهاد انحصاری", "پول سریع",
                "ثروتمند شو", "درآمد میلیونی", "سریع پولدار شو"
            ]
        }
    
    def preprocess_email(self, email_text: str) -> str:
        email_text = re.sub(r'^From:.*$', '', email_text, flags=re.MULTILINE)
        email_text = re.sub(r'^To:.*$', '', email_text, flags=re.MULTILINE)
        email_text = re.sub(r'^Subject:.*$', '', email_text, flags=re.MULTILINE)
        email_text = re.sub(r'^Date:.*$', '', email_text, flags=re.MULTILINE)
        email_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', email_text)
        email_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', email_text)
        email_text = re.sub(r'\s+', ' ', email_text).strip()
        return email_text
    
    def classify_category_rule_based(self, email_text: str) -> Dict[str, float]:
        email_lower = email_text.lower()
        scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in email_lower:
                    score += 1
            scores[category] = score / len(keywords)
        
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v/total_score for k, v in scores.items()}
        else:
            scores = {"corporate": 1.0, "support": 0.0, "marketing": 0.0, "spam": 0.0}
        
        return scores
    
    def classify_with_ml(self, email_text: str) -> Dict[str, float]:
        if not self.use_ml:
            return {}
        
        try:
            if len(email_text) > 512:
                email_text = email_text[:512]
            
            result = self.classifier(email_text)
            
            sentiment_scores = {}
            for item in result:
                label = item['label'].lower()
                score = item['score']
                
                if 'positive' in label:
                    sentiment_scores['marketing'] = score * 0.3
                    sentiment_scores['corporate'] = score * 0.2
                elif 'negative' in label:
                    sentiment_scores['support'] = score * 0.4
                    sentiment_scores['spam'] = score * 0.1
            
            return sentiment_scores
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return {}

    def classify_email(self, email_text: str) -> Dict[str, Any]:
        try:
            cleaned_text = self.preprocess_email(email_text)
            
            if not cleaned_text.strip():
                raise ValueError("Email content is empty after preprocessing")
            
            rule_scores = self.classify_category_rule_based(cleaned_text)
            ml_scores = self.classify_with_ml(cleaned_text)
            
            if ml_scores:
                combined_scores = {}
                for category in rule_scores.keys():
                    rule_weight = 0.7
                    ml_weight = 0.3
                    
                    rule_score = rule_scores[category]
                    ml_score = ml_scores.get(category, 0.0)
                    
                    combined_scores[category] = (rule_weight * rule_score) + (ml_weight * ml_score)
                
                total = sum(combined_scores.values())
                if total > 0:
                    category_scores = {k: v/total for k, v in combined_scores.items()}
                else:
                    category_scores = rule_scores
            else:
                category_scores = rule_scores
            
            predicted_category = max(category_scores.items(), key=lambda x: x[1])
            is_spam = category_scores["spam"] > 0.3
            
            return {
                "category": {
                    "predicted": predicted_category[0],
                    "confidence": round(predicted_category[1], 3),
                    "all_scores": {k: round(v, 3) for k, v in category_scores.items()}
                },
                "is_spam": is_spam,
                "spam_score": round(category_scores["spam"], 3),
                "text_length": len(cleaned_text),
                "processed_text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
                "method": "hybrid" if ml_scores else "rule-based"
            }
            
        except Exception as e:
            logger.error(f"Error classifying email: {str(e)}")
            raise e
