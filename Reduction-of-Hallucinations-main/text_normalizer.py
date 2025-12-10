"""
Text Normalization Module
Handles tokenization, casing, and consistent formatting for clinical text.
"""

import re
from typing import List, Dict
import unicodedata


class TextNormalizer:
    """Normalize and canonicalize clinical text entries."""
    
    def __init__(self, lowercase: bool = False, remove_extra_whitespace: bool = True):
        """
        Initialize the text normalizer.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_extra_whitespace: Whether to remove extra whitespace
        """
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        
        # Common medical abbreviations to standardize
        self.medical_abbrev_map = {
            r'\bpt\b': 'patient',
            r'\bhx\b': 'history',
            r'\btx\b': 'treatment',
            r'\bdx\b': 'diagnosis',
            r'\brx\b': 'prescription',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\bc/o\b': 'complains of',
            r'\bs/p\b': 'status post',
        }
        
        # Unit standardization
        self.unit_map = {
            r'(\d+)\s*deg\s*C': r'\1°C',
            r'(\d+)\s*degree[s]?\s*C': r'\1°C',
            r'(\d+)\s*deg\s*F': r'\1°F',
            r'(\d+)\s*degree[s]?\s*F': r'\1°F',
            r'(\d+)\s*mg': r'\1mg',
            r'(\d+)\s*ml': r'\1ml',
            r'(\d+)\s*%': r'\1%',
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to consistent form."""
        # Remove BOM and normalize to NFC form
        text = text.replace('\ufeff', '')
        text = unicodedata.normalize('NFC', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation for consistency."""
        # Ensure space after periods, commas, colons
        text = re.sub(r'\.(?=[A-Za-z])', '. ', text)
        text = re.sub(r',(?=[A-Za-z])', ', ', text)
        text = re.sub(r':(?=[A-Za-z])', ': ', text)
        
        # Standardize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def standardize_medical_terms(self, text: str) -> str:
        """Standardize common medical abbreviations (case-insensitive)."""
        for abbrev, full in self.medical_abbrev_map.items():
            text = re.sub(abbrev, full, text, flags=re.IGNORECASE)
        return text
    
    def standardize_units(self, text: str) -> str:
        """Standardize measurement units."""
        for pattern, replacement in self.unit_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Normalize number formats."""
        # Standardize decimal points
        text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)  # European format to US
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for clinical text.
        
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation, but keep decimal numbers together
        tokens = re.findall(r'\d+\.\d+|\w+|[^\w\s]', text)
        return tokens
    
    def normalize(self, text: str) -> str:
        """
        Apply all normalization steps to text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Normalize whitespace
        if self.remove_extra_whitespace:
            text = self.normalize_whitespace(text)
        
        # Step 3: Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Step 4: Standardize units and numbers
        text = self.standardize_units(text)
        text = self.normalize_numbers(text)
        
        # Step 5: Optionally lowercase (usually not recommended for medical text)
        if self.lowercase:
            text = text.lower()
        
        # Step 6: Final whitespace cleanup
        text = self.normalize_whitespace(text)
        
        return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a batch of texts."""
        return [self.normalize(text) for text in texts]


if __name__ == "__main__":
    # Test the normalizer
    normalizer = TextNormalizer()
    
    test_texts = [
        "Patient   reports fever of 38.5 deg C and cough.",
        "32-year-old female w/ migraine episodes.",
        "Pt   has   multiple   spaces   and  odd   formatting.",
    ]
    
    print("Text Normalizer Test:")
    print("-" * 60)
    for text in test_texts:
        normalized = normalizer.normalize(text)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print()
