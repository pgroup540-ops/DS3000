"""
Adversarial Data Augmentation Module
Generates hard negatives through entity/date/medication swaps and fabrications.
"""

import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    """Represents an extracted entity from text."""
    text: str
    type: str
    start: int
    end: int


class AdversarialAugmenter:
    """Generate adversarial examples and hard negatives for training."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Medical entity vocabularies for swapping
        self.medications = [
            "aspirin", "ibuprofen", "acetaminophen", "metformin", "lisinopril",
            "atorvastatin", "amlodipine", "omeprazole", "levothyroxine", "albuterol",
            "fluoxetine", "sertraline", "gabapentin", "prednisone", "warfarin",
            "insulin", "metoprolol", "losartan", "simvastatin", "clopidogrel"
        ]
        
        self.conditions = [
            "hypertension", "diabetes", "asthma", "COPD", "depression", "anxiety",
            "arthritis", "migraine", "pneumonia", "bronchitis", "influenza",
            "COVID-19", "neuropathy", "infection", "fracture", "sprain"
        ]
        
        self.procedures = [
            "surgery", "CT scan", "MRI", "X-ray", "ultrasound", "ECG", "EEG",
            "biopsy", "endoscopy", "colonoscopy", "blood test", "urinalysis"
        ]
        
        self.test_results = [
            "normal", "abnormal", "negative", "positive", "elevated", "decreased",
            "stable", "improving", "worsening", "borderline"
        ]
        
        self.symptoms = [
            "pain", "fever", "cough", "shortness of breath", "nausea", "vomiting",
            "headache", "dizziness", "fatigue", "weakness", "chest pain"
        ]
        
        self.negation_phrases = [
            "no", "denies", "without", "negative for", "absent", "not present"
        ]
        
        self.affirmation_phrases = [
            "has", "reports", "with", "positive for", "present", "experiencing"
        ]
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract medical entities from text.
        
        Args:
            text: Input clinical text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract medications with dosages
        med_pattern = r'\b(\d+\s*mg|ml)\s+(\w+)\b|\b(\w+)\s+(\d+\s*mg|ml)\b'
        for match in re.finditer(med_pattern, text, re.IGNORECASE):
            entities.append(Entity(
                text=match.group(0),
                type="medication",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract standalone medications
        for med in self.medications:
            pattern = r'\b' + re.escape(med) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(0),
                    type="medication",
                    start=match.start(),
                    end=match.end()
                ))
        
        # Extract conditions/diagnoses
        for condition in self.conditions:
            pattern = r'\b' + re.escape(condition) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(0),
                    type="condition",
                    start=match.start(),
                    end=match.end()
                ))
        
        # Extract test results
        for result in self.test_results:
            pattern = r'\b' + re.escape(result) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(0),
                    type="test_result",
                    start=match.start(),
                    end=match.end()
                ))
        
        # Extract numeric values (vitals, lab results)
        vital_pattern = r'\b(\d+\.?\d*)\s*(?:°C|°F|%|mg/dL|mmHg|/\d+)\b'
        for match in re.finditer(vital_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                type="measurement",
                start=match.start(),
                end=match.end()
            ))
        
        # Sort by position and remove overlaps
        entities.sort(key=lambda e: e.start)
        non_overlapping = []
        last_end = -1
        for entity in entities:
            if entity.start >= last_end:
                non_overlapping.append(entity)
                last_end = entity.end
        
        return non_overlapping
    
    def swap_entities(self, text: str, entity_type: str = None) -> Tuple[str, str]:
        """
        Swap entities in text to create adversarial examples.
        
        Args:
            text: Original text
            entity_type: Specific entity type to swap (None for random)
            
        Returns:
            Tuple of (modified_text, modification_type)
        """
        entities = self.extract_entities(text)
        
        if not entities:
            return text, "no_swap"
        
        # Filter by entity type if specified
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]
            if not entities:
                return text, "no_swap"
        
        # Select random entity to swap
        entity = random.choice(entities)
        
        # Generate replacement based on entity type
        replacement = None
        if entity.type == "medication":
            replacement = random.choice(self.medications)
        elif entity.type == "condition":
            replacement = random.choice(self.conditions)
        elif entity.type == "test_result":
            replacement = random.choice(self.test_results)
        elif entity.type == "measurement":
            # Modify numeric value slightly
            match = re.search(r'(\d+\.?\d*)', entity.text)
            if match:
                value = float(match.group(1))
                new_value = value * random.uniform(0.7, 1.3)
                replacement = entity.text.replace(match.group(1), f"{new_value:.1f}")
        
        if replacement:
            modified_text = text[:entity.start] + replacement + text[entity.end:]
            return modified_text, f"{entity.type}_swap"
        
        return text, "no_swap"
    
    def invert_negation(self, text: str) -> Tuple[str, bool]:
        """
        Invert negation/affirmation to create contradictions.
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (modified_text, was_inverted)
        """
        modified = text
        inverted = False
        
        # Try to invert negations
        for negation in self.negation_phrases:
            pattern = r'\b' + re.escape(negation) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                replacement = random.choice(self.affirmation_phrases)
                modified = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                inverted = True
                break
        
        # If no negation found, try to invert affirmation
        if not inverted:
            for affirmation in self.affirmation_phrases:
                pattern = r'\b' + re.escape(affirmation) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    replacement = random.choice(self.negation_phrases)
                    modified = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                    inverted = True
                    break
        
        return modified, inverted
    
    def fabricate_information(self, text: str) -> Tuple[str, str]:
        """
        Add fabricated medical information to create hallucinations.
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (modified_text, fabrication_type)
        """
        fabrication_templates = [
            ("medication", lambda: f" Patient is on {random.choice(self.medications)}."),
            ("condition", lambda: f" Patient has history of {random.choice(self.conditions)}."),
            ("symptom", lambda: f" Patient reports {random.choice(self.symptoms)}."),
            ("test", lambda: f" {random.choice(self.procedures)} shows {random.choice(self.test_results)} findings."),
        ]
        
        fab_type, template = random.choice(fabrication_templates)
        fabrication = template()
        
        # Add fabrication to the text
        modified_text = text + fabrication
        
        return modified_text, f"fabrication_{fab_type}"
    
    def generate_adversarial_negative(
        self,
        clinical_note: str,
        original_summary: str,
        strategy: str = "random"
    ) -> Dict[str, str]:
        """
        Generate an adversarial negative example.
        
        Args:
            clinical_note: Original clinical note
            original_summary: Original (correct) summary
            strategy: Augmentation strategy:
                - "random": Random strategy
                - "entity_swap": Swap medical entities
                - "negation_invert": Invert negations
                - "fabrication": Add fabricated information
                - "multiple": Apply multiple strategies
                
        Returns:
            Dictionary with adversarial example and metadata
        """
        strategies = ["entity_swap", "negation_invert", "fabrication"]
        
        if strategy == "random":
            strategy = random.choice(strategies)
        
        adversarial_summary = original_summary
        modifications = []
        
        if strategy == "entity_swap" or strategy == "multiple":
            adversarial_summary, mod_type = self.swap_entities(adversarial_summary)
            modifications.append(mod_type)
        
        if strategy == "negation_invert" or strategy == "multiple":
            adversarial_summary, inverted = self.invert_negation(adversarial_summary)
            if inverted:
                modifications.append("negation_inverted")
        
        if strategy == "fabrication" or strategy == "multiple":
            adversarial_summary, fab_type = self.fabricate_information(adversarial_summary)
            modifications.append(fab_type)
        
        return {
            "clinical_note": clinical_note,
            "model_summary": adversarial_summary,
            "label": "hallucinated",
            "hallucination_type": "adversarial_" + "_".join(modifications),
            "original_summary": original_summary,
            "modifications": modifications
        }


if __name__ == "__main__":
    # Test the augmenter
    augmenter = AdversarialAugmenter()
    
    clinical_note = "Patient reports mild chest pain for 2 days. ECG normal. No history of hypertension."
    original_summary = "The patient has chest pain with normal ECG findings and no hypertension history."
    
    print("Adversarial Augmentation Test:")
    print("-" * 60)
    print(f"Clinical Note: {clinical_note}")
    print(f"Original Summary: {original_summary}")
    print()
    
    for strategy in ["entity_swap", "negation_invert", "fabrication", "multiple"]:
        result = augmenter.generate_adversarial_negative(clinical_note, original_summary, strategy)
        print(f"Strategy: {strategy}")
        print(f"Adversarial: {result['model_summary']}")
        print(f"Modifications: {result['modifications']}")
        print()
