"""
DPO Triplet Generator Module
Generates (prompt, chosen, rejected) triplets for Direct Preference Optimization training.
"""

from typing import List, Dict, Tuple
import json
from adversarial_augmenter import AdversarialAugmenter


class DPOTripletGenerator:
    """Generate DPO-compatible triplets from factual and adversarial examples."""
    
    def __init__(self, num_hard_negatives_per_positive: int = 3):
        """
        Initialize the triplet generator.
        
        Args:
            num_hard_negatives_per_positive: Number of adversarial negatives to generate per positive
        """
        self.augmenter = AdversarialAugmenter()
        self.num_hard_negatives_per_positive = num_hard_negatives_per_positive
    
    def generate_triplet(
        self,
        record_id: str,
        clinical_note: str,
        factual_summary: str,
        adversarial_strategies: List[str] = None
    ) -> List[Dict]:
        """
        Generate DPO triplets from a single positive example.
        
        Args:
            record_id: Unique identifier for the record
            clinical_note: Original clinical note (prompt)
            factual_summary: Correct/factual summary (chosen)
            adversarial_strategies: List of strategies to use for negatives
                Default: ["entity_swap", "negation_invert", "fabrication"]
            
        Returns:
            List of triplet dictionaries
        """
        if adversarial_strategies is None:
            adversarial_strategies = ["entity_swap", "negation_invert", "fabrication"]
        
        triplets = []
        
        # Generate multiple hard negatives
        for strategy_idx, strategy in enumerate(adversarial_strategies):
            adversarial_result = self.augmenter.generate_adversarial_negative(
                clinical_note,
                factual_summary,
                strategy=strategy
            )
            
            triplet = {
                "id": f"{record_id}_dpo_{strategy_idx}",
                "prompt": clinical_note,
                "chosen": factual_summary,
                "rejected": adversarial_result["model_summary"],
                "data_format": "dpo_triplet",
                "hallucination_type": adversarial_result["hallucination_type"],
                "strategy_used": strategy,
                "modifications": adversarial_result.get("modifications", []),
            }
            
            triplets.append(triplet)
        
        return triplets
    
    def generate_triplets_batch(
        self,
        records: List[Dict],
        strategies: List[str] = None
    ) -> List[Dict]:
        """
        Generate DPO triplets for a batch of records.
        
        Args:
            records: List of record dictionaries with 'id', 'clinical_note', 'model_summary', 'label'
            strategies: Adversarial strategies to apply
            
        Returns:
            List of triplet dictionaries
        """
        if strategies is None:
            strategies = ["entity_swap", "negation_invert", "fabrication"]
        
        triplets = []
        
        for record in records:
            # Only generate triplets from factual examples
            if record.get("label") != "factual":
                continue
            
            record_triplets = self.generate_triplet(
                record_id=record.get("id", "unknown"),
                clinical_note=record.get("clinical_note", ""),
                factual_summary=record.get("model_summary", ""),
                adversarial_strategies=strategies
            )
            
            triplets.extend(record_triplets)
        
        return triplets
    
    def triplet_to_row(self, triplet: Dict) -> Dict:
        """
        Convert triplet to CSV row format.
        
        Args:
            triplet: Triplet dictionary
            
        Returns:
            Dictionary suitable for CSV output
        """
        return {
            "id": triplet["id"],
            "prompt": triplet["prompt"],
            "chosen": triplet["chosen"],
            "rejected": triplet["rejected"],
            "data_format": triplet["data_format"],
            "hallucination_type": triplet["hallucination_type"],
            "strategy_used": triplet["strategy_used"],
            "modifications": json.dumps(triplet.get("modifications", [])),
        }
    
    def validate_triplet(self, triplet: Dict) -> Tuple[bool, str]:
        """
        Validate a triplet for DPO training.
        
        Args:
            triplet: Triplet dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["prompt", "chosen", "rejected"]
        
        for field in required_fields:
            if field not in triplet or not triplet[field]:
                return False, f"Missing or empty field: {field}"
        
        if not isinstance(triplet["prompt"], str) or len(triplet["prompt"].strip()) == 0:
            return False, "Prompt cannot be empty"
        
        if not isinstance(triplet["chosen"], str) or len(triplet["chosen"].strip()) == 0:
            return False, "Chosen output cannot be empty"
        
        if not isinstance(triplet["rejected"], str) or len(triplet["rejected"].strip()) == 0:
            return False, "Rejected output cannot be empty"
        
        # Chosen and rejected should be different
        if triplet["chosen"].strip() == triplet["rejected"].strip():
            return False, "Chosen and rejected outputs are identical"
        
        return True, ""


if __name__ == "__main__":
    # Test the triplet generator
    generator = DPOTripletGenerator()
    
    clinical_note = "Patient reports mild chest pain for 2 days. ECG normal. No history of hypertension."
    factual_summary = "The patient has chest pain with normal ECG findings and no hypertension history."
    
    print("DPO Triplet Generator Test:")
    print("-" * 80)
    print(f"Clinical Note: {clinical_note}")
    print(f"Factual Summary: {factual_summary}")
    print()
    
    triplets = generator.generate_triplet(
        record_id="test_001",
        clinical_note=clinical_note,
        factual_summary=factual_summary
    )
    
    for idx, triplet in enumerate(triplets):
        print(f"Triplet {idx + 1}:")
        print(f"  ID: {triplet['id']}")
        print(f"  Strategy: {triplet['strategy_used']}")
        print(f"  Prompt: {triplet['prompt'][:60]}...")
        print(f"  Chosen: {triplet['chosen'][:60]}...")
        print(f"  Rejected: {triplet['rejected'][:60]}...")
        print()
        
        is_valid, error = generator.validate_triplet(triplet)
        print(f"  Valid: {is_valid} {f'({error})' if error else ''}")
        print()
