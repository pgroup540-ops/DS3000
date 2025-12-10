"""
Evidence Annotation Module
Generates summaries with inline pointers to supporting sentences from source text.
"""

import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher


class EvidenceAnnotator:
    """Annotate summaries with evidence pointers to source sentences."""
    
    def __init__(self, similarity_threshold: float = 0.3):
        """
        Initialize the evidence annotator.
        
        Args:
            similarity_threshold: Minimum similarity score to link evidence (0-1)
        """
        self.similarity_threshold = similarity_threshold
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter - handles common medical text patterns
        # Split on periods, but not on common abbreviations
        text = re.sub(r'(?<!\b[A-Z])\.(?=\s+[A-Z])', '.|', text)
        text = re.sub(r'(?<=[0-9])\.(?=\s+[A-Z])', '.|', text)
        
        sentences = [s.strip() for s in text.split('|')]
        sentences = [s for s in sentences if s]
        
        return sentences
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Use SequenceMatcher for basic similarity
        return SequenceMatcher(None, text1, text2).ratio()
    
    def extract_keywords(self, text: str) -> set:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = {w for w in words if w not in stop_words and len(w) > 2}
        
        return keywords
    
    def calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate keyword overlap between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score between 0 and 1
        """
        keywords1 = self.extract_keywords(text1)
        keywords2 = self.extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def find_supporting_sentences(
        self,
        summary_sentence: str,
        source_sentences: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Find source sentences that support a summary sentence.
        
        Args:
            summary_sentence: Sentence from the summary
            source_sentences: List of sentences from the source text
            
        Returns:
            List of tuples (sentence_index, confidence_score)
        """
        supporting = []
        
        for idx, source_sent in enumerate(source_sentences):
            # Calculate combined similarity score
            seq_similarity = self.calculate_similarity(summary_sentence, source_sent)
            keyword_overlap = self.calculate_keyword_overlap(summary_sentence, source_sent)
            
            # Weighted average
            combined_score = 0.6 * seq_similarity + 0.4 * keyword_overlap
            
            if combined_score >= self.similarity_threshold:
                supporting.append((idx, combined_score))
        
        # Sort by confidence score (descending)
        supporting.sort(key=lambda x: x[1], reverse=True)
        
        return supporting
    
    def annotate_with_evidence(
        self,
        clinical_note: str,
        summary: str,
        max_evidence_per_sentence: int = 2
    ) -> Dict[str, any]:
        """
        Annotate a summary with evidence pointers to the clinical note.
        
        Args:
            clinical_note: Original clinical note
            summary: Summary to annotate
            max_evidence_per_sentence: Maximum number of evidence pointers per sentence
            
        Returns:
            Dictionary with annotated summary and metadata
        """
        source_sentences = self.split_sentences(clinical_note)
        summary_sentences = self.split_sentences(summary)
        
        annotated_parts = []
        evidence_map = {}
        total_confidence = 0.0
        num_supported = 0
        
        for sum_idx, summary_sent in enumerate(summary_sentences):
            supporting = self.find_supporting_sentences(summary_sent, source_sentences)
            
            # Take top evidence
            top_evidence = supporting[:max_evidence_per_sentence]
            
            if top_evidence:
                # Create evidence pointer
                evidence_indices = [idx for idx, _ in top_evidence]
                evidence_str = ",".join(str(i+1) for i in evidence_indices)  # 1-indexed
                avg_confidence = sum(score for _, score in top_evidence) / len(top_evidence)
                
                annotated_sent = f"{summary_sent} [Evidence: S{evidence_str}; Conf: {avg_confidence:.2f}]"
                
                evidence_map[sum_idx] = {
                    "sentence": summary_sent,
                    "evidence_indices": evidence_indices,
                    "confidence_scores": [score for _, score in top_evidence],
                    "evidence_text": [source_sentences[i] for i in evidence_indices]
                }
                
                total_confidence += avg_confidence
                num_supported += 1
            else:
                # No supporting evidence found
                annotated_sent = f"{summary_sent} [Evidence: None]"
                evidence_map[sum_idx] = {
                    "sentence": summary_sent,
                    "evidence_indices": [],
                    "confidence_scores": [],
                    "evidence_text": []
                }
            
            annotated_parts.append(annotated_sent)
        
        # Calculate overall statistics
        avg_confidence = total_confidence / num_supported if num_supported > 0 else 0.0
        support_ratio = num_supported / len(summary_sentences) if summary_sentences else 0.0
        
        return {
            "original_summary": summary,
            "annotated_summary": " ".join(annotated_parts),
            "evidence_map": evidence_map,
            "source_sentences": source_sentences,
            "statistics": {
                "total_summary_sentences": len(summary_sentences),
                "supported_sentences": num_supported,
                "unsupported_sentences": len(summary_sentences) - num_supported,
                "support_ratio": support_ratio,
                "average_confidence": avg_confidence
            }
        }
    
    def generate_evidence_augmented_positive(
        self,
        clinical_note: str,
        summary: str
    ) -> Dict[str, any]:
        """
        Generate an evidence-augmented positive example for training.
        
        Args:
            clinical_note: Original clinical note
            summary: Summary to augment
            
        Returns:
            Dictionary with augmented example
        """
        annotation_result = self.annotate_with_evidence(clinical_note, summary)
        
        return {
            "clinical_note": clinical_note,
            "model_summary": annotation_result["annotated_summary"],
            "original_summary": summary,
            "label": "factual_with_evidence",
            "evidence_map": annotation_result["evidence_map"],
            "statistics": annotation_result["statistics"]
        }
    
    def create_evidence_citation_format(
        self,
        clinical_note: str,
        summary: str,
        citation_style: str = "inline"
    ) -> str:
        """
        Create a summary with evidence citations in different formats.
        
        Args:
            clinical_note: Original clinical note
            summary: Summary text
            citation_style: Citation format:
                - "inline": [S1,S2] inline with text
                - "superscript": [1,2] as superscripts
                - "footnote": With footnotes at the end
                
        Returns:
            Formatted summary with citations
        """
        result = self.annotate_with_evidence(clinical_note, summary, max_evidence_per_sentence=3)
        
        if citation_style == "inline":
            return result["annotated_summary"]
        
        elif citation_style == "superscript":
            parts = []
            for idx, sent_data in result["evidence_map"].items():
                sent = sent_data["sentence"]
                if sent_data["evidence_indices"]:
                    evidence_str = ",".join(str(i+1) for i in sent_data["evidence_indices"])
                    parts.append(f"{sent}[{evidence_str}]")
                else:
                    parts.append(sent)
            return " ".join(parts)
        
        elif citation_style == "footnote":
            main_parts = []
            footnotes = []
            
            for idx, sent_data in result["evidence_map"].items():
                sent = sent_data["sentence"]
                if sent_data["evidence_indices"]:
                    evidence_str = ",".join(str(i+1) for i in sent_data["evidence_indices"])
                    main_parts.append(f"{sent}[{evidence_str}]")
                    
                    for ev_idx, ev_text in zip(sent_data["evidence_indices"], sent_data["evidence_text"]):
                        footnotes.append(f"[S{ev_idx+1}] {ev_text}")
                else:
                    main_parts.append(sent)
            
            summary_text = " ".join(main_parts)
            footnote_text = "\n\nEvidence:\n" + "\n".join(footnotes) if footnotes else ""
            
            return summary_text + footnote_text
        
        return result["annotated_summary"]


if __name__ == "__main__":
    # Test the annotator
    annotator = EvidenceAnnotator(similarity_threshold=0.3)
    
    clinical_note = "Patient reports mild chest pain for 2 days. ECG normal. No history of hypertension."
    summary = "The patient has chest pain with normal ECG findings."
    
    print("Evidence Annotation Test:")
    print("-" * 60)
    print(f"Clinical Note: {clinical_note}")
    print(f"Summary: {summary}")
    print()
    
    result = annotator.annotate_with_evidence(clinical_note, summary)
    
    print(f"Annotated Summary: {result['annotated_summary']}")
    print()
    print("Evidence Map:")
    for idx, evidence in result['evidence_map'].items():
        print(f"  Sentence {idx+1}: {evidence['sentence']}")
        print(f"    Evidence indices: {evidence['evidence_indices']}")
        print(f"    Confidence: {evidence['confidence_scores']}")
        print()
    
    print("Statistics:")
    for key, value in result['statistics'].items():
        print(f"  {key}: {value}")
