"""
Query Type Classifier
─────────────────────
Determines if a user's query is:
- NEW: First question or completely new topic
- FOLLOWUP: Continuing previous question
- CLARIFICATION: Asking to explain previous answer
- RELATED: Related but distinct topic
"""

import re
from typing import List
from ..core.models import QueryType, ConversationMessage


class QueryClassifier:
    """
    Classify query type based on content and conversation history.
    Uses simple heuristics (can be upgraded to ML model if needed).
    """
    
    # Keywords indicating follow-up questions
    FOLLOWUP_KEYWORDS = [
        "what about", "how about", "and what", "also",
        "in addition", "furthermore", "moreover",
        "can you explain", "tell me more", "elaborate",
        "go on", "continue", "next",
    ]
    
    # Keywords indicating clarification requests
    CLARIFICATION_KEYWORDS = [
        "what do you mean", "what does that mean", "clarify",
        "explain that", "i don't understand", "unclear",
        "confusing", "rephrase", "say that again",
        "what is", "define", "meaning of",
    ]
    
    # Pronouns/references indicating continuation
    REFERENCE_WORDS = [
        "that", "this", "it", "they", "those", "these",
        "he", "she", "him", "her", "them",
    ]
    
    @staticmethod
    def classify(
        query: str,
        history: List[ConversationMessage] = None,
    ) -> QueryType:
        """
        Classify the query type.
        
        Args:
            query: User's question
            history: Conversation history
            
        Returns:
            QueryType enum
        """
        query_lower = query.lower().strip()
        
        # No history → always NEW
        if not history or len(history) == 0:
            return QueryType.NEW
        
        # Check for clarification requests
        if QueryClassifier._is_clarification(query_lower):
            return QueryType.CLARIFICATION
        
        # Check for follow-up indicators
        if QueryClassifier._is_followup(query_lower, history):
            return QueryType.FOLLOWUP
        
        # Check if query references previous messages
        if QueryClassifier._has_references(query_lower):
            return QueryType.FOLLOWUP
        
        # Check if query is very short (likely a follow-up)
        if len(query.split()) <= 5 and history:
            return QueryType.FOLLOWUP
        
        # Default: treat as new question
        return QueryType.NEW
    
    @staticmethod
    def _is_clarification(query_lower: str) -> bool:
        """Check if query is asking for clarification."""
        for keyword in QueryClassifier.CLARIFICATION_KEYWORDS:
            if keyword in query_lower:
                return True
        
        # Check for question marks with reference words
        if "?" in query_lower:
            for ref in QueryClassifier.REFERENCE_WORDS[:6]:  # that, this, it, they, those, these
                if ref in query_lower.split():
                    return True
        
        return False
    
    @staticmethod
    def _is_followup(query_lower: str, history: List[ConversationMessage]) -> bool:
        """Check if query is a follow-up question."""
        # Check for explicit follow-up keywords
        for keyword in QueryClassifier.FOLLOWUP_KEYWORDS:
            if keyword in query_lower:
                return True
        
        # Check if query starts with "and" or "but"
        if re.match(r'^(and|but|also|or)\b', query_lower):
            return True
        
        # Check for topic continuity (shared keywords with last question)
        if len(history) >= 2:
            last_user_msg = None
            for msg in reversed(history):
                if msg.role == "user":
                    last_user_msg = msg.content.lower()
                    break
            
            if last_user_msg:
                # Extract significant words (3+ chars, not common words)
                common_words = {
                    "the", "is", "are", "was", "were", "what", "when",
                    "where", "who", "how", "why", "can", "will", "does"
                }
                
                current_words = set(
                    w for w in re.findall(r'\b\w{3,}\b', query_lower)
                    if w not in common_words
                )
                
                last_words = set(
                    w for w in re.findall(r'\b\w{3,}\b', last_user_msg)
                    if w not in common_words
                )
                
                # If >30% overlap in keywords → likely follow-up
                if current_words and last_words:
                    overlap = len(current_words & last_words)
                    overlap_ratio = overlap / min(len(current_words), len(last_words))
                    
                    if overlap_ratio > 0.3:
                        return True
        
        return False
    
    @staticmethod
    def _has_references(query_lower: str) -> bool:
        """Check if query uses reference words (pronouns)."""
        # Starts with reference word
        first_word = query_lower.split()[0] if query_lower.split() else ""
        if first_word in QueryClassifier.REFERENCE_WORDS:
            return True
        
        # Contains multiple reference words
        ref_count = sum(1 for word in query_lower.split() if word in QueryClassifier.REFERENCE_WORDS)
        if ref_count >= 2:
            return True
        
        return False


# ══════════════════════════════════════════════════════════════════════════
# Examples for testing
# ══════════════════════════════════════════════════════════════════════════

def _test_classifier():
    """Test the classifier with examples."""
    classifier = QueryClassifier()
    
    # Create mock history
    history = [
        ConversationMessage(role="user", content="What are the payment terms in the contract?"),
        ConversationMessage(role="assistant", content="The payment terms specify net 30 days..."),
    ]
    
    test_cases = [
        # (query, expected_type)
        ("What are the termination terms?", QueryType.NEW),
        ("What about the late fees?", QueryType.FOLLOWUP),
        ("And what happens if payment is late?", QueryType.FOLLOWUP),
        ("Can you clarify that?", QueryType.CLARIFICATION),
        ("What does net 30 mean?", QueryType.CLARIFICATION),
        ("That's confusing, explain it again", QueryType.CLARIFICATION),
        ("Tell me more", QueryType.FOLLOWUP),
        ("Also, are there any penalties?", QueryType.FOLLOWUP),
        ("Who is the landlord?", QueryType.NEW),
    ]
    
    print("Query Type Classification Tests:\n")
    for query, expected in test_cases:
        result = classifier.classify(query, history)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{query}'")
        print(f"   Expected: {expected.value}, Got: {result.value}\n")


if __name__ == "__main__":
    _test_classifier()