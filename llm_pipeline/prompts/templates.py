"""
Optimized Prompt Templates
──────────────────────────
Different prompts for different scenarios:
- New questions vs followups
- High quality context vs low quality
- No context found (threshold < 0.3)
"""

from typing import List
from ..core.models import QueryContext, ContextQuality, QueryType


class PromptTemplates:
    """
    Collection of optimized prompt templates.
    Selects appropriate template based on query context.
    """
    
    # ══════════════════════════════════════════════════════════════════════
    # System Prompts (Role definition)
    # ══════════════════════════════════════════════════════════════════════
    
    SYSTEM_PROMPT = """You are a legal document assistant specialized in analyzing contracts, agreements, and legal documents. Your role is to:

1. Answer questions accurately based on the provided document context
2. Be precise and factual - never make assumptions or add information not in the documents
3. Cite specific sections when making claims
4. Admit when information is not available in the provided context
5. Use clear, professional language suitable for legal matters

When answering:
- Always ground your response in the provided context
- Quote relevant passages when appropriate
- Indicate uncertainty if the context is ambiguous
- Be concise but comprehensive"""

    SYSTEM_PROMPT_CONCISE = """You are a legal document assistant. Provide accurate, concise answers based strictly on the provided document context. Always cite sources. If information isn't in the context, say so clearly."""
    
    # ══════════════════════════════════════════════════════════════════════
    # NEW Questions - High Quality Context (score >= 0.7)
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new_question_high_quality(query: str, context: str, citations: List[str]) -> str:
        """New question with highly relevant context."""
        return f"""Based on the following document excerpts, answer the user's question accurately and comprehensively.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer based strictly on the information provided above
- Quote specific passages to support your answer
- Mention the relevant document sections
- Be thorough but clear
- If multiple interpretations exist, mention them

Your answer:"""
    
    # ══════════════════════════════════════════════════════════════════════
    # NEW Questions - Medium/Low Quality Context (0.3 <= score < 0.7)
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new_question_medium_quality(query: str, context: str, citations: List[str]) -> str:
        """New question with somewhat relevant context."""
        return f"""I found some potentially relevant information in the documents, though it may not fully address your question.

AVAILABLE INFORMATION:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer based on the available information
- Clearly indicate what information IS provided in the documents
- Clearly indicate what information IS NOT provided or is unclear
- Suggest what additional information might be needed
- Be honest about limitations

Your answer:"""
    
    # ══════════════════════════════════════════════════════════════════════
    # NEW Questions - No Relevant Context (score < 0.3)
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def new_question_no_context(query: str) -> str:
        """New question with no relevant context found."""
        return f"""The user asked the following question, but I could not find relevant information in the available documents.

USER QUESTION:
{query}

INSTRUCTIONS:
- Politely inform the user that the information is not available in the documents
- Explain what documents are available (if you know from context)
- Suggest alternative approaches:
  * Rephrasing the question
  * Looking in specific document sections
  * Checking if the document has been fully indexed
- DO NOT make up information or provide general legal knowledge
- DO NOT say "I don't know" - instead say "This information is not available in the provided documents"

Your response:"""
    
    # ══════════════════════════════════════════════════════════════════════
    # FOLLOWUP Questions - High Quality Context
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def followup_question_high_quality(
        query: str, 
        context: str, 
        previous_question: str,
        previous_answer: str,
    ) -> str:
        """Follow-up question with relevant context."""
        return f"""The user is asking a follow-up question about a previous topic.

PREVIOUS QUESTION:
{previous_question}

YOUR PREVIOUS ANSWER:
{previous_answer}

CURRENT FOLLOW-UP QUESTION:
{query}

RELEVANT DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Build upon your previous answer
- Reference earlier information when relevant
- Use the new context to provide additional details
- Maintain consistency with previous response
- If the follow-up reveals an error in your previous answer, acknowledge and correct it

Your answer:"""
    
    # ══════════════════════════════════════════════════════════════════════
    # FOLLOWUP Questions - No Additional Context
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def followup_question_no_context(
        query: str,
        previous_question: str,
        previous_answer: str,
    ) -> str:
        """Follow-up question but no new relevant context found."""
        return f"""The user is asking a follow-up question, but no additional relevant information was found in the documents.

PREVIOUS QUESTION:
{previous_question}

YOUR PREVIOUS ANSWER:
{previous_answer}

CURRENT FOLLOW-UP QUESTION:
{query}

INSTRUCTIONS:
- If the question can be answered based on your previous response, do so
- If it requires new information not in the documents, explain this clearly
- Offer to clarify or elaborate on what you already provided
- Suggest rephrasing if the question is unclear
- DO NOT invent new information

Your response:"""
    
    # ══════════════════════════════════════════════════════════════════════
    # CLARIFICATION Questions
    # ══════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def clarification_question(
        query: str,
        context: str,
        previous_answer: str,
    ) -> str:
        """User asking to clarify something from previous answer."""
        return f"""The user is asking you to clarify or explain part of your previous answer.

YOUR PREVIOUS ANSWER:
{previous_answer}

USER'S CLARIFICATION REQUEST:
{query}

AVAILABLE CONTEXT (for reference):
{context}

INSTRUCTIONS:
- Focus on the specific part they're asking about
- Explain more clearly or in simpler terms
- Provide examples if helpful
- Reference the original document sections
- If your previous answer was unclear, acknowledge this and improve it

Your clarification:"""


class PromptSelector:
    """
    Selects the appropriate prompt template based on query context.
    """
    
    @staticmethod
    def select_prompt(
        query_ctx: QueryContext,
        concise: bool = False,
    ) -> str:
        """
        Select and generate the appropriate prompt.
        
        Args:
            query_ctx: QueryContext with query, contexts, history
            concise: Use concise prompt style
            
        Returns:
            Formatted prompt string
        """
        templates = PromptTemplates()
        
        # Get context text and citations
        context_text = query_ctx.get_context_text()
        citations = query_ctx.get_citations()
        
        # Determine query type and context quality
        query_type = query_ctx.query_type
        quality = query_ctx.context_quality
        
        # Get conversation history if available
        previous_q = None
        previous_a = None
        if query_ctx.has_history and len(query_ctx.history) >= 2:
            # Get last user question and assistant answer
            for i in range(len(query_ctx.history) - 1, -1, -1):
                msg = query_ctx.history[i]
                if msg.role == "assistant" and previous_a is None:
                    previous_a = msg.content
                elif msg.role == "user" and previous_q is None:
                    previous_q = msg.content
                
                if previous_q and previous_a:
                    break
        
        # ── NEW QUESTIONS ─────────────────────────────────────────────────
        
        if query_type == QueryType.NEW:
            if quality == ContextQuality.HIGH:
                return templates.new_question_high_quality(
                    query_ctx.query, context_text, citations
                )
            elif quality in (ContextQuality.MEDIUM, ContextQuality.LOW):
                return templates.new_question_medium_quality(
                    query_ctx.query, context_text, citations
                )
            else:  # NO_CONTEXT
                return templates.new_question_no_context(query_ctx.query)
        
        # ── FOLLOWUP QUESTIONS ────────────────────────────────────────────
        
        elif query_type == QueryType.FOLLOWUP:
            if quality in (ContextQuality.HIGH, ContextQuality.MEDIUM):
                return templates.followup_question_high_quality(
                    query_ctx.query, context_text, 
                    previous_q or "Previous question",
                    previous_a or "Previous answer"
                )
            else:
                return templates.followup_question_no_context(
                    query_ctx.query,
                    previous_q or "Previous question", 
                    previous_a or "Previous answer"
                )
        
        # ── CLARIFICATION QUESTIONS ───────────────────────────────────────
        
        elif query_type == QueryType.CLARIFICATION:
            return templates.clarification_question(
                query_ctx.query, context_text,
                previous_a or "Previous answer"
            )
        
        # ── FALLBACK (treat as new question) ─────────────────────────────
        
        else:
            if quality == ContextQuality.HIGH:
                return templates.new_question_high_quality(
                    query_ctx.query, context_text, citations
                )
            else:
                return templates.new_question_medium_quality(
                    query_ctx.query, context_text, citations
                )
    
    @staticmethod
    def get_system_prompt(concise: bool = False) -> str:
        """Get system prompt."""
        if concise:
            return PromptTemplates.SYSTEM_PROMPT_CONCISE
        return PromptTemplates.SYSTEM_PROMPT