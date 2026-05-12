"""
Conversational Memory for Legal RAG
===================================
"""

from typing import List, Dict, Any, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
import time

@dataclass
class ConversationTurn:
    user_query: str
    system_response: str
    citations: List[str]
    timestamp: float

class LegalConversationMemory:
    """
    يحافظ على السياق ويدعم الأسئلة المتتابعة
    """
    
    def __init__(self, max_history: int = 10):
        self.history: Deque[ConversationTurn] = deque(maxlen=max_history)
    
    def add_turn(self, query: str, response: str, citations: List[str]):
        self.history.append(ConversationTurn(
            user_query=query,
            system_response=response[:500],
            citations=citations[:3],
            timestamp=time.time()
        ))
    
    def build_contextual_prompt(self, current_query: str) -> str:
        if not self.history:
            return current_query
        
        context_parts = ["Previous conversation context:"]
        
        for turn in list(self.history)[-3:]:
            context_parts.append(f"User asked: {turn.user_query}")
            context_parts.append(f"Assistant answered: {turn.system_response[:200]}...\n")
        
        context_parts.append(f"Current question (answer based on previous context if relevant): {current_query}")
        
        return "\n".join(context_parts)
    
    def get_last_topic(self) -> Optional[str]:
        if self.history:
            return self.history[-1].user_query[:50]
        return None
    
    def clear(self):
        self.history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_turns": len(self.history),
            "last_topic": self.get_last_topic(),
            "history": [{"query": t.user_query[:50], "timestamp": t.timestamp} for t in self.history]
        }