import json
import random
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta


class MemoryItem:
    """Individual memory item with metadata"""
    def __init__(self, content: str, memory_type: str, importance: float = 0.5, 
                 timestamp: Optional[datetime] = None, scenario_id: Optional[int] = None):
        self.content = content
        self.memory_type = memory_type  # "conversation", "insight", "strategy", "goal"
        self.importance = max(0.0, min(1.0, importance))  # Clamp between 0 and 1
        self.timestamp = timestamp or datetime.now()
        self.scenario_id = scenario_id
        self.access_count = 0
        self.last_accessed = self.timestamp
        
    def access(self):
        """Mark memory as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
    def decay_score(self, current_time: datetime) -> float:
        """Calculate memory decay score based on time and access patterns"""
        time_diff = (current_time - self.last_accessed).total_seconds() / 3600  # hours
        # Exponential decay with access count bonus
        decay = 0.95 ** time_diff
        access_bonus = min(0.3, self.access_count * 0.05)  # Max 30% bonus from access
        return decay + access_bonus
        
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "scenario_id": self.scenario_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryItem':
        memory = cls(
            content=data["content"],
            memory_type=data["memory_type"],
            importance=data["importance"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            scenario_id=data["scenario_id"]
        )
        memory.access_count = data["access_count"]
        memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return memory


class AgentMemory:
    """
    Within-scenario memory system for social agents (resets between independent scenarios)
    """
    def __init__(self, agent_name: str, partner_name: str, 
                 short_term_capacity: int = 20):
        self.agent_name = agent_name
        self.partner_name = partner_name
        
        # Memory capacity (simplified since scenarios are independent)
        self.short_term_capacity = short_term_capacity
        
        # Short-term memory for within-conversation learning
        self.short_term_memory: List[MemoryItem] = []
        
        # Partner insights learned during this conversation
        self.partner_insights = {
            "communication_style": None,  # concise, balanced, verbose
            "response_patterns": [],      # observed response patterns
        }
        
        # Language barrier adaptation within this conversation
        self.language_barrier = {
            "detected": False,
            "working_phrases": [],     # Phrases that seemed to work
            "difficult_phrases": []    # Phrases that caused confusion
        }
        
    def add_memory(self, content: str, memory_type: str, importance: float = 0.5):
        """Add a new memory item to short-term memory"""
        memory_item = MemoryItem(content, memory_type, importance)
        self.short_term_memory.append(memory_item)
        
        # Simple capacity management - keep most recent items
        if len(self.short_term_memory) > self.short_term_capacity:
            self.short_term_memory = self.short_term_memory[-self.short_term_capacity:]
            
    def search_memories(self, query: str, memory_type: Optional[str] = None, 
                       limit: int = 5) -> List[MemoryItem]:
        """Search memories by content and type"""
        query_lower = query.lower()
        
        # Filter by type if specified
        memories_to_search = self.short_term_memory
        if memory_type:
            memories_to_search = [m for m in memories_to_search if m.memory_type == memory_type]
            
        # Simple keyword matching
        relevant_memories = []
        for memory in memories_to_search:
            if query_lower in memory.content.lower():
                memory.access()  # Mark as accessed
                relevant_memories.append(memory)
                
        # Sort by importance and recency
        relevant_memories.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        return relevant_memories[:limit]
        
    def get_recent_memories(self, hours: int = 1, limit: int = 10) -> List[MemoryItem]:
        """Get memories from the last N hours (within current conversation)"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_memories = []
        
        for memory in self.short_term_memory:
            if memory.timestamp >= cutoff_time:
                recent_memories.append(memory)
                
        recent_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return recent_memories[:limit]
        
    def reset_for_new_scenario(self):
        """Reset memory for a new independent scenario simulation"""
        # Clear all memories since scenarios are independent
        self.short_term_memory = []
        
        # Reset conversation-specific insights
        self.partner_insights = {
            "communication_style": None,
            "response_patterns": [],
        }
        
        # Reset language barrier learning
        self.language_barrier = {
            "detected": False,
            "working_phrases": [],
            "difficult_phrases": []
        }

    def update_from_exchange(self, agent_message: str, partner_response: str, turn_number: int):
        """Update memory from a single conversation exchange in real-time (within scenario only)"""
        partner_response_lower = partner_response.lower()
        
        # Quick phrase effectiveness learning for language barriers
        if self.language_barrier["detected"] or any(word in partner_response_lower for word in ["don't understand", "confused", "what?", "huh?"]):
            # Mark language barrier as detected
            self.language_barrier["detected"] = True
            
            # Extract failed phrase (first 3 words of agent message)
            words = agent_message.split()
            if len(words) >= 2:
                failed_phrase = " ".join(words[:min(3, len(words))])
                if failed_phrase not in self.language_barrier["difficult_phrases"]:
                    self.language_barrier["difficult_phrases"].append(failed_phrase)
                    # Limit list size
                    if len(self.language_barrier["difficult_phrases"]) > 3:
                        self.language_barrier["difficult_phrases"] = self.language_barrier["difficult_phrases"][-3:]
                    
                    # Add to memory for within-scenario learning only
                    self.add_memory(
                        f"Turn {turn_number}: '{failed_phrase}' caused confusion with {self.partner_name}",
                        "strategy", 
                        importance=0.9
                    )
        
        elif any(word in partner_response_lower for word in ["yes", "understand", "got it", "ok", "okay", "sure"]):
            # Extract successful phrase (first 3 words of agent message)
            words = agent_message.split()
            if len(words) >= 2:
                success_phrase = " ".join(words[:min(3, len(words))])
                if success_phrase not in self.language_barrier["working_phrases"]:
                    self.language_barrier["working_phrases"].append(success_phrase)
                    # Limit list size
                    if len(self.language_barrier["working_phrases"]) > 3:
                        self.language_barrier["working_phrases"] = self.language_barrier["working_phrases"][-3:]
                    
                    # Add to memory for within-scenario learning only
                    self.add_memory(
                        f"Turn {turn_number}: '{success_phrase}' worked well with {self.partner_name}",
                        "strategy",
                        importance=0.8
                    )
        
        # Quick partner communication style analysis
        response_length = len(partner_response.split())
        if response_length < 5:
            current_style = "concise"
        elif response_length > 15:
            current_style = "verbose"  
        else:
            current_style = "balanced"
            
        # Update communication style if it's changed or not set
        if (self.partner_insights["communication_style"] != current_style and 
            self.partner_insights["communication_style"] is None):
            self.partner_insights["communication_style"] = current_style
            self.add_memory(
                f"Turn {turn_number}: {self.partner_name} seems to prefer {current_style} communication",
                "insight",
                importance=0.7
            )


    
    def get_memory_context(self, detailed: bool = False, include_recent: bool = True) -> str:
        """Generate memory context for agent instructions (within-scenario learning only)"""
        memory_lines = []
        
        # Since scenarios are independent, focus on current conversation learning
        if not self.short_term_memory and not self.long_term_memory:
            return f"This is a new conversation with {self.partner_name}."
        
        # Add partner insights learned during this conversation
        if self.partner_insights["communication_style"]:
            memory_lines.append(f"So far in this conversation, {self.partner_name} seems to prefer {self.partner_insights['communication_style']} communication.")
        
        if self.partner_insights["response_patterns"]:
            patterns = ", ".join(self.partner_insights["response_patterns"][:2])
            memory_lines.append(f"During this conversation, you've noticed that {self.partner_name} {patterns}.")
        
        # Add recent learning from this conversation
        if include_recent:
            recent_memories = self.get_recent_memories(hours=1, limit=3)  # Only very recent within conversation
            if recent_memories:
                memory_lines.append("\nWhat you've learned in this conversation:")
                for memory in recent_memories:
                    memory_lines.append(f"- {memory.content}")
        
        # Add language barrier strategies learned during this conversation
        if self.language_barrier["detected"]:
            memory_lines.append("\nCommunication patterns noticed in this conversation:")
            
            # Add examples of working/difficult phrases discovered in this conversation
            if self.language_barrier["working_phrases"]:
                memory_lines.append("- Phrases that seemed to work well:")
                for phrase in self.language_barrier["working_phrases"]:
                    memory_lines.append(f"  \"{phrase}...\"")
            
            if self.language_barrier["difficult_phrases"]:
                memory_lines.append("- Phrases that caused confusion:")
                for phrase in self.language_barrier["difficult_phrases"]:
                    memory_lines.append(f"  \"{phrase}...\"")
        
        # Add detailed memories from this conversation if requested
        if detailed and self.short_term_memory:
            memory_lines.append("\nKey moments from this conversation:")
            # Get most important memories from current conversation
            important_memories = sorted(self.short_term_memory, 
                                      key=lambda m: m.importance, reverse=True)[:3]
            for memory in important_memories:
                memory_lines.append(f"- {memory.content}")
        
        if not memory_lines:
            return f"This is a new conversation with {self.partner_name}."
        
        return "\n".join(memory_lines)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage within current conversation"""
        return {
            "memory_count": len(self.short_term_memory),
            "capacity": self.short_term_capacity,
            "language_barrier_detected": self.language_barrier["detected"],
            "working_phrases_learned": len(self.language_barrier["working_phrases"]),
            "difficult_phrases_learned": len(self.language_barrier["difficult_phrases"]),
            "communication_style_learned": self.partner_insights["communication_style"] is not None
        }