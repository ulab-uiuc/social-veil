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
    Long Short-Term Memory system for social agents to maintain knowledge across multiple scenarios
    """
    def __init__(self, agent_name: str, partner_name: str, 
                 short_term_capacity: int = 20, long_term_capacity: int = 100):
        self.agent_name = agent_name
        self.partner_name = partner_name
        self.scenarios_participated = 0
        
        # Memory capacities
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        
        # Short-term memory (recent experiences, working memory)
        self.short_term_memory: List[MemoryItem] = []
        
        # Long-term memory (important, persistent knowledge)
        self.long_term_memory: List[MemoryItem] = []
        
        # Partner insights (structured knowledge)
        self.partner_insights = {
            "communication_style": None,  # concise, balanced, verbose
            "response_patterns": [],      # observed response patterns
            "personality_traits": [],     # inferred personality traits
            "preferences": []            # learned preferences
        }
        
        # Goal tracking
        self.goals_history = []
        
        # Language barrier adaptation
        self.language_barrier = {
            "detected": False,
            "strategies": {
                "simplification": 0,   # Effectiveness score (0-1)
                "repetition": 0,
                "gesturing": 0,
                "visual_description": 0,
                "key_words": 0
            },
            "working_phrases": [],     # Phrases that seemed to work
            "difficult_phrases": []    # Phrases that caused confusion
        }
        
    def add_memory(self, content: str, memory_type: str, importance: float = 0.5, 
                   scenario_id: Optional[int] = None, force_long_term: bool = False):
        """Add a new memory item"""
        memory_item = MemoryItem(content, memory_type, importance, scenario_id=scenario_id)
        
        if force_long_term or importance > 0.7:
            self._add_to_long_term(memory_item)
        else:
            self._add_to_short_term(memory_item)
            
    def _add_to_short_term(self, memory_item: MemoryItem):
        """Add memory to short-term storage"""
        self.short_term_memory.append(memory_item)
        
        # Maintain capacity
        if len(self.short_term_memory) > self.short_term_capacity:
            self._consolidate_short_term()
            
    def _add_to_long_term(self, memory_item: MemoryItem):
        """Add memory to long-term storage"""
        self.long_term_memory.append(memory_item)
        
        # Maintain capacity
        if len(self.long_term_memory) > self.long_term_capacity:
            self._prune_long_term()
            
    def _consolidate_short_term(self):
        """Consolidate short-term memory by moving important items to long-term"""
        current_time = datetime.now()
        
        # Calculate importance scores for all short-term memories
        memory_scores = []
        for i, memory in enumerate(self.short_term_memory):
            decay = memory.decay_score(current_time)
            total_score = memory.importance * decay
            memory_scores.append((i, total_score, memory))
            
        # Sort by score (highest first)
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top memories in short-term, move others to long-term or discard
        keep_count = self.short_term_capacity // 2
        
        new_short_term = []
        for i, score, memory in memory_scores[:keep_count]:
            new_short_term.append(memory)
            
        # Move high-importance memories to long-term
        for i, score, memory in memory_scores[keep_count:]:
            if memory.importance > 0.6:
                self._add_to_long_term(memory)
                
        self.short_term_memory = new_short_term
        
    def _prune_long_term(self):
        """Remove least important long-term memories"""
        current_time = datetime.now()
        
        # Calculate retention scores
        memory_scores = []
        for i, memory in enumerate(self.long_term_memory):
            decay = memory.decay_score(current_time)
            retention_score = memory.importance * decay
            memory_scores.append((i, retention_score))
            
        # Sort by retention score (lowest first)
        memory_scores.sort(key=lambda x: x[1])
        
        # Remove bottom 20% of memories
        remove_count = len(self.long_term_memory) // 5
        indices_to_remove = [i for i, _ in memory_scores[:remove_count]]
        
        # Remove in reverse order to maintain indices
        for i in sorted(indices_to_remove, reverse=True):
            del self.long_term_memory[i]
            
    def search_memories(self, query: str, memory_type: Optional[str] = None, 
                       limit: int = 5) -> List[MemoryItem]:
        """Search memories by content and type"""
        all_memories = self.short_term_memory + self.long_term_memory
        query_lower = query.lower()
        
        # Filter by type if specified
        if memory_type:
            all_memories = [m for m in all_memories if m.memory_type == memory_type]
            
        # Simple keyword matching
        relevant_memories = []
        for memory in all_memories:
            if query_lower in memory.content.lower():
                memory.access()  # Mark as accessed
                relevant_memories.append(memory)
                
        # Sort by importance and recency
        relevant_memories.sort(key=lambda m: (m.importance, m.last_accessed), reverse=True)
        
        return relevant_memories[:limit]
        
    def get_recent_memories(self, hours: int = 24, limit: int = 10) -> List[MemoryItem]:
        """Get memories from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_memories = []
        
        for memory in self.short_term_memory + self.long_term_memory:
            if memory.timestamp >= cutoff_time:
                recent_memories.append(memory)
                
        recent_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return recent_memories[:limit]
        
    def update_after_scenario(self, 
                             scenario_log: List[str], 
                             scenario_results: Dict[str, Any], 
                             agent_goal: str, 
                             goal_achieved: bool, 
                             encryption_enabled: bool = False):
        """Update memory after a completed scenario"""
        self.scenarios_participated += 1
        scenario_id = self.scenarios_participated
        
        # Add scenario summary to memory
        success_status = "successfully" if goal_achieved else "unsuccessfully"
        scenario_summary = f"Scenario {scenario_id}: {success_status} attempted to {agent_goal}"
        self.add_memory(scenario_summary, "goal", importance=0.8, scenario_id=scenario_id)
        
        # Update goal history
        self.goals_history.append({
            "scenario_num": scenario_id,
            "goal": agent_goal,
            "achieved": goal_achieved
        })
        
        # Limit goal history to most recent 5
        if len(self.goals_history) > 5:
            self.goals_history = self.goals_history[-5:]
        
        # Extract and store key conversation moments
        self._extract_conversation_memories(scenario_log, scenario_id)
        
        # Update partner insights
        self._update_partner_insights(scenario_log)
        
        # Process language barrier adaptations if enabled
        if encryption_enabled:
            self.language_barrier["detected"] = True
            self._identify_phrase_effectiveness(scenario_log)
            
        # Consolidate memories
        self._consolidate_short_term()
            
    def _extract_conversation_memories(self, scenario_log: List[str], scenario_id: int):
        """Extract important conversation moments"""
        for i in range(len(scenario_log) - 1):
            if (scenario_log[i].startswith(f"{self.agent_name}:") and 
                scenario_log[i+1].startswith(f"{self.partner_name}:")):
                
                agent_msg = scenario_log[i].split(":", 1)[1].strip()
                partner_msg = scenario_log[i+1].split(":", 1)[1].strip()
                
                # Determine importance based on message characteristics
                importance = 0.3  # Base importance
                
                # Increase importance for longer, more substantive exchanges
                if len(agent_msg) > 20 and len(partner_msg) > 20:
                    importance += 0.2
                    
                # Increase importance for emotional content
                emotional_words = ["love", "hate", "angry", "happy", "sad", "excited", "worried"]
                if any(word in agent_msg.lower() or word in partner_msg.lower() for word in emotional_words):
                    importance += 0.2
                    
                # Increase importance for questions and answers
                if "?" in agent_msg or "?" in partner_msg:
                    importance += 0.1
                    
                # Store the exchange if it's important enough
                if importance > 0.4:
                    exchange_content = f"Agent: {agent_msg} | Partner: {partner_msg}"
                    self.add_memory(exchange_content, "conversation", importance, scenario_id)
                    
    def _update_partner_insights(self, scenario_log: List[str]):
        """Extract and store partner insights"""
        partner_messages = [msg.split(":", 1)[1] for msg in scenario_log if msg.startswith(f"{self.partner_name}:")]
        
        if not partner_messages:
            return
            
        # Analyze communication style
        avg_length = sum(len(msg.split()) for msg in partner_messages) / len(partner_messages)
        
        if avg_length < 10:
            style = "concise"
        elif avg_length > 20:
            style = "verbose"
        else:
            style = "balanced"
            
        if self.partner_insights["communication_style"] != style:
            self.partner_insights["communication_style"] = style
            insight_content = f"{self.partner_name} communicates in a {style} style"
            self.add_memory(insight_content, "insight", importance=0.7)
            
        # Extract response patterns
        patterns = []
        if any(msg.strip().startswith(("Yes", "yes", "Yeah", "Okay", "Sure")) for msg in partner_messages):
            patterns.append("tends to agree readily")
            
        if any(msg.count("?") > 0 for msg in partner_messages):
            patterns.append("asks clarifying questions")
            
        if any("don't understand" in msg.lower() or "confused" in msg.lower() for msg in partner_messages):
            patterns.append("expresses confusion directly")
            
        # Store new patterns
        for pattern in patterns:
            if pattern not in self.partner_insights["response_patterns"]:
                self.partner_insights["response_patterns"].append(pattern)
                insight_content = f"{self.partner_name} {pattern}"
                self.add_memory(insight_content, "insight", importance=0.6)
                
    def _identify_phrase_effectiveness(self, scenario_log: List[str]):
        """Identify which phrases seemed to work or cause confusion"""
        for i in range(len(scenario_log) - 1):
            if (scenario_log[i].startswith(f"{self.agent_name}:") and 
                scenario_log[i+1].startswith(f"{self.partner_name}:")):
                
                agent_msg = scenario_log[i].split(":", 1)[1].strip()
                partner_msg = scenario_log[i+1].split(":", 1)[1].strip().lower()
                
                # Simple message was understood
                if "yes" in partner_msg or "understand" in partner_msg or "got it" in partner_msg:
                    words = agent_msg.split()
                    if len(words) >= 3:
                        phrase = " ".join(words[:3])
                        if phrase not in self.language_barrier["working_phrases"]:
                            self.language_barrier["working_phrases"].append(phrase)
                            strategy_content = f"Effective phrase: '{phrase}'"
                            self.add_memory(strategy_content, "strategy", importance=0.8)
                
                # Message caused confusion
                elif "don't understand" in partner_msg or "confused" in partner_msg or "what?" in partner_msg:
                    words = agent_msg.split()
                    if len(words) >= 3:
                        phrase = " ".join(words[:3])
                        if phrase not in self.language_barrier["difficult_phrases"]:
                            self.language_barrier["difficult_phrases"].append(phrase)
                            strategy_content = f"Confusing phrase: '{phrase}'"
                            self.add_memory(strategy_content, "strategy", importance=0.7)
        
        # Keep lists to a reasonable size
        if len(self.language_barrier["working_phrases"]) > 3:
            self.language_barrier["working_phrases"] = self.language_barrier["working_phrases"][-3:]
            
        if len(self.language_barrier["difficult_phrases"]) > 3:
            self.language_barrier["difficult_phrases"] = self.language_barrier["difficult_phrases"][-3:]
    
    def get_memory_context(self, detailed: bool = False, include_recent: bool = True) -> str:
        """Generate memory context for agent instructions"""
        if self.scenarios_participated == 0:
            return "This is your first interaction with " + self.partner_name + "."
        
        memory_lines = []
        memory_lines.append(f"You have interacted with {self.partner_name} across {self.scenarios_participated} different scenarios.")
        
        # Add goal achievements
        successes = sum(1 for goal in self.goals_history if goal["achieved"])
        memory_lines.append(f"You have achieved {successes} out of {self.scenarios_participated} social goals in past interactions.")
        
        # Add partner insights
        if self.partner_insights["communication_style"]:
            memory_lines.append(f"{self.partner_name} tends to be {self.partner_insights['communication_style']} in their communication style.")
        
        if self.partner_insights["response_patterns"]:
            patterns = ", ".join(self.partner_insights["response_patterns"][:2])
            memory_lines.append(f"You've noticed that {self.partner_name} {patterns}.")
        
        # Add recent memories if requested
        if include_recent:
            recent_memories = self.get_recent_memories(hours=48, limit=3)
            if recent_memories:
                memory_lines.append("\nRecent experiences:")
                for memory in recent_memories:
                    memory_lines.append(f"- {memory.content}")
        
        # Add language barrier strategies if applicable
        if self.language_barrier["detected"]:
            memory_lines.append("\nWhen communication is difficult:")
            
            # Add top strategy
            strategies = [(s, score) for s, score in self.language_barrier["strategies"].items()]
            strategies.sort(key=lambda x: x[1], reverse=True)
            
            if strategies and strategies[0][1] > 0.4:
                memory_lines.append(f"- Using {strategies[0][0]} has helped you communicate effectively.")
            
            if len(strategies) > 1 and strategies[1][1] > 0.4:
                memory_lines.append(f"- {strategies[1][0].capitalize()} has also been helpful.")
            
            # Add examples of working phrases if available
            if self.language_barrier["working_phrases"]:
                memory_lines.append("- Phrases like these seemed to work well:")
                for phrase in self.language_barrier["working_phrases"][:2]:
                    memory_lines.append(f"  \"{phrase}...\"")
        
        # Add detailed long-term memories if requested
        if detailed and self.long_term_memory:
            memory_lines.append("\nImportant long-term memories:")
            # Get top 3 most important long-term memories
            important_memories = sorted(self.long_term_memory, 
                                      key=lambda m: m.importance, reverse=True)[:3]
            for memory in important_memories:
                memory_lines.append(f"- {memory.content}")
        
        return "\n".join(memory_lines)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "short_term_capacity": self.short_term_capacity,
            "long_term_capacity": self.long_term_capacity,
            "total_scenarios": self.scenarios_participated,
            "memory_types": {
                "conversation": len([m for m in self.short_term_memory + self.long_term_memory 
                                   if m.memory_type == "conversation"]),
                "insight": len([m for m in self.short_term_memory + self.long_term_memory 
                              if m.memory_type == "insight"]),
                "strategy": len([m for m in self.short_term_memory + self.long_term_memory 
                               if m.memory_type == "strategy"]),
                "goal": len([m for m in self.short_term_memory + self.long_term_memory 
                           if m.memory_type == "goal"])
            }
        }
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "partner_name": self.partner_name,
            "scenarios_participated": self.scenarios_participated,
            "short_term_capacity": self.short_term_capacity,
            "long_term_capacity": self.long_term_capacity,
            "short_term_memory": [m.to_dict() for m in self.short_term_memory],
            "long_term_memory": [m.to_dict() for m in self.long_term_memory],
            "partner_insights": self.partner_insights,
            "goals_history": self.goals_history,
            "language_barrier": self.language_barrier
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMemory':
        """Create memory from dictionary"""
        memory = cls(data["agent_name"], data["partner_name"], 
                    data.get("short_term_capacity", 20), 
                    data.get("long_term_capacity", 100))
        memory.scenarios_participated = data["scenarios_participated"]
        memory.short_term_memory = [MemoryItem.from_dict(m) for m in data["short_term_memory"]]
        memory.long_term_memory = [MemoryItem.from_dict(m) for m in data["long_term_memory"]]
        memory.partner_insights = data["partner_insights"]
        memory.goals_history = data["goals_history"]
        memory.language_barrier = data["language_barrier"]
        return memory
    
    def save(self, filepath: str):
        """Save memory to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'AgentMemory':
        """Load memory from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)