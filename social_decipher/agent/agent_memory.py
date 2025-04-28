from typing import Dict, List, Any
import json
import random


class AgentMemory:
    """
    Memory system for social agents to maintain knowledge across multiple scenarios
    """
    def __init__(self, agent_name: str, partner_name: str):
        self.agent_name = agent_name
        self.partner_name = partner_name
        self.scenarios_participated = 0
        
        # Partner insights
        self.partner_insights = {
            "communication_style": None,  # concise, balanced, verbose
            "response_patterns": []       # observed response patterns
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
        
        # Most important memories (limited to prevent prompt bloat)
        self.key_memories = []  # Limited to 3 items
        
    def update_after_scenario(self, 
                             scenario_log: List[str], 
                             scenario_results: Dict[str, Any], 
                             agent_goal: str, 
                             goal_achieved: bool, 
                             encryption_enabled: bool = False):
        """Update memory after a completed scenario"""
        self.scenarios_participated += 1
        
        # Update goal history
        self.goals_history.append({
            "scenario_num": self.scenarios_participated,
            "goal": agent_goal,
            "achieved": goal_achieved
        })
        
        # Limit goal history to most recent 5
        if len(self.goals_history) > 5:
            self.goals_history = self.goals_history[-5:]
        
        # Update partner insights
        self._update_partner_insights(scenario_log)
        
        # Process language barrier adaptations if enabled
        if encryption_enabled:
            self.language_barrier["detected"] = True
            self._update_language_strategies(scenario_log, goal_achieved)
            
        # Create a key memory from this scenario
        self._add_key_memory(scenario_log, goal_achieved)
            
    def _update_partner_insights(self, scenario_log: List[str]):
        """Extract simple patterns about partner communication"""
        partner_messages = [msg.split(":", 1)[1] for msg in scenario_log if msg.startswith(f"{self.partner_name}:")]
        
        if not partner_messages:
            return
            
        # Analyze message length
        avg_length = sum(len(msg.split()) for msg in partner_messages) / len(partner_messages)
        
        if avg_length < 10:
            self.partner_insights["communication_style"] = "concise"
        elif avg_length > 20:
            self.partner_insights["communication_style"] = "verbose"
        else:
            self.partner_insights["communication_style"] = "balanced"
            
        # Check for basic response patterns
        if any(msg.strip().startswith(("Yes", "yes", "Yeah", "Okay", "Sure")) for msg in partner_messages):
            if "tends to agree readily" not in self.partner_insights["response_patterns"]:
                self.partner_insights["response_patterns"].append("tends to agree readily")
                
        if any(msg.count("?") > 0 for msg in partner_messages):
            if "asks clarifying questions" not in self.partner_insights["response_patterns"]:
                self.partner_insights["response_patterns"].append("asks clarifying questions")
                
        if any("don't understand" in msg.lower() or "confused" in msg.lower() for msg in partner_messages):
            if "expresses confusion directly" not in self.partner_insights["response_patterns"]:
                self.partner_insights["response_patterns"].append("expresses confusion directly")
    
    def _update_language_strategies(self, scenario_log: List[str], goal_achieved: bool):
        """Update language barrier strategies based on conversation outcome"""
        # Simple strategy detection
        strategy_keywords = {
            "simplification": ["simple", "simpler", "simplified", "basic", "easier"],
            "repetition": ["repeat", "again", "reiterate", "say again"],
            "gesturing": ["gesture", "pointing", "wave", "hand", "motion"],
            "visual_description": ["show", "draw", "picture", "image", "visual"],
            "key_words": ["key", "important", "essential", "main", "critical"]
        }
        
        # Count strategies used in this conversation
        strategies_used = {}
        for msg in scenario_log:
            if msg.startswith(f"{self.agent_name}:"):
                message = msg.lower()
                for strategy, keywords in strategy_keywords.items():
                    if any(keyword in message for keyword in keywords):
                        strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        # Update strategy effectiveness 
        effectiveness_value = 0.7 if goal_achieved else 0.3  # Positive/negative adjustment
        
        for strategy in strategies_used:
            # Simple weighted average update
            current = self.language_barrier["strategies"][strategy]
            self.language_barrier["strategies"][strategy] = (current * 0.7) + (effectiveness_value * 0.3)
        
        # Identify working and difficult phrases
        self._identify_phrase_effectiveness(scenario_log)
    
    def _identify_phrase_effectiveness(self, scenario_log: List[str]):
        """Identify which phrases seemed to work or cause confusion"""
        for i in range(len(scenario_log) - 1):
            if (scenario_log[i].startswith(f"{self.agent_name}:") and 
                scenario_log[i+1].startswith(f"{self.partner_name}:")):
                
                agent_msg = scenario_log[i].split(":", 1)[1].strip()
                partner_msg = scenario_log[i+1].split(":", 1)[1].strip().lower()
                
                # Simple message was understood
                if "yes" in partner_msg or "understand" in partner_msg or "got it" in partner_msg:
                    # Extract a short phrase
                    words = agent_msg.split()
                    if len(words) >= 3:
                        phrase = " ".join(words[:3])  # First 3 words
                        if phrase not in self.language_barrier["working_phrases"]:
                            self.language_barrier["working_phrases"].append(phrase)
                
                # Message caused confusion
                elif "don't understand" in partner_msg or "confused" in partner_msg or "what?" in partner_msg:
                    words = agent_msg.split()
                    if len(words) >= 3:
                        phrase = " ".join(words[:3])  # First 3 words
                        if phrase not in self.language_barrier["difficult_phrases"]:
                            self.language_barrier["difficult_phrases"].append(phrase)
        
        # Keep lists to a reasonable size
        if len(self.language_barrier["working_phrases"]) > 3:
            self.language_barrier["working_phrases"] = self.language_barrier["working_phrases"][-3:]
            
        if len(self.language_barrier["difficult_phrases"]) > 3:
            self.language_barrier["difficult_phrases"] = self.language_barrier["difficult_phrases"][-3:]
    
    def _add_key_memory(self, scenario_log: List[str], goal_achieved: bool):
        """Add a key memory from this scenario"""
        # Extract one significant exchange
        if len(scenario_log) >= 2:
            # Try to find a meaningful exchange
            for i in range(len(scenario_log) - 1):
                if (scenario_log[i].startswith(f"{self.agent_name}:") and 
                    scenario_log[i+1].startswith(f"{self.partner_name}:")):
                    
                    agent_msg = scenario_log[i].split(":", 1)[1].strip()
                    partner_msg = scenario_log[i+1].split(":", 1)[1].strip()
                    
                    if len(agent_msg) > 10 and len(partner_msg) > 10:  # Skip trivial exchanges
                        memory = {
                            "scenario": self.scenarios_participated,
                            "goal_achieved": goal_achieved,
                            "exchange": {
                                "said": agent_msg[:50] + "..." if len(agent_msg) > 50 else agent_msg,
                                "response": partner_msg[:50] + "..." if len(partner_msg) > 50 else partner_msg
                            }
                        }
                        
                        self.key_memories.append(memory)
                        break
        
        if len(self.key_memories) > 3:
            self.key_memories = self.key_memories[-3:]
    
    def get_memory_context(self, detailed: bool = False) -> str:
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
        
        # Add key memories for detailed context
        if detailed and self.key_memories:
            memory_lines.append("\nMemories from past interactions:")
            latest_memory = self.key_memories[-1]
            memory_lines.append(f"- When you said \"{latest_memory['exchange']['said']}\",")
            memory_lines.append(f"  {self.partner_name} responded: \"{latest_memory['exchange']['response']}\"")
            
            if len(self.key_memories) > 1:
                earlier_memory = self.key_memories[-2]
                memory_lines.append(f"- In another conversation, when you said \"{earlier_memory['exchange']['said']}\",")
                memory_lines.append(f"  {self.partner_name} responded: \"{earlier_memory['exchange']['response']}\"")
        
        return "\n".join(memory_lines)
    
    def to_dict(self) -> Dict:
        """Convert memory to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "partner_name": self.partner_name,
            "scenarios_participated": self.scenarios_participated,
            "partner_insights": self.partner_insights,
            "goals_history": self.goals_history,
            "language_barrier": self.language_barrier,
            "key_memories": self.key_memories
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMemory':
        """Create memory from dictionary"""
        memory = cls(data["agent_name"], data["partner_name"])
        memory.scenarios_participated = data["scenarios_participated"]
        memory.partner_insights = data["partner_insights"]
        memory.goals_history = data["goals_history"]
        memory.language_barrier = data["language_barrier"]
        memory.key_memories = data["key_memories"]
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