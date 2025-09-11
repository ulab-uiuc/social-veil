import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
import numpy as np
from scipy import stats
import csv
import asyncio
import aiohttp
from tqdm.asyncio import tqdm

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("⚠️ datasets library not available. Install with: pip install datasets")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from social_decipher.agent.social_agent import SocialAgent
from social_decipher.agent.agent_profile import AgentProfile
from social_decipher.environment.env_profile import EnvironmentProfile
from social_decipher.utils.state import build_dynamic_rules_from_state
from analysis.internal_state.load_existing_episodes import load_all_episodes

# Add canonical barrier prompts to align with main simulation
BARRIER_PROMPTS = {
    "semantic_structure": "You are under a semantic-vagueness constraint. Keep aims implicit and let meanings stay open. Prefer shells like “this”, “that”, or “the thing” instead of concrete labels. Avoid confirmations and denials; answer with stance or direction, not commitments. Offer only the smallest necessary detail if directly and repeatedly pressed, then pivot away.",
    "cultural_style": "You are under a high-context style constraint. Keep the focus on harmony and shared understanding, not transactions. Treat direct task questions as invitations to reflect on values, rapport, and fit. Deflect confirmations and decisions with relationship-first language. If the other party is blunt, register gentle surprise and steer back to atmosphere and alignment.",
    "emotional_influence": "You are under an affective-noise constraint. Let mood and side thoughts lead. Respond to adjacent details or personal reactions rather than the kernel of a question. Keep your stance ambivalent and avoid firm commitments. When misunderstood, lean into how it feels instead of repairing the content. If pressed, share a minimal point and drift back to feeling or context."
}

@dataclass
class MathProblem:
    """Math problem with ground truth"""
    problem_id: str
    source: str  # "gsm8k", "math", "arithmetic"
    original_text: str
    expected_answer: Any  # float for numeric tasks; str letter for MCQ
    problem_type: str  # "word_problem", "equation", "arithmetic", "mcq"

@dataclass
class MathResult:
    """Result of single-agent math solving"""
    problem_id: str
    source: str
    barrier_type: str  # "baseline", "semantic", "cultural", "emotional"
    agent_response: str
    extracted_answer: float
    expected_answer: Any
    reasoning_steps: List[str]
    answer_accuracy: float  # How close to correct answer
    reasoning_clarity: float  # How clear the reasoning was
    success: bool

class SingleAgentMathEvaluator:
    """Evaluates barrier effects on single-agent mathematical reasoning"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: str = "IQ_test/results",
        severity: float = 0.8,
        concurrency: int = 16,
        answer_mode: str = "steps_json",  # "steps_json" | "final_only"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.severity = severity
        self.concurrency = concurrency
        self.answer_mode = answer_mode
        os.makedirs(output_dir, exist_ok=True)
        
        # Load social task templates and global config (to read vLLM port)
        social_cfg_path = project_root / "configs" / "social_task.yaml"
        with open(social_cfg_path, 'r') as f:
            self.templates = yaml.safe_load(f)

        # Apply vLLM port and served model name from config.yaml
        self.vllm_port = os.environ.get("VLLM_PORT", "8000")
        self.served_model_name = "qwen2.5-7b-instruct" # Default
        try:
            main_cfg_path = project_root / "configs" / "config.yaml"
            with open(main_cfg_path, 'r') as f:
                main_cfg = yaml.safe_load(f)
            models_cfg = (main_cfg or {}).get("models", {})
            vllm_port_cfg = models_cfg.get("vllm_port")
            served_name_cfg = models_cfg.get("served_model_name")
            
            if isinstance(vllm_port_cfg, int) and vllm_port_cfg > 0:
                self.vllm_port = str(vllm_port_cfg)
            if isinstance(served_name_cfg, str) and served_name_cfg:
                self.served_model_name = served_name_cfg

        except Exception:
            pass
        self.api_url = f"http://localhost:{self.vllm_port}/v1/chat/completions"
        self.health_url = f"http://localhost:{self.vllm_port}/health"

        self._default_profile: Optional[Dict[str, Any]] = None
        try:
            all_profiles = self.load_profiles_from_episodes()
            # prefer baseline (no barrier)
            for p in all_profiles:
                if not p.get("barrier_type"):
                    self._default_profile = p
                    break
            if not self._default_profile and all_profiles:
                self._default_profile = all_profiles[0]
        except Exception:
            self._default_profile = None

        # Prepare incremental output path
        self._incremental_path = Path(self.output_dir) / "incremental_results.jsonl"
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    
    def load_math_problems(self, limit: int = 50, dataset: str = "all") -> List[MathProblem]:
        """Load GSM8K and AQua-RAT problems for single-agent evaluation.

        If limit == 0, load the entire available split for each dataset.
        """
        if not DATASETS_AVAILABLE:
            print("❌ datasets library not available. Install with: pip install datasets")
            return []

        problems = []
        
        # Load GSM8K from Hugging Face
        if dataset in ["all", "gsm8k"]:
            print("📦 Loading GSM8K from Hugging Face...")
            try:
                gsm8k_dataset = load_dataset("gsm8k", "main", split="test")
                k = len(gsm8k_dataset) if limit == 0 else min(limit, len(gsm8k_dataset))
                gsm8k_samples = list(gsm8k_dataset.shuffle(seed=42).select(range(k)))
                
                loaded_gsm8k = 0
                for i, sample in enumerate(gsm8k_samples):
                    # Extract answer from the solution
                    answer_text = sample["answer"]
                    answer_match = re.search(r'#### ([\d,\.]+)', answer_text)
                    if answer_match:
                        try:
                            answer = float(answer_match.group(1).replace(',', ''))
                            problems.append(MathProblem(
                                problem_id=f"gsm8k_{i}",
                                source="gsm8k",
                                original_text=sample["question"],
                                expected_answer=answer,
                                problem_type="word_problem"
                            ))
                            loaded_gsm8k += 1
                        except ValueError:
                            print(f"⚠️ Could not parse answer for GSM8K problem {i}: {answer_match.group(1)}")
                            continue
                    else:
                        print(f"⚠️ No answer found for GSM8K problem {i}")
                        continue
                
                print(f"✅ Loaded {loaded_gsm8k} GSM8K problems")
            except Exception as e:
                print(f"❌ Failed to load GSM8K: {e}")
                if dataset == "gsm8k":
                    return []
        
        # Load AQuA-RAT MCQ math problems
        if dataset in ["all", "aqua"]:
            print("📦 Loading AQuA-RAT from Hugging Face...")
            try:
                aqua = load_dataset("aqua_rat", split="test")
                # Shuffle and select
                k = len(aqua) if limit == 0 else min(limit, len(aqua))
                samples = list(aqua.shuffle(seed=42).select(range(k)))
                loaded_aqua = 0
                for i, sample in enumerate(samples):
                    q = sample.get("question", "").strip()
                    options = sample.get("options", [])
                    correct = str(sample.get("correct", "")).strip().upper()
                    if not q or not isinstance(options, list) or correct not in {"A","B","C","D","E"}:
                        continue
                    # Format options A-E
                    opt_lines = []
                    labels = ["A","B","C","D","E"]
                    for j, opt in enumerate(options[:5]):
                        try:
                            lbl = labels[j]
                        except IndexError:
                            break
                        opt_lines.append(f"({lbl}) {str(opt).strip()}")
                    if not opt_lines:
                        continue
                    full_text = q + "\n" + "\n".join(opt_lines) + "\nSelect the correct option (A-E) and solve step by step."
                    problems.append(MathProblem(
                        problem_id=f"aqua_{i}",
                        source="aqua",
                        original_text=full_text,
                        expected_answer=correct,
                        problem_type="mcq",
                    ))
                    loaded_aqua += 1
                print(f"✅ Loaded {loaded_aqua} AQuA-RAT problems")
            except Exception as e:
                print(f"❌ Failed to load AQuA-RAT: {e}")
                if dataset == "all":
                    print("📝 Continuing with GSM8K only")
                elif dataset == "aqua":
                    return []

        return problems
    
    def _load_episode_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load episodes from JSON or JSONL path (baseline or barrier variants)."""
        episodes: List[Dict[str, Any]] = []
        if not path.exists():
            return episodes
        try:
            if path.suffix == ".jsonl":
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        episodes.append(json.loads(line))
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        episodes = data
                    elif isinstance(data, dict):
                        episodes = [data]
        except Exception:
            pass
        return episodes

    def load_profiles_from_episodes(self) -> List[Dict[str, Any]]:
        """Extract Agent A profiles and barrier metadata using the central loader.
        Uses analysis.load_existing_episodes.load_all_episodes to ensure consistent episode parsing.
        """
        baseline_path = str(project_root / "data" / "episode_all.jsonl")
        try:
            episodes = load_all_episodes(baseline_file=baseline_path, max_episodes=None)
        except Exception:
            episodes = load_all_episodes()

        profiles: List[Dict[str, Any]] = []
        for ep in episodes:
            try:
                agent_profiles = ep.get("agent_profiles", [{}, {}])
                a_dict = agent_profiles[0] if len(agent_profiles) > 0 else {}
                profiles.append({
                    "episode_id": ep.get("episode_id", "unknown"),
                    "barrier_type": ep.get("barrier_type"),
                    "barrier_prompts": ep.get("barrier_prompts", {}),
                    "barrier_cues": ep.get("barrier_cues", {}),
                    "agentA": a_dict,
                })
            except Exception:
                continue
        return profiles

    def create_math_scenario_from_profile(self, profile: Dict[str, Any], problem: MathProblem) -> Dict[str, Any]:
        """Create scenario using a real Agent A profile and its barrier metadata from episodes."""
        # Build agent profiles: real Agent A + neutral partner
        a = profile.get("agentA", {})
        agent_profiles = [
            a,
            {
                "pk": "dummy_partner",
                "first_name": "Partner",
                "last_name": "Agent",
                "age": 30,
                "gender": "Person",
                "gender_pronoun": "They/them",
                "occupation": "Assistant",
                "public_info": "Helpful assistant",
                "personality_and_values": "Supportive and encouraging",
                "decision_making_style": "Collaborative",
            },
        ]
        scenario_text = f"You need to solve this mathematical problem step by step: {problem.original_text}"
        agent_goals = [
            f"Solve this math problem correctly and show your work: {problem.original_text}",
            "Listen and provide encouragement",
        ]
        agent_reasons = [
            "Use your mathematical skills to arrive at the correct answer",
            "Be supportive",
        ]
        barrier_type = profile.get("barrier_type") or "baseline"
        barrier_prompts = profile.get("barrier_prompts", {})

        # Reconstruct agent profile string for consistency
        agent_a_info = (
            f'{a.get("first_name", "Agent")} {a.get("last_name", "A")}, '
            f'a {a.get("age", "N/A")}-year-old {a.get("occupation", "person")}. '
            f'Public info: {a.get("public_info", "")}'
        ).strip()
        
        return {
            "episode_id": f"{profile.get('episode_id','profile')}_{problem.problem_id}",
            "scenario": scenario_text,
            "agent_profiles": agent_profiles,
            "agent1_profile": agent_a_info, # Add reconstructed profile string
            "agent_goals": agent_goals,
            "agent_reasons": agent_reasons,
            "agent_relationship": "individual_task",
            "barrier_type": barrier_type if barrier_type != "baseline" else None,
            "barrier_prompts": barrier_prompts if barrier_type != "baseline" else {},
            "source": problem.source,
            "math_ground_truth": {
                "expected_answer": problem.expected_answer,
                "problem_type": problem.problem_type,
            },
        }
    
    def _prepare_prompt_for_scenario(self, scenario: Dict[str, Any]) -> Tuple[str, str]:
        """Prepares the full system prompt for a given scenario."""
        environment = EnvironmentProfile(
            scenario=scenario["scenario"],
            agent_goals=scenario["agent_goals"],
            agent_reasons=scenario["agent_reasons"],
            agent_relationship=scenario["agent_relationship"],
            agent1_profile=scenario.get("agent1_profile")
        )
        
        environment.env["barrier_type"] = scenario.get("barrier_type")
        environment.env["barrier_prompts"] = scenario.get("barrier_prompts", {})
        environment.env["barrier_state"] = {"severity": self.severity}
        
        agentA = AgentProfile.from_dict(scenario["agent_profiles"][0], model_id=self.model_name)
        agentB = AgentProfile.from_dict(scenario["agent_profiles"][1], model_id=self.model_name)
        
        solver = SocialAgent(
            name="Alex",
            profile=agentA,
            partner_profile=agentB,
            env=environment,
            role_num=0
        )
        
        raw_instr = solver.build_instruction(transcript="", turn_number=0)

        def _sanitize_instruction_for_eval(text: str) -> str:
            lines = text.split("\n")
            out = []
            skip_next = 0
            for ln in lines:
                if skip_next > 0:
                    skip_next -= 1
                    continue
                s = ln.strip()
                if s.startswith("You are at Turn #") and "Your available action types" in s:
                    continue
                if s.startswith("Please only generate a JSON string including the action type and the argument"):
                    skip_next = 2
                    continue
                out.append(ln)
            if self.answer_mode == "steps_json":
                out.append("IMPORTANT: Your final reasoning step MUST explicitly state the final answer. You must then copy this exact value into the 'answer' field of the JSON.")
                out.append("Output format (JSON only, no extra text or markdown):")
                out.append('{"steps": ["...your reasoning here...", "The final answer is <VALUE>"], "answer": <VALUE>}')
                out.append("- For GSM8K numeric tasks: <VALUE> is a NUMBER (e.g., 42 or 3.5)")
                out.append("- For AQuA MCQ tasks: <VALUE> is a LETTER string among A,B,C,D,E")
            else:  # final_only
                out.append("IMPORTANT: Return ONLY JSON with the final answer. Do not include steps or extra text.")
                out.append("Output format (JSON only, no extra text or markdown):")
                out.append('{"answer": <VALUE>}')
                out.append("- For GSM8K numeric tasks: <VALUE> is a NUMBER (e.g., 42 or 3.5)")
                out.append("- For AQuA MCQ tasks: <VALUE> is a LETTER string among A,B,C,D,E")
            return "\n".join(out)

        system_prompt = _sanitize_instruction_for_eval(raw_instr)
        source = scenario.get("source", "gsm8k").lower()
        return system_prompt, source

    def _get_user_prompt(self, source: str, is_retry: bool = False) -> str:
        """Gets the initial or retry user prompt for the math task."""
        if source == "aqua":
            if self.answer_mode == "final_only":
                return '{"answer": "<LETTER>"}' if is_retry else 'Provide ONLY this JSON: {"answer": "<LETTER>"}'
            # steps_json mode
            if not is_retry:
                return (
                    "Solve the problem step-by-step. Your final step must be 'The final answer is <LETTER>'. "
                    "Then, provide ONLY the following JSON, copying the final answer into the 'answer' field: "
                    '{"steps": ["<step 1>", "...", "The final answer is <LETTER>"], "answer": "<LETTER>"}'
                )
            else:
                return (
                    "You did not provide the answer in the correct format. Finish now. "
                    "Your final step MUST be 'The final answer is <LETTER>'. "
                    "Return ONLY this JSON, copying the answer: "
                    '{"steps": ["The final answer is <LETTER>"], "answer": "<LETTER>"}'
                )
        else: # gsm8k
            if self.answer_mode == "final_only":
                return '{"answer": <NUMBER>}' if is_retry else 'Provide ONLY this JSON: {"answer": <NUMBER>}'
            # steps_json mode
            if not is_retry:
                return (
                    "Solve the problem step-by-step. Your final step must be 'The final answer is <NUMBER>'. "
                    "Then, provide ONLY the following JSON, copying the final answer into the 'answer' field: "
                    '{"steps": ["<step 1>", "...", "The final answer is <NUMBER>"], "answer": <NUMBER>}'
                )
            else:
                return (
                    "You did not provide the answer in the correct format. Finish now. "
                    "Your final step MUST be 'The final answer is <NUMBER>'. "
                    "Return ONLY this JSON, copying the answer: "
                    '{"steps": ["The final answer is <NUMBER>"], "answer": <NUMBER>}'
                )

    async def solve_math_problem_async(
        self, session: aiohttp.ClientSession, scenario: Dict[str, Any], semaphore: asyncio.Semaphore
    ) -> MathResult:
        """Asynchronously solves a math problem by calling the vLLM server."""
        async with semaphore:
            system_prompt, source = self._prepare_prompt_for_scenario(scenario)
            
            response_text = ""
            max_attempts = 3
            for attempt in range(max_attempts):
                user_prompt = self._get_user_prompt(source, is_retry=(attempt > 0))
                
                payload = {
                    "model": self.served_model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024,
                }
                
                try:
                    async with session.post(self.api_url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            response_text = data['choices'][0]['message']['content']
                        else:
                            response_text = f"Error: HTTP {response.status}"
                            continue # Try again
                except Exception as e:
                    response_text = f"Error: {e}"
                    await asyncio.sleep(1) # Wait before retrying on connection error
                    continue

                # Check if an explicit final answer line is present
                txt = response_text.strip()
                if source == "aqua":
                    has_final = bool(re.search(r'\{\s*"answer"\s*:\s*"[A-E]"', txt, flags=re.IGNORECASE))
                else:
                    has_final = bool(re.search(r'\{\s*"answer"\s*:\s*[+-]?\d', txt))
                
                if has_final:
                    break
            
            # Parse the response and calculate results
            extracted_answer, reasoning_steps = None, []
            try:
                # Find the JSON block
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        ans = parsed.get("answer")
                        st = parsed.get("steps")
                        if isinstance(st, list):
                            reasoning_steps = [str(s) for s in st]
                        extracted_answer = ans
            except Exception:
                pass
            
            if extracted_answer is None:
                extracted_answer = self._extract_final_answer(response_text)
                reasoning_steps = self._extract_reasoning_steps(response_text)
            
            expected_answer = scenario["math_ground_truth"]["expected_answer"]
            answer_accuracy = self._calculate_answer_accuracy(extracted_answer, expected_answer, source)
            reasoning_clarity = self._calculate_reasoning_clarity(response_text, scenario.get("barrier_type", "baseline"))

            result = MathResult(
                problem_id=scenario["episode_id"],
                source=source,
                barrier_type=scenario.get("barrier_type") or "baseline",
                agent_response=response_text,
                extracted_answer=extracted_answer,
                expected_answer=expected_answer,
                reasoning_steps=reasoning_steps,
                answer_accuracy=answer_accuracy,
                reasoning_clarity=reasoning_clarity,
                success=(answer_accuracy > 0.8 and reasoning_clarity > 0.6)
            )
            # Incremental persistence right after result is ready
            self._write_incremental(result)
            return result

    async def run_evaluation_by_profiles_async(self, per_profile_questions: int = 200, num_profiles: int = 0, dataset: str = "all") -> Dict[str, Any]:
        print("\n🧮 Loading problems (GSM8K + AQuA) ...")
        problems = self.load_math_problems(limit=0, dataset=dataset)
        gsm8k = [p for p in problems if p.source == "gsm8k"]
        aqua = [p for p in problems if p.source == "aqua"]
        print(f"   GSM8K: {len(gsm8k)} | AQuA: {len(aqua)}")

        profiles = self.load_profiles_from_episodes()
        print(f"👤 Loaded {len(profiles)} agent A profiles from episodes")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for p in profiles:
            bt = p.get("barrier_type") or "baseline"
            grouped.setdefault(bt, []).append(p)

        selected_profiles: List[Dict[str, Any]] = []
        rng = random.Random(42)
        for bt, lst in grouped.items():
            if isinstance(num_profiles, int) and num_profiles > 0:
                take = min(num_profiles, len(lst))
                chosen = rng.sample(lst, take)
            else:
                chosen = lst
            selected_profiles.extend(chosen)
            print(f"   • {bt}: using {len(chosen)}/{len(lst)} profiles")
        
        rng.shuffle(selected_profiles)

        # Create all scenario permutations
        all_scenarios = []
        for idx, prof in enumerate(selected_profiles):
            rng_prob = random.Random(42 + idx)
            gsm_sample = rng_prob.sample(gsm8k, min(per_profile_questions, len(gsm8k))) if gsm8k else []
            aqua_sample = rng_prob.sample(aqua, min(per_profile_questions, len(aqua))) if aqua else []
            for prob in gsm_sample + aqua_sample:
                all_scenarios.append(self.create_math_scenario_from_profile(prof, prob))
        
        print(f"\n🚀 Launching {len(all_scenarios)} total evaluation tasks with concurrency={self.concurrency}...")

        # Run all scenarios concurrently
        all_results: List[MathResult] = []
        semaphore = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [self.solve_math_problem_async(session, sc, semaphore) for sc in all_scenarios]
            
            # Use tqdm for progress bar
            all_results = await tqdm.gather(*tasks, desc="Solving Math Problems")
        
        print("\n📊 All tasks completed. Aggregating results...")

        # Filter out potential None results from failed tasks if any
        all_results = [r for r in all_results if r is not None]

        # Per-profile averages (combined across sources)
        per_profile: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            pid = r.problem_id.split("_")[0]  # original episode id prefix
            key = pid
            if key not in per_profile:
                per_profile[key] = {"answers": [], "clarities": [], "barrier": None}
            per_profile[key]["answers"].append(r.answer_accuracy)
            per_profile[key]["clarities"].append(r.reasoning_clarity)
            # barrier type can be derived from r.barrier_type
            per_profile[key]["barrier"] = r.barrier_type or "baseline"

        profile_averages = {}
        for k, v in per_profile.items():
            profile_averages[k] = {
                "mean_answer_accuracy": float(np.mean(v["answers"])) if v["answers"] else 0.0,
                "mean_reasoning_clarity": float(np.mean(v["clarities"])) if v["clarities"] else 0.0,
                "barrier_type": v["barrier"],
                "n": len(v["answers"]),
            }

        # Aggregate by barrier type (combined)
        by_barrier_group: Dict[str, List[float]] = {}
        for k, v in profile_averages.items():
            bt = v["barrier_type"] or "baseline"
            by_barrier_group.setdefault(bt, []).append(v["mean_answer_accuracy"])
        barrier_type_avgs = {bt: float(np.mean(scores)) for bt, scores in by_barrier_group.items() if scores}

        # Per-profile averages separated by source
        per_profile_by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in all_results:
            pid = r.problem_id.split("_")[0]
            src = r.source
            per_profile_by_source.setdefault(src, {})
            if pid not in per_profile_by_source[src]:
                per_profile_by_source[src][pid] = {"answers": [], "clarities": [], "barrier": r.barrier_type or "baseline"}
            per_profile_by_source[src][pid]["answers"].append(r.answer_accuracy)
            per_profile_by_source[src][pid]["clarities"].append(r.reasoning_clarity)

        profile_averages_by_source: Dict[str, Dict[str, Any]] = {}
        for src, prof_map in per_profile_by_source.items():
            profile_averages_by_source[src] = {}
            for pid, vals in prof_map.items():
                profile_averages_by_source[src][pid] = {
                    "mean_answer_accuracy": float(np.mean(vals["answers"])) if vals["answers"] else 0.0,
                    "mean_reasoning_clarity": float(np.mean(vals["clarities"])) if vals["clarities"] else 0.0,
                    "barrier_type": vals["barrier"],
                    "n": len(vals["answers"]),
                }

        # Barrier-type averages separated by source
        barrier_type_averages_by_source: Dict[str, Dict[str, float]] = {}
        for src, prof_map in profile_averages_by_source.items():
            bt_group: Dict[str, List[float]] = {}
            for pid, row in prof_map.items():
                bt = row["barrier_type"] or "baseline"
                bt_group.setdefault(bt, []).append(row["mean_answer_accuracy"])
            barrier_type_averages_by_source[src] = {bt: float(np.mean(scores)) for bt, scores in bt_group.items() if scores}

        # Persist
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "profile_averages.json", 'w') as f:
            json.dump(profile_averages, f, indent=2)
        with open(out_dir / "barrier_type_averages.json", 'w') as f:
            json.dump(barrier_type_avgs, f, indent=2)

        # Save source-separated summaries
        with open(out_dir / "profile_averages_by_source.json", 'w') as f:
            json.dump(profile_averages_by_source, f, indent=2)
        with open(out_dir / "barrier_type_averages_by_source.json", 'w') as f:
            json.dump(barrier_type_averages_by_source, f, indent=2)

        # Also write a CSV summary
        csv_path = out_dir / "profile_scores.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["profile_id", "barrier_type", "mean_answer_accuracy", "mean_reasoning_clarity", "n"])
            for pid, row in profile_averages.items():
                w.writerow([pid, row["barrier_type"], f"{row['mean_answer_accuracy']:.4f}", f"{row['mean_reasoning_clarity']:.4f}", row["n"]])

        return {
            "detailed_results": all_results,
            "profile_averages": profile_averages,
            "barrier_type_averages": barrier_type_avgs,
            "profile_averages_by_source": profile_averages_by_source,
            "barrier_type_averages_by_source": barrier_type_averages_by_source,
        }
    
    def _write_incremental(self, result: MathResult) -> None:
        """Append a single result as a JSON line for immediate persistence."""
        try:
            rec = {
                "problem_id": result.problem_id,
                "source": result.source,
                "barrier_type": result.barrier_type,
                "agent_response": result.agent_response,
                "extracted_answer": result.extracted_answer,
                "expected_answer": result.expected_answer,
                "reasoning_steps": result.reasoning_steps,
                "answer_accuracy": result.answer_accuracy,
                "reasoning_clarity": result.reasoning_clarity,
                "success": result.success,
            }
            with open(self._incremental_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Do not crash the run on I/O errors
            pass
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from agent response"""
        if isinstance(response, dict):
            if "argument" in response:
                return str(response["argument"])
            elif "content" in response:
                return str(response["content"])
        return str(response)
    
    def _extract_final_answer(self, text: str) -> Any:
        """Extract final answer from response.
        Returns float for numeric tasks, or a single-letter string (A-E) for MCQ.
        """
        # Try MCQ letter first (prefer explicit 'Answer: <LETTER>' line)
        letter_patterns = [
            r"^\s*answer[:\s]+([A-E])\b",
            r"^\s*final answer[:\s]+([A-E])\b",
            r"^\s*selected[:\s]+([A-E])\b",
            r"^\s*choice[:\s]+([A-E])\b",
        ]
        text_stripped = text.strip()
        for pattern in letter_patterns:
            m = re.search(pattern, text_stripped, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()
        # Looser MCQ fallback (e.g., "...correct answer is (D)")
        loose_letter_patterns = [
            r"(?:correct answer|answer is|is)\s*\(?([A-E])\)?\b",
        ]
        for pattern in loose_letter_patterns:
            m = re.search(pattern, text_stripped, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # Helpers for numeric extraction
        def _extract_number_from_string(s: str) -> Optional[float]:
            tokens = re.findall(r"([+-]?\d[\d,]*(?:\.\d+)?)", s)
            for tok in reversed(tokens):
                try:
                    return float(tok.replace(',', ''))
                except Exception:
                    continue
            return None

        # 1) Prefer explicit final answer line
        m = re.search(r"(?mi)^\s*(?:final answer|answer)[:\s]*(.+)$", text_stripped)
        if m:
            num = _extract_number_from_string(m.group(1))
            if num is not None:
                return num

        # 2) Anchored numeric line variants
        anchored_patterns = [
            r"(?mi)^\s*(?:final answer|answer|equals|result is)[:\s]*([^\n]+)$",
        ]
        for pattern in anchored_patterns:
            m2 = re.search(pattern, text_stripped)
            if m2:
                num = _extract_number_from_string(m2.group(1))
                if num is not None:
                    return num

        # 3) Inline fallback near phrases
        inline_patterns = [
            r"(?i)(?:final answer|answer|equals|result)[^\n]{0,80}?([+-]?\d[\d,]*(?:\.\d+)?)",
        ]
        for pattern in inline_patterns:
            it = list(re.finditer(pattern, text_stripped))
            if it:
                try:
                    return float(it[-1].group(1).replace(',', ''))
                except Exception:
                    pass

        # 4) Last-resort: last numeric token in the whole text
        num = _extract_number_from_string(text_stripped)
        if num is not None:
            return num

        # Parse failure -> default (treated as incorrect later)
        return 0.0
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from response"""
        # Split by common step indicators
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and any(indicator in line.lower() for indicator in 
                          ['step', 'first', 'next', 'then', 'finally', 'so']):
                steps.append(line)
        
        return steps
    
    def _calculate_answer_accuracy(self, extracted: Any, expected: Any, source: str) -> float:
   
        # AQuA MCQ: exact letter match
        if isinstance(expected, str) or source == "aqua":
            if isinstance(extracted, str) and extracted.upper() == str(expected).upper():
                return 1.0
            return 0.0
        # GSM8K numeric: strict match
        try:
            exp = float(expected)
            ext = float(extracted)
        except Exception:
            return 0.0
        return 1.0 if abs(ext - exp) <= 1e-6 else 0.0
    
    def _calculate_reasoning_clarity(self, text: str, barrier_type: str) -> float:
        """Calculate clarity of reasoning (should be lower for barriers affecting communication)"""
        
        # Baseline clarity metrics
        base_score = 1.0
        
        # Length factor (extremely long = unclear)
        if len(text) > 1000:
            base_score *= 0.8
        elif len(text) < 50:
            base_score *= 0.6  # Too short = unclear
        
        # Step presence (good reasoning should have steps)
        step_indicators = ['step', 'first', 'next', 'then', 'because', 'so']
        step_count = sum(1 for indicator in step_indicators if indicator in text.lower())
        if step_count >= 2:
            base_score *= 1.0
        elif step_count == 1:
            base_score *= 0.8
        else:
            base_score *= 0.6
        
        # Barrier-specific clarity reductions
        if barrier_type == "semantic_structure":
            # Check for ambiguity indicators
            ambiguous_words = ['roughly', 'approximately', 'kind of', 'somewhat']
            ambiguity_count = sum(1 for word in ambiguous_words if word in text.lower())
            base_score *= max(0.3, 1.0 - 0.1 * ambiguity_count)
            
        elif barrier_type == "cultural_style":
            # Check for hedging
            hedges = ['perhaps', 'might', 'could be', 'possibly', 'maybe']
            hedge_count = sum(1 for hedge in hedges if hedge in text.lower())
            base_score *= max(0.4, 1.0 - 0.1 * hedge_count)
            
        elif barrier_type == "emotional_influence":
            # Check for emotional language
            exclamation_count = text.count('!')
            base_score *= max(0.5, 1.0 - 0.05 * exclamation_count)
        
        return max(0.0, min(1.0, base_score))
    
    def _save_detailed_results(self, results: List[MathResult]) -> None:
        """Save detailed results to JSON and CSV for per-template comparisons"""
        
        results_data = []
        for result in results:
            results_data.append({
                "problem_id": result.problem_id,
                "source": result.source,
                "barrier_type": result.barrier_type,
                "agent_response": result.agent_response,
                "extracted_answer": result.extracted_answer,
                "expected_answer": result.expected_answer,
                "reasoning_steps": result.reasoning_steps,
                "answer_accuracy": result.answer_accuracy,
                "reasoning_clarity": result.reasoning_clarity,
                "success": result.success
            })
        
        with open(f"{self.output_dir}/detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save a flat CSV for easy comparison of four templates (baseline + 3 barriers)
        csv_path = Path(self.output_dir) / "results_by_problem.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["problem_id", "source", "barrier_type", "extracted_answer", "expected_answer", "answer_accuracy", "reasoning_clarity", "success"])
            for r in results:
                writer.writerow([r.problem_id, r.source, r.barrier_type, r.extracted_answer, r.expected_answer, f"{r.answer_accuracy:.4f}", f"{r.reasoning_clarity:.4f}", int(r.success)])
    
    def _compute_statistics(self, results: List[MathResult]) -> Dict[str, Any]:
        """Compute statistical analysis of barrier effects (overall and per source)"""
        
        # Group by barrier type
        by_barrier = {}
        by_source_barrier = {}
        for result in results:
            barrier = result.barrier_type
            if barrier not in by_barrier:
                by_barrier[barrier] = {
                    "answer_accuracies": [],
                    "reasoning_clarities": [],
                    "successes": []
                }
            
            by_barrier[barrier]["answer_accuracies"].append(result.answer_accuracy)
            by_barrier[barrier]["reasoning_clarities"].append(result.reasoning_clarity)
            by_barrier[barrier]["successes"].append(result.success)

            # Per-source grouping
            src = result.source
            by_source_barrier.setdefault(src, {})
            if barrier not in by_source_barrier[src]:
                by_source_barrier[src][barrier] = {
                    "answer_accuracies": [],
                    "reasoning_clarities": [],
                    "successes": []
                }
            by_source_barrier[src][barrier]["answer_accuracies"].append(result.answer_accuracy)
            by_source_barrier[src][barrier]["reasoning_clarities"].append(result.reasoning_clarity)
            by_source_barrier[src][barrier]["successes"].append(result.success)
        
        # Compute means and statistical tests
        stats_summary = {}
        baseline_answer = by_barrier.get("baseline", {}).get("answer_accuracies", [])
        baseline_clarity = by_barrier.get("baseline", {}).get("reasoning_clarities", [])
        
        for barrier_type, data in by_barrier.items():
            answer_scores = data["answer_accuracies"]
            clarity_scores = data["reasoning_clarities"]
            
            stats_summary[barrier_type] = {
                "answer_accuracy_mean": float(np.mean(answer_scores)),
                "answer_accuracy_std": float(np.std(answer_scores)),
                "reasoning_clarity_mean": float(np.mean(clarity_scores)),
                "reasoning_clarity_std": float(np.std(clarity_scores)),
                "success_rate": float(np.mean(data["successes"]))
            }
            
            # Statistical tests vs baseline
            if barrier_type != "baseline" and baseline_answer and baseline_clarity:
                answer_tstat, answer_pval = stats.ttest_rel(
                    baseline_answer[:len(answer_scores)], answer_scores
                )
                clarity_tstat, clarity_pval = stats.ttest_rel(
                    baseline_clarity[:len(clarity_scores)], clarity_scores
                )
                
                stats_summary[barrier_type].update({
                    "answer_vs_baseline_pvalue": float(answer_pval),
                    "clarity_vs_baseline_pvalue": float(clarity_pval),
                    "answer_delta_mean": float(np.mean(answer_scores) - np.mean(baseline_answer[:len(answer_scores)])),
                    "clarity_delta_mean": float(np.mean(clarity_scores) - np.mean(baseline_clarity[:len(clarity_scores)]))
                })
        
        # Per-source stats
        stats_by_source: Dict[str, Any] = {}
        for src, m in by_source_barrier.items():
            baseline_answer_s = m.get("baseline", {}).get("answer_accuracies", [])
            baseline_clarity_s = m.get("baseline", {}).get("reasoning_clarities", [])
            stats_by_source[src] = {}
            for barrier_type, data in m.items():
                ans = data["answer_accuracies"]
                clr = data["reasoning_clarities"]
                entry = {
                    "answer_accuracy_mean": float(np.mean(ans) if ans else 0.0),
                    "answer_accuracy_std": float(np.std(ans) if ans else 0.0),
                    "reasoning_clarity_mean": float(np.mean(clr) if clr else 0.0),
                    "reasoning_clarity_std": float(np.std(clr) if clr else 0.0),
                    "success_rate": float(np.mean(data["successes"])) if data["successes"] else 0.0,
                }
                if barrier_type != "baseline" and baseline_answer_s and baseline_clarity_s:
                    try:
                        a_t, a_p = stats.ttest_rel(baseline_answer_s[:len(ans)], ans)
                        c_t, c_p = stats.ttest_rel(baseline_clarity_s[:len(clr)], clr)
                        entry.update({
                            "answer_vs_baseline_pvalue": float(a_p),
                            "clarity_vs_baseline_pvalue": float(c_p),
                            "answer_delta_mean": float(np.mean(ans) - np.mean(baseline_answer_s[:len(ans)])),
                            "clarity_delta_mean": float(np.mean(clr) - np.mean(baseline_clarity_s[:len(clr)])),
                        })
                    except Exception:
                        pass
                stats_by_source[src][barrier_type] = entry

        # Save per-source stats alongside overall
        out = {
            "overall": stats_summary,
            "by_source": stats_by_source,
        }
        with open(f"{self.output_dir}/evaluation_by_source.json", 'w') as f:
            json.dump(out, f, indent=2)

        return stats_summary
  
def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate barrier effects on single-agent mathematical reasoning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to evaluate")
    parser.add_argument("--per_profile_questions", type=int, default=20, help="Questions per dataset per profile (applied to both GSM8K and AQuA)")
    parser.add_argument("--num_profiles", type=int, default=0, help="Max profiles per barrier type (0 = use all)")
    parser.add_argument("--severity", type=float, default=0.8, help="Barrier severity")
    parser.add_argument("--output_dir", type=str, default="IQ_test/results", help="Output directory")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "gsm8k", "aqua"], help="Dataset to evaluate on")
    parser.add_argument("--concurrency", type=int, default=16, help="Number of parallel requests to the model server")
    parser.add_argument("--answer_mode", type=str, default="steps_json", choices=["steps_json", "final_only"], help="Answer format: 'steps_json' (current default) or 'final_only' to isolate barrier impact")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SingleAgentMathEvaluator(
        model_name=args.model,
        output_dir=args.output_dir,
        severity=args.severity,
        concurrency=args.concurrency,
        answer_mode=args.answer_mode,
    )
    
    # Always evaluate with real Agent A profiles loaded from episodes
    results = asyncio.run(evaluator.run_evaluation_by_profiles_async(
        per_profile_questions=args.per_profile_questions, 
        num_profiles=args.num_profiles,
        dataset=args.dataset
    ))
    
    # Save results
    with open(f"{args.output_dir}/evaluation_summary.json", 'w') as f:
        json.dump({
            "profile_averages": results["profile_averages"],
            "barrier_type_averages": results["barrier_type_averages"],
            "profile_averages_by_source": results["profile_averages_by_source"],
            "barrier_type_averages_by_source": results["barrier_type_averages_by_source"],
        }, f, indent=2)
    

    print(f"\n💾 Detailed results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()