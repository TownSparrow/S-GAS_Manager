import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Turn:
    """ One turn step per dialogue in scenario """
    query: str
    ground_truth: List[str]  # Ground truth chunk references for retrieval evaluation
    ground_truth_answer: str = ""  # Expected answer for generation evaluation (BERTScore)
    ground_truth_text: str = ""   # Distinctive key phrase that must appear in a retrieved chunk
    ground_truth_texts: Optional[List[str]] = None  # Chunk-ID-independent evidence snippets
    expected_entities: Optional[List[str]] = None


@dataclass
class BenchmarkScenario:
    """ Full test scenario """
    name: str
    document: str
    description: str
    baseline_mode: bool
    turns: List[Turn]
    # Optional per-scenario chunk granularity (overrides system config during upload).
    # Set to a small value (e.g. 40) when the document has short labeled paragraphs.
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class ScenarioLoader:
    """ Loader scenarios from prepared JSON-files """
    
    def __init__(self, scenarios_dir: str = "tests/scenarios"):
        self.scenarios_dir = Path(scenarios_dir)
        if not self.scenarios_dir.exists():
            self.scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    def load_scenario(self, scenario_name: str) -> BenchmarkScenario:
        """ Load by name """
        scenario_file = self.scenarios_dir / f"{scenario_name}.json"
        
        if not scenario_file.exists():
            raise FileNotFoundError(f"❌ Scenario {scenario_name} is not found in {self.scenarios_dir}")
        
        with open(scenario_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        turns = [
            Turn(
                query=turn['query'],
                # Support both 'ground_truth' and 'ground_truth_chunks' keys
                ground_truth=turn.get('ground_truth_chunks') or turn.get('ground_truth', []),
                ground_truth_answer=turn.get('ground_truth_answer', ''),
                ground_truth_text=turn.get('ground_truth_text', ''),
                ground_truth_texts=turn.get('ground_truth_texts'),
                expected_entities=turn.get('expected_entities'),
            )
            for turn in data['turns']
        ]
        
        return BenchmarkScenario(
            name=data['name'],
            document=data['document'],
            description=data.get('description', ''),
            baseline_mode=data.get('baseline_mode', False),
            turns=turns,
            chunk_size=data.get('chunk_size'),
            chunk_overlap=data.get('chunk_overlap'),
        )
    
    def list_available_scenarios(self) -> List[str]:
        """ List of available scenarios """
        return [
            f.stem for f in self.scenarios_dir.glob("*.json")
        ]
    
    def create_example_scenario(self):
        """C reate example scenario for demo """
        
        # TODO: Make a better test scenario with test document
        example = {
            "name": "test_scenario",
            "description": "Test scenario example",
            "document": "test_scenario.txt",
            "baseline_mode": False,
            "turns": [
                {
                    "query": "This is some kind of query for test scenario. Is this scenario, is there any text?",
                    "ground_truth": ["chunk_ref_1", "chunk_ref_2"],
                    "ground_truth_answer": "Yes, this is a test scenario containing example text for benchmarking.",
                    "expected_entities": ["Scenario", "Test"]
                }
            ]
        }
        
        example_file = self.scenarios_dir / "test_scenario_example.json"
        with open(example_file, 'w', encoding='utf-8') as f:
            json.dump(example, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Example of scenario is created: {example_file}")
