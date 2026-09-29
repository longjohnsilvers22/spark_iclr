"""
SPARK BT Fragment Library: Voyager-style skill accumulation.

Stores successful behavior tree fragments (YAML scores) indexed by:
- Task description (natural language)
- Object types involved (pick, place)
- Action patterns (pick-place, open-drawer, push, turn-knob)

On new tasks, retrieves relevant fragments and provides them as examples
to the LLM planner, enabling few-shot composition of known skills.

Usage:
    library = BTLibrary()

    # After successful execution:
    library.add(instruction="pick up the red mug and place it on the plate",
                score={"tree": {...}},
                objects=["red mug", "plate"],
                success=True)

    # Before planning a new task:
    examples = library.retrieve("pick up the blue cup and place it on the tray", k=3)
    # Pass examples to Gemini as few-shot context
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


LIBRARY_DIR = os.environ.get('SPARK_BT_LIBRARY',
    os.path.expanduser('~/spark/bt_library'))


@dataclass
class BTFragment:
    """A stored behavior tree fragment."""
    instruction: str
    score: dict  # The YAML score (behavior tree)
    objects: list  # Object labels involved
    action_pattern: str  # e.g., "pick-place", "open-drawer", "push", "turn-knob"
    robot: str = "franka"  # Robot used
    scene_type: str = ""  # e.g., "kitchen", "living_room", "tabletop"
    success: bool = True
    timestamp: float = 0.0
    num_uses: int = 0  # Times retrieved and used successfully
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique ID from instruction hash."""
        return hashlib.md5(self.instruction.encode()).hexdigest()[:12]


class BTLibrary:
    """Voyager-style BT fragment library with retrieval."""

    def __init__(self, library_dir: str = None):
        self.library_dir = Path(library_dir or LIBRARY_DIR)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._fragments: dict[str, BTFragment] = {}
        self._load()

    def _load(self):
        """Load all fragments from disk."""
        index_path = self.library_dir / 'index.json'
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
            for frag_data in data.get('fragments', []):
                frag = BTFragment(**frag_data)
                self._fragments[frag.id] = frag

    def _save(self):
        """Save index to disk."""
        index_path = self.library_dir / 'index.json'
        data = {
            'version': 1,
            'num_fragments': len(self._fragments),
            'fragments': [asdict(f) for f in self._fragments.values()],
        }
        with open(index_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def add(self, instruction: str, score: dict, objects: list = None,
            success: bool = True, robot: str = "franka",
            scene_type: str = "", metadata: dict = None):
        """Add a BT fragment after successful execution."""
        if not success:
            return  # Only store successes

        # Classify action pattern from the score
        pattern = self._classify_pattern(score)

        frag = BTFragment(
            instruction=instruction,
            score=score,
            objects=objects or [],
            action_pattern=pattern,
            robot=robot,
            scene_type=scene_type,
            success=True,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Update or add
        existing = self._fragments.get(frag.id)
        if existing:
            existing.num_uses += 1
            existing.timestamp = time.time()
        else:
            self._fragments[frag.id] = frag

        self._save()

        # Also save the full score YAML
        score_path = self.library_dir / f'{frag.id}.json'
        with open(score_path, 'w') as f:
            json.dump({'instruction': instruction, 'score': score,
                       'pattern': pattern, 'objects': objects}, f, indent=2)

    def retrieve(self, instruction: str, k: int = 3,
                 robot: str = None, pattern: str = None) -> list[BTFragment]:
        """Retrieve the k most relevant fragments for a new instruction.

        Uses keyword overlap scoring (simple but effective for manipulation tasks).
        """
        if not self._fragments:
            return []

        query_words = set(instruction.lower().split())
        scored = []

        for frag in self._fragments.values():
            if robot and frag.robot != robot:
                continue
            if pattern and frag.action_pattern != pattern:
                continue

            # Score by keyword overlap
            frag_words = set(frag.instruction.lower().split())
            overlap = len(query_words & frag_words)
            # Bonus for matching objects
            obj_bonus = sum(1 for obj in frag.objects
                          if any(w in instruction.lower() for w in obj.lower().split()))
            # Bonus for frequently used fragments
            use_bonus = min(frag.num_uses * 0.1, 1.0)
            score = overlap + obj_bonus * 2 + use_bonus
            scored.append((score, frag))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:k]]

    def get_examples_prompt(self, instruction: str, k: int = 3) -> str:
        """Generate a prompt section with retrieved BT examples."""
        examples = self.retrieve(instruction, k=k)
        if not examples:
            return ""

        lines = ["\nHere are examples of successful behavior trees for similar tasks:"]
        for i, frag in enumerate(examples):
            import yaml
            score_yaml = yaml.dump(frag.score, default_flow_style=False)
            lines.append(f"\nExample {i+1}: \"{frag.instruction}\"")
            lines.append(f"```yaml\n{score_yaml}```")
        lines.append("\nUse these as reference patterns. Adapt keypoint labels to match the current detected objects.")
        return "\n".join(lines)

    def _classify_pattern(self, score: dict) -> str:
        """Classify the action pattern of a score."""
        tree = score.get('tree', {})
        actions = []
        def _collect(node):
            # 'retry' and 'fallback' are recovery-grammar composites:
            # recurse them like any other composite, or a wrapped
            # pick-place classifies as its own wrapper type.
            if node.get('type') in ('sequence', 'selector',
                                     'retry', 'fallback'):
                for c in node.get('children', []):
                    _collect(c)
            else:
                actions.append(node.get('type', ''))
        _collect(tree)

        if 'open_drawer' in actions:
            return 'open-drawer'
        if 'turn_knob' in actions:
            return 'turn-knob'
        if 'push_object' in actions:
            return 'push'
        if 'grasp' in actions and 'release' in actions:
            return 'pick-place'
        if 'move_to_keypoint' in actions:
            return 'move'
        return 'unknown'

    @property
    def stats(self) -> dict:
        """Library statistics."""
        patterns = {}
        for f in self._fragments.values():
            patterns[f.action_pattern] = patterns.get(f.action_pattern, 0) + 1
        return {
            'total': len(self._fragments),
            'patterns': patterns,
            'total_uses': sum(f.num_uses for f in self._fragments.values()),
        }

    def __len__(self):
        return len(self._fragments)

    def __repr__(self):
        return f"BTLibrary({len(self)} fragments, {self.stats})"


# Module-level singleton
_library = None

def get_library() -> BTLibrary:
    """Get or create the global BT library."""
    global _library
    if _library is None:
        _library = BTLibrary()
    return _library
