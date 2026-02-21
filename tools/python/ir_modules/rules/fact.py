"""Fact definitions for rule-based systems.

This module provides Fact and FactTemplate classes for
representing knowledge in working memory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json


@dataclass
class FactTemplate:
    """Template for structured facts (deftemplate in CLIPS).
    
    Attributes:
        name: Template name
        slots: List of (slot_name, slot_type) tuples for single-value slots
        multislots: List of (slot_name, slot_type) tuples for multi-value slots
        default_values: Default values for slots
        documentation: Optional documentation string
    """
    name: str
    slots: List[Tuple[str, str]]  # (slot_name, slot_type) pairs
    multislots: List[Tuple[str, str]] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)
    documentation: Optional[str] = None
    
    def validate_fact(self, fact: 'Fact') -> bool:
        """Validate that a fact conforms to this template.
        
        Args:
            fact: The fact to validate
            
        Returns:
            True if fact is valid for this template
        """
        if fact.template_name != self.name:
            return False
        slot_names = {s[0] for s in self.slots}
        multislot_names = {s[0] for s in self.multislots}
        all_slot_names = slot_names | multislot_names
        for key in fact.slots.keys():
            if key not in all_slot_names:
                return False
        return True
    
    def get_slot_type(self, slot_name: str) -> Optional[str]:
        """Get the type of a slot.
        
        Args:
            slot_name: Name of the slot
            
        Returns:
            The slot type or None if not found
        """
        for name, stype in self.slots:
            if name == slot_name:
                return stype
        for name, stype in self.multislots:
            if name == slot_name:
                return stype
        return None
    
    def create_fact(self, **slot_values) -> 'Fact':
        """Create a fact from this template with given slot values.
        
        Args:
            **slot_values: Slot name/value pairs
            
        Returns:
            A new Fact instance
        """
        # Apply defaults
        all_slots = dict(self.default_values)
        all_slots.update(slot_values)
        return Fact(template_name=self.name, slots=all_slots)


@dataclass
class Fact:
    """A fact in working memory.
    
    Facts can be either:
    - Ordered facts: (value1 value2 value3)
    - Template facts: (template-name (slot1 value1) (slot2 value2))
    
    Attributes:
        fact_id: Unique identifier assigned by working memory
        template_name: Name of the template (None for ordered facts)
        slots: Slot values for template facts
        values: Values for ordered facts
    """
    fact_id: Optional[int] = None
    template_name: Optional[str] = None  # None for ordered facts
    slots: Dict[str, Any] = field(default_factory=dict)  # For template facts
    values: Tuple[Any, ...] = field(default_factory=tuple)  # For ordered facts
    
    def is_ordered(self) -> bool:
        """Check if this is an ordered fact (no template)."""
        return self.template_name is None
    
    def is_template(self) -> bool:
        """Check if this is a template fact."""
        return self.template_name is not None
    
    def get_hash(self) -> str:
        """Get deterministic hash for this fact.
        
        Returns:
            16-character hex hash string
        """
        data = {
            'template': self.template_name,
            'slots': dict(sorted(self.slots.items())),
            'values': list(self.values)
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a slot value or ordered value by index.
        
        Args:
            key: Slot name for template facts, or index as string for ordered
            default: Default value if not found
            
        Returns:
            The value or default
        """
        if self.is_template():
            return self.slots.get(key, default)
        else:
            try:
                idx = int(key)
                return self.values[idx] if 0 <= idx < len(self.values) else default
            except (ValueError, IndexError):
                return default
    
    def matches_template(self, template_name: Optional[str]) -> bool:
        """Check if fact matches a template name.
        
        Args:
            template_name: Template name to match (None matches ordered facts)
            
        Returns:
            True if fact matches the template
        """
        return self.template_name == template_name
    
    def __str__(self) -> str:
        """String representation of the fact."""
        if self.is_ordered():
            values_str = " ".join(str(v) for v in self.values)
            return f"f-{self.fact_id or '?'}: ({values_str})"
        else:
            slots_str = " ".join(
                f"({k} {v})" for k, v in self.slots.items()
            )
            return f"f-{self.fact_id or '?'}: ({self.template_name} {slots_str})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on content (not fact_id)."""
        if not isinstance(other, Fact):
            return False
        return (
            self.template_name == other.template_name and
            self.slots == other.slots and
            self.values == other.values
        )
    
    def __hash__(self) -> int:
        """Hash based on content."""
        return hash((self.template_name, tuple(sorted(self.slots.items())), self.values))
