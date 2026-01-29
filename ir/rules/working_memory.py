"""Working memory for rule-based systems.

This module provides the WorkingMemory class for managing facts
in the inference engine.
"""

from collections import defaultdict
from typing import Dict, Set, List, Iterator, Callable, Optional, Any
from .fact import Fact, FactTemplate


class WorkingMemory:
    """Working memory for the inference engine.
    
    Working memory stores facts and provides operations for
    asserting, retracting, and modifying facts. It also supports
    listeners for fact change notifications.
    """
    
    def __init__(self):
        """Initialize working memory."""
        self._facts: Dict[int, Fact] = {}  # fact_id -> Fact
        self._next_id: int = 1
        self._templates: Dict[str, FactTemplate] = {}
        self._indices: Dict[str, Set[int]] = defaultdict(set)  # template -> fact_ids
        self._listeners: List[Callable[[str, Fact], None]] = []
    
    def assert_fact(self, fact: Fact) -> int:
        """Assert a new fact into working memory.
        
        Args:
            fact: The fact to assert
            
        Returns:
            The fact ID (existing if duplicate, new otherwise)
        """
        # Check for duplicates
        for existing_id, existing in self._facts.items():
            if self._facts_equal(fact, existing):
                return existing_id
        
        fact.fact_id = self._next_id
        self._facts[self._next_id] = fact
        self._next_id += 1
        
        # Update index
        template = fact.template_name or "__ordered__"
        self._indices[template].add(fact.fact_id)
        
        # Notify listeners
        self._notify("assert", fact)
        
        return fact.fact_id
    
    def retract_fact(self, fact_id: int) -> bool:
        """Retract a fact from working memory.
        
        Args:
            fact_id: ID of the fact to retract
            
        Returns:
            True if fact was retracted, False if not found
        """
        if fact_id not in self._facts:
            return False
        
        fact = self._facts[fact_id]
        template = fact.template_name or "__ordered__"
        self._indices[template].discard(fact_id)
        del self._facts[fact_id]
        
        # Notify listeners
        self._notify("retract", fact)
        
        return True
    
    def modify_fact(self, fact_id: int, modifications: Dict[str, Any]) -> bool:
        """Modify a fact's slot values.
        
        Args:
            fact_id: ID of the fact to modify
            modifications: Dict of slot_name -> new_value
            
        Returns:
            True if fact was modified, False if not found
        """
        if fact_id not in self._facts:
            return False
        
        fact = self._facts[fact_id]
        
        for slot, value in modifications.items():
            fact.slots[slot] = value
        
        self._notify("modify", fact)
        return True
    
    def get_fact(self, fact_id: int) -> Optional[Fact]:
        """Get a fact by ID.
        
        Args:
            fact_id: ID of the fact
            
        Returns:
            The fact or None if not found
        """
        return self._facts.get(fact_id)
    
    def get_facts_by_template(self, template_name: str) -> Iterator[Fact]:
        """Get all facts matching a template.
        
        Args:
            template_name: Name of the template
            
        Yields:
            Facts matching the template
        """
        for fact_id in self._indices.get(template_name, set()):
            yield self._facts[fact_id]
    
    def get_ordered_facts(self) -> Iterator[Fact]:
        """Get all ordered facts (no template).
        
        Yields:
            All ordered facts
        """
        for fact_id in self._indices.get("__ordered__", set()):
            yield self._facts[fact_id]
    
    def get_all_facts(self) -> Iterator[Fact]:
        """Get all facts in working memory.
        
        Yields:
            All facts
        """
        return iter(self._facts.values())
    
    def register_template(self, template: FactTemplate) -> None:
        """Register a fact template.
        
        Args:
            template: The template to register
        """
        self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[FactTemplate]:
        """Get a registered template by name.
        
        Args:
            name: Template name
            
        Returns:
            The template or None if not found
        """
        return self._templates.get(name)
    
    def clear(self) -> None:
        """Clear all facts from working memory."""
        self._facts.clear()
        self._indices.clear()
        self._next_id = 1
    
    def add_listener(self, listener: Callable[[str, Fact], None]) -> None:
        """Add a listener for fact changes.
        
        Args:
            listener: Callback function(event_type, fact)
        """
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[str, Fact], None]) -> bool:
        """Remove a listener.
        
        Args:
            listener: The listener to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            self._listeners.remove(listener)
            return True
        except ValueError:
            return False
    
    def _notify(self, event: str, fact: Fact) -> None:
        """Notify listeners of fact changes.
        
        Args:
            event: Event type (assert, retract, modify)
            fact: The affected fact
        """
        for listener in self._listeners:
            listener(event, fact)
    
    def _facts_equal(self, f1: Fact, f2: Fact) -> bool:
        """Check if two facts are equal.
        
        Args:
            f1: First fact
            f2: Second fact
            
        Returns:
            True if facts are equal
        """
        if f1.template_name != f2.template_name:
            return False
        if f1.slots != f2.slots:
            return False
        if f1.values != f2.values:
            return False
        return True
    
    def __len__(self) -> int:
        """Return number of facts in working memory."""
        return len(self._facts)
    
    def __contains__(self, fact_id: int) -> bool:
        """Check if a fact ID exists."""
        return fact_id in self._facts
    
    def __iter__(self) -> Iterator[Fact]:
        """Iterate over all facts."""
        return iter(self._facts.values())
