"""
Knowledge Graph for STUNIR Logical Reasoning System.

Stores entities (findings, files, patterns) and their relationships
for cross-method inference and explanation generation.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    FILE = "file"
    FUNCTION = "function"
    FINDING = "finding"
    PATTERN = "pattern"
    RULE = "rule"
    HYPOTHESIS = "hypothesis"
    SYMPTOM = "symptom"
    METRIC = "metric"


class RelationshipType(Enum):
    """Types of relationships between entities."""
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    EXPLAINS = "explains"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    SIMILAR_TO = "similar_to"
    INSTANCE_OF = "instance_of"


@dataclass
class Entity:
    """Represents a node in the knowledge graph."""
    id: str
    type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = ""  # Which analysis method created this
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False


@dataclass
class Relationship:
    """Represents an edge in the knowledge graph."""
    source: Entity
    target: Entity
    type: RelationshipType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source.id, self.target.id, self.type))
    
    def __eq__(self, other):
        if isinstance(other, Relationship):
            return (self.source.id == other.source.id and 
                   self.target.id == other.target.id and
                   self.type == other.type)
        return False


class KnowledgeGraph:
    """
    Graph-based knowledge representation for STUNIR analysis.
    
    Supports:
    - Entity and relationship storage
    - Path finding for explanations
    - Subgraph extraction for focused analysis
    - Confidence propagation
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[Entity, Set[Relationship]] = defaultdict(set)
        self.reverse_relationships: Dict[Entity, Set[Relationship]] = defaultdict(set)
        self._index_by_type: Dict[EntityType, Set[Entity]] = defaultdict(set)
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        self._index_by_type[entity.type].add(entity)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship between entities."""
        self.relationships[relationship.source].add(relationship)
        self.reverse_relationships[relationship.target].add(relationship)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> Set[Entity]:
        """Get all entities of a specific type."""
        return self._index_by_type[entity_type]
    
    def get_neighbors(self, entity: Entity, 
                     relationship_type: Optional[RelationshipType] = None) -> List[Entity]:
        """Get all neighbors of an entity."""
        neighbors = []
        for rel in self.relationships[entity]:
            if relationship_type is None or rel.type == relationship_type:
                neighbors.append(rel.target)
        return neighbors
    
    def get_predecessors(self, entity: Entity,
                        relationship_type: Optional[RelationshipType] = None) -> List[Entity]:
        """Get all entities that point to this entity."""
        predecessors = []
        for rel in self.reverse_relationships[entity]:
            if relationship_type is None or rel.type == relationship_type:
                predecessors.append(rel.source)
        return predecessors
    
    def find_path(self, start: Entity, end: Entity, 
                  max_depth: int = 5) -> Optional[List[Relationship]]:
        """
        Find a path between two entities using BFS.
        
        Returns:
            List of relationships forming the path, or None if no path exists.
        """
        from collections import deque
        
        visited = {start}
        queue = deque([(start, [])])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                return path
            
            for rel in self.relationships[current]:
                if rel.target not in visited:
                    visited.add(rel.target)
                    queue.append((rel.target, path + [rel]))
        
        return None
    
    def propagate_confidence(self, source: Entity, 
                            decay_factor: float = 0.9) -> Dict[Entity, float]:
        """
        Propagate confidence from a source entity through the graph.
        
        Uses iterative relaxation to compute confidence scores
        for all reachable entities.
        """
        confidences = {source: source.confidence}
        queue = [source]
        visited = {source}
        
        while queue:
            current = queue.pop(0)
            current_conf = confidences[current]
            
            for rel in self.relationships[current]:
                target = rel.target
                propagated = current_conf * rel.weight * decay_factor
                
                if target in confidences:
                    # Combine confidences (probabilistic sum)
                    confidences[target] = 1 - (1 - confidences[target]) * (1 - propagated)
                else:
                    confidences[target] = propagated
                
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
        
        return confidences
    
    def get_subgraph(self, entities: Set[Entity], 
                    include_neighbors: bool = False) -> 'KnowledgeGraph':
        """
        Extract a subgraph containing specified entities.
        
        Args:
            entities: Set of entities to include
            include_neighbors: If True, include direct neighbors
        
        Returns:
            New KnowledgeGraph with the subgraph
        """
        subgraph = KnowledgeGraph()
        
        # Add entities
        for entity in entities:
            subgraph.add_entity(entity)
        
        if include_neighbors:
            for entity in list(entities):
                for rel in self.relationships[entity]:
                    if rel.target in entities:
                        subgraph.add_relationship(rel)
        else:
            # Only add relationships between included entities
            for entity in entities:
                for rel in self.relationships[entity]:
                    if rel.target in entities:
                        subgraph.add_relationship(rel)
        
        return subgraph
    
    def query(self, entity_type: Optional[EntityType] = None,
             min_confidence: float = 0.0,
             properties: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Query entities based on filters.
        
        Args:
            entity_type: Filter by type
            min_confidence: Minimum confidence threshold
            properties: Required properties (key-value pairs)
        
        Returns:
            List of matching entities
        """
        candidates = self.entities.values()
        
        if entity_type:
            candidates = [e for e in candidates if e.type == entity_type]
        
        if min_confidence > 0:
            candidates = [e for e in candidates if e.confidence >= min_confidence]
        
        if properties:
            candidates = [
                e for e in candidates 
                if all(e.properties.get(k) == v for k, v in properties.items())
            ]
        
        return list(candidates)
    
    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        return {
            'entities': [
                {
                    'id': e.id,
                    'type': e.type.value,
                    'properties': e.properties,
                    'confidence': e.confidence,
                    'source': e.source
                }
                for e in self.entities.values()
            ],
            'relationships': [
                {
                    'source': r.source.id,
                    'target': r.target.id,
                    'type': r.type.value,
                    'weight': r.weight,
                    'properties': r.properties
                }
                for rels in self.relationships.values()
                for r in rels
            ]
        }
    
    def save(self, filepath: str) -> None:
        """Save graph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        graph = cls()
        
        # Create entities
        for e_data in data['entities']:
            entity = Entity(
                id=e_data['id'],
                type=EntityType(e_data['type']),
                properties=e_data['properties'],
                confidence=e_data['confidence'],
                source=e_data['source']
            )
            graph.add_entity(entity)
        
        # Create relationships
        for r_data in data['relationships']:
            source = graph.get_entity(r_data['source'])
            target = graph.get_entity(r_data['target'])
            if source and target:
                relationship = Relationship(
                    source=source,
                    target=target,
                    type=RelationshipType(r_data['type']),
                    weight=r_data['weight'],
                    properties=r_data['properties']
                )
                graph.add_relationship(relationship)
        
        return graph
    
    def __len__(self) -> int:
        return len(self.entities)
    
    def __repr__(self) -> str:
        rel_count = sum(len(rels) for rels in self.relationships.values())
        return f"KnowledgeGraph(entities={len(self)}, relationships={rel_count})"
