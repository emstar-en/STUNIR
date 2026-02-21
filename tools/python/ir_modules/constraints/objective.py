"""Objective function and Constraint Model definitions.

This module defines objective functions and the main
ConstraintModel class for constraint satisfaction problems.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .constraint_ir import ObjectiveType, SearchStrategy, ValueChoice, VariableType
from .variable import Variable, ArrayVariable, IndexSet, Parameter
from .domain import Domain
from .constraint import Constraint, Expression, VariableRef


@dataclass
class Objective:
    """Objective function for optimization.
    
    Attributes:
        objective_type: MINIMIZE, MAXIMIZE, or SATISFY
        expression: Expression to optimize (None for SATISFY)
    """
    objective_type: ObjectiveType
    expression: Optional[Expression] = None
    
    @classmethod
    def minimize(cls, expr: Expression) -> 'Objective':
        """Create minimize objective.
        
        Args:
            expr: Expression to minimize
            
        Returns:
            Objective with MINIMIZE type
        """
        return cls(ObjectiveType.MINIMIZE, expr)
    
    @classmethod
    def maximize(cls, expr: Expression) -> 'Objective':
        """Create maximize objective.
        
        Args:
            expr: Expression to maximize
            
        Returns:
            Objective with MAXIMIZE type
        """
        return cls(ObjectiveType.MAXIMIZE, expr)
    
    @classmethod
    def satisfy(cls) -> 'Objective':
        """Create satisfaction objective.
        
        Returns:
            Objective with SATISFY type
        """
        return cls(ObjectiveType.SATISFY)
    
    def __str__(self) -> str:
        if self.objective_type == ObjectiveType.SATISFY:
            return "satisfy"
        elif self.objective_type == ObjectiveType.MINIMIZE:
            return f"minimize {self.expression}"
        elif self.objective_type == ObjectiveType.MAXIMIZE:
            return f"maximize {self.expression}"
        return "unknown"


@dataclass
class SearchAnnotation:
    """Search annotation for solving.
    
    Attributes:
        variables: Variables to search over (list of names or 'all')
        strategy: Variable selection strategy
        value_choice: Value choice heuristic
        restart: Restart strategy (optional)
    """
    variables: Union[List[str], str]
    strategy: SearchStrategy = SearchStrategy.INPUT_ORDER
    value_choice: ValueChoice = ValueChoice.INDOMAIN_MIN
    restart: Optional[str] = None
    
    def to_minizinc(self) -> str:
        """Convert to MiniZinc search annotation.
        
        Returns:
            MiniZinc annotation string
        """
        if isinstance(self.variables, str):
            vars_str = self.variables
        else:
            vars_str = "[" + ", ".join(self.variables) + "]"
        
        strategy_map = {
            SearchStrategy.INPUT_ORDER: "input_order",
            SearchStrategy.FIRST_FAIL: "first_fail",
            SearchStrategy.ANTI_FIRST_FAIL: "anti_first_fail",
            SearchStrategy.SMALLEST: "smallest",
            SearchStrategy.LARGEST: "largest",
            SearchStrategy.OCCURRENCE: "occurrence",
            SearchStrategy.MOST_CONSTRAINED: "most_constrained",
            SearchStrategy.MAX_REGRET: "max_regret",
            SearchStrategy.DOM_W_DEG: "dom_w_deg",
        }
        
        value_map = {
            ValueChoice.INDOMAIN_MIN: "indomain_min",
            ValueChoice.INDOMAIN_MAX: "indomain_max",
            ValueChoice.INDOMAIN_MEDIAN: "indomain_median",
            ValueChoice.INDOMAIN_RANDOM: "indomain_random",
            ValueChoice.INDOMAIN_SPLIT: "indomain_split",
            ValueChoice.INDOMAIN_REVERSE_SPLIT: "indomain_reverse_split",
        }
        
        strat = strategy_map.get(self.strategy, "input_order")
        val = value_map.get(self.value_choice, "indomain_min")
        
        return f"int_search({vars_str}, {strat}, {val}, complete)"


@dataclass
class ConstraintModel:
    """Complete constraint satisfaction/optimization model.
    
    Attributes:
        name: Model name
        variables: Decision variables
        arrays: Array variables
        parameters: Model parameters (constants)
        constraints: List of constraints
        objective: Optimization objective
        search: Search annotations
        output: Output specification
    """
    name: str
    variables: List[Variable] = field(default_factory=list)
    arrays: List[ArrayVariable] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    objective: Objective = field(default_factory=Objective.satisfy)
    search: List[SearchAnnotation] = field(default_factory=list)
    output: List[str] = field(default_factory=list)
    
    def add_variable(self, name: str, var_type: VariableType, domain: Domain, 
                     annotations: List[str] = None) -> Variable:
        """Add a decision variable to the model.
        
        Args:
            name: Variable name
            var_type: Variable type
            domain: Variable domain
            annotations: Optional annotations
            
        Returns:
            The created Variable
        """
        var = Variable(name, var_type, domain, annotations or [])
        self.variables.append(var)
        return var
    
    def add_int_variable(self, name: str, lower: int, upper: int) -> Variable:
        """Add an integer variable with range domain.
        
        Args:
            name: Variable name
            lower: Lower bound
            upper: Upper bound
            
        Returns:
            The created Variable
        """
        return self.add_variable(name, VariableType.INT, Domain.int_range(lower, upper))
    
    def add_bool_variable(self, name: str) -> Variable:
        """Add a boolean variable.
        
        Args:
            name: Variable name
            
        Returns:
            The created Variable
        """
        return self.add_variable(name, VariableType.BOOL, Domain.bool_domain())
    
    def add_array(self, name: str, element_type: VariableType, index_set: IndexSet,
                  element_domain: Domain, annotations: List[str] = None) -> ArrayVariable:
        """Add an array variable to the model.
        
        Args:
            name: Array name
            element_type: Type of elements
            index_set: Index set
            element_domain: Domain for elements
            annotations: Optional annotations
            
        Returns:
            The created ArrayVariable
        """
        arr = ArrayVariable(name, element_type, index_set, element_domain, annotations or [])
        self.arrays.append(arr)
        return arr
    
    def add_int_array(self, name: str, size: int, lower: int, upper: int) -> ArrayVariable:
        """Add an integer array with range domain.
        
        Args:
            name: Array name
            size: Array size
            lower: Lower bound for elements
            upper: Upper bound for elements
            
        Returns:
            The created ArrayVariable
        """
        return self.add_array(
            name, 
            VariableType.INT, 
            IndexSet([(1, size)]),
            Domain.int_range(lower, upper)
        )
    
    def add_parameter(self, name: str, value: any, param_type: VariableType = None) -> Parameter:
        """Add a parameter (constant) to the model.
        
        Args:
            name: Parameter name
            value: Parameter value
            param_type: Optional type
            
        Returns:
            The created Parameter
        """
        param = Parameter(name, value, param_type)
        self.parameters.append(param)
        return param
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the model.
        
        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)
    
    def set_objective(self, obj: Objective) -> None:
        """Set the optimization objective.
        
        Args:
            obj: Objective to set
        """
        self.objective = obj
    
    def minimize(self, expr: Expression) -> None:
        """Set minimize objective.
        
        Args:
            expr: Expression to minimize
        """
        self.objective = Objective.minimize(expr)
    
    def maximize(self, expr: Expression) -> None:
        """Set maximize objective.
        
        Args:
            expr: Expression to maximize
        """
        self.objective = Objective.maximize(expr)
    
    def add_search(self, variables: Union[List[str], str], 
                   strategy: SearchStrategy = SearchStrategy.INPUT_ORDER,
                   value_choice: ValueChoice = ValueChoice.INDOMAIN_MIN) -> None:
        """Add a search annotation.
        
        Args:
            variables: Variables to search over
            strategy: Variable selection strategy
            value_choice: Value choice heuristic
        """
        annotation = SearchAnnotation(variables, strategy, value_choice)
        self.search.append(annotation)
    
    def add_output(self, output_str: str) -> None:
        """Add an output specification.
        
        Args:
            output_str: Output string/expression
        """
        self.output.append(output_str)
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a variable by name.
        
        Args:
            name: Variable name
            
        Returns:
            Variable or None if not found
        """
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def get_array(self, name: str) -> Optional[ArrayVariable]:
        """Get an array by name.
        
        Args:
            name: Array name
            
        Returns:
            ArrayVariable or None if not found
        """
        for arr in self.arrays:
            if arr.name == name:
                return arr
        return None
    
    def validate(self) -> List[str]:
        """Validate the model and return any errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        var_names = {v.name for v in self.variables}
        array_names = {a.name for a in self.arrays}
        param_names = {p.name for p in self.parameters}
        
        # Check for duplicate names
        all_names = [v.name for v in self.variables] + \
                    [a.name for a in self.arrays] + \
                    [p.name for p in self.parameters]
        if len(all_names) != len(set(all_names)):
            errors.append("Duplicate variable/array/parameter names detected")
        
        # Validate model name
        if not self.name:
            errors.append("Model name is required")
        
        return errors
    
    def __str__(self) -> str:
        parts = [f"Model: {self.name}"]
        parts.append(f"  Variables: {len(self.variables)}")
        parts.append(f"  Arrays: {len(self.arrays)}")
        parts.append(f"  Parameters: {len(self.parameters)}")
        parts.append(f"  Constraints: {len(self.constraints)}")
        parts.append(f"  Objective: {self.objective}")
        return "\n".join(parts)
