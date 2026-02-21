# STUNIR Business IR

Intermediate Representation for business-oriented programming languages.

## Overview

The Business IR provides a comprehensive set of nodes for representing business programming constructs, particularly for COBOL and BASIC languages. It supports:

- **Record Structures**: Hierarchical data definitions with level numbers
- **PICTURE Clauses**: Data formatting and validation
- **File Operations**: Sequential, indexed, and relative file handling
- **Data Processing**: Business arithmetic and string operations
- **Control Flow**: Procedures, loops, and conditional statements

## Architecture

```
ir/business/
├── __init__.py        # Package exports
├── business_ir.py     # Core IR classes and statements
├── records.py         # Record structures and data items
├── files.py           # File definitions and I/O operations
└── README.md          # This file
```

## Usage

### Basic Import

```python
from ir.business import (
    # Program structure
    BusinessProgram, Division, Section, Paragraph,
    # Enumerations
    FileOrganization, FileAccess, DataUsage, PictureType,
    # Records
    RecordStructure, DataItem, PictureClause, OccursClause,
    # Files
    FileDefinition, FileControl, OpenStatement, ReadStatement,
    # Statements
    MoveStatement, ComputeStatement, PerformStatement,
)
```

### Creating a COBOL Program

```python
from ir.business import (
    BusinessProgram, Division, Paragraph, DataItem, PictureClause,
    FileDefinition, FileOrganization, DisplayStatement, Literal
)

# Create a simple COBOL program
program = BusinessProgram(
    name='HELLO-WORLD',
    divisions=[
        Division(name='IDENTIFICATION'),
        Division(name='ENVIRONMENT'),
        Division(name='DATA'),
        Division(name='PROCEDURE'),
    ],
    data_items=[
        DataItem(
            name='WS-MESSAGE',
            level=1,
            picture=PictureClause(pattern='X(50)'),
            value='Hello, COBOL World!'
        )
    ],
    paragraphs=[
        Paragraph(
            name='MAIN-PARA',
            statements=[
                DisplayStatement(items=[Literal(value='Hello, COBOL World!')])
            ]
        )
    ]
)

# Convert to dictionary
ir_dict = program.to_dict()
```

### Creating a BASIC Program

```python
from ir.business import (
    BusinessProgram, DimStatement, DefFunction, Assignment,
    ForLoop, BasicPrintStatement, PrintItem, Literal, Identifier, BinaryExpr
)

# Create a simple BASIC program
program = BusinessProgram(
    name='SQUARES',
    line_numbers=True,
    dim_statements=[
        DimStatement(variable='VALUES', dimensions=[10])
    ],
    def_functions=[
        DefFunction(
            name='SQUARE',
            parameter='X',
            expression=BinaryExpr(
                left=Identifier(name='X'),
                op='mul',
                right=Identifier(name='X')
            )
        )
    ],
    statements=[
        ForLoop(
            variable='I',
            start=Literal(value=1),
            end=Literal(value=10),
            statements=[
                BasicPrintStatement(items=[
                    PrintItem(value=Identifier(name='I')),
                    PrintItem(value=Literal(value=' squared = ')),
                    PrintItem(value=FunctionCall(name='FNSQUARE', arguments=[Identifier(name='I')]))
                ])
            ]
        )
    ]
)
```

## Record Structures

### COBOL Level Numbers

COBOL uses hierarchical level numbers (01-49) to define record structures:

```python
from ir.business import RecordStructure, DataItem, PictureClause

employee_record = RecordStructure(
    name='EMPLOYEE-RECORD',
    level=1,
    fields=[
        DataItem(name='EMP-ID', level=5, picture=PictureClause(pattern='9(5)')),
        DataItem(name='EMP-NAME', level=5, picture=PictureClause(pattern='X(30)')),
        DataItem(
            name='EMP-ADDRESS',
            level=5,
            children=[
                DataItem(name='ADDR-STREET', level=10, picture=PictureClause(pattern='X(30)')),
                DataItem(name='ADDR-CITY', level=10, picture=PictureClause(pattern='X(20)')),
                DataItem(name='ADDR-STATE', level=10, picture=PictureClause(pattern='XX')),
                DataItem(name='ADDR-ZIP', level=10, picture=PictureClause(pattern='9(5)')),
            ]
        ),
        DataItem(name='EMP-SALARY', level=5, picture=PictureClause(pattern='9(7)V99')),
    ]
)
```

### PICTURE Clause

The `PictureClause` class supports COBOL PICTURE patterns:

| Symbol | Description |
|--------|-------------|
| 9 | Numeric digit |
| A | Alphabetic |
| X | Alphanumeric |
| V | Implied decimal point |
| S | Sign |
| Z | Zero suppression |
| * | Check protection |
| $ | Currency symbol |
| , | Comma |
| . | Decimal point |

```python
from ir.business import PictureClause

# Numeric with 2 decimal places
pic1 = PictureClause(pattern='9(5)V99')
print(pic1.size)           # 7
print(pic1.decimal_places) # 2

# Edited currency field
pic2 = PictureClause(pattern='$Z,ZZZ,ZZ9.99')
print(pic2.data_type)      # PictureType.EDITED
```

### OCCURS Clause (Arrays)

```python
from ir.business import OccursClause, SortKey

# Fixed-size array
fixed_array = OccursClause(times=100, indexed_by=['IDX-1'])

# Variable-length array
variable_array = OccursClause(
    min_times=1,
    max_times=999,
    depending_on='ITEM-COUNT',
    indexed_by=['IDX-1'],
    keys=[SortKey(name='ITEM-KEY', ascending=True)]
)
```

## File Operations

### File Definitions

```python
from ir.business import FileDefinition, FileControl, FileOrganization, FileAccess

# Indexed file
indexed_file = FileDefinition(
    name='CUSTOMER-FILE',
    organization=FileOrganization.INDEXED,
    access=FileAccess.DYNAMIC,
    record_key='CUST-ID',
    status='WS-FILE-STATUS'
)

# FILE-CONTROL entry
file_control = FileControl(
    select_name='CUSTOMER-FILE',
    assign_to='CUSTOMER.DAT',
    organization=FileOrganization.INDEXED,
    access=FileAccess.DYNAMIC,
    record_key='CUST-ID',
    file_status='WS-FILE-STATUS'
)
```

### File I/O Statements

```python
from ir.business import (
    OpenStatement, OpenFile, ReadStatement, WriteStatement,
    CloseStatement, OpenMode
)

# Open file
open_stmt = OpenStatement(files=[
    OpenFile(name='CUSTOMER-FILE', mode=OpenMode.IO)
])

# Read with AT END handling
read_stmt = ReadStatement(
    file_name='CUSTOMER-FILE',
    into='WS-CUSTOMER',
    at_end=[MoveStatement(source=Literal(value=1), destinations=['WS-EOF'])]
)

# Write with advancing
write_stmt = WriteStatement(
    record_name='REPORT-LINE',
    from_value='WS-OUTPUT',
    after_advancing=AdvanceSpec(lines=2)
)
```

## Data Processing Statements

### COBOL Statements

```python
from ir.business import (
    MoveStatement, ComputeStatement, AddStatement, SubtractStatement,
    PerformStatement, VaryingClause
)

# MOVE statement
move = MoveStatement(
    source=Literal(value='SPACES'),
    destinations=['WS-NAME', 'WS-ADDRESS']
)

# COMPUTE statement
compute = ComputeStatement(
    target='WS-TOTAL',
    expression=BinaryExpr(
        left=Identifier(name='QTY'),
        op='mul',
        right=Identifier(name='PRICE')
    ),
    rounded=True
)

# PERFORM VARYING
perform = PerformStatement(
    paragraph_name='PROCESS-TABLE',
    varying=VaryingClause(
        identifier='IDX',
        from_value=Literal(value=1),
        by_value=Literal(value=1),
        until_value=Condition(
            left=Identifier(name='IDX'),
            op='gt',
            right=Literal(value=100)
        )
    )
)
```

### BASIC Statements

```python
from ir.business import (
    Assignment, ForLoop, WhileLoop, GotoStatement, GosubStatement,
    BasicPrintStatement, BasicInputStatement, DataStatement
)

# Assignment (LET)
assignment = Assignment(variable='COUNT', value=Literal(value=0))

# FOR...NEXT loop
for_loop = ForLoop(
    variable='I',
    start=Literal(value=1),
    end=Literal(value=10),
    step=Literal(value=2),
    statements=[...]
)

# DATA and READ
data_stmt = DataStatement(values=[
    Literal(value='Apple'),
    Literal(value=1.50),
    Literal(value='Banana'),
    Literal(value=0.75)
])
```

## Utility Functions

```python
from ir.business import (
    create_numeric_field,
    create_alphanumeric_field,
    create_edited_field,
    parse_picture
)

# Create common field types
amount = create_numeric_field('AMOUNT', level=5, digits=9, decimals=2, signed=True)
name = create_alphanumeric_field('NAME', level=5, size=30)
edited = create_edited_field('DISPLAY-AMT', level=5, pattern='$ZZZ,ZZ9.99')
```

## Enumerations

| Enum | Values | Description |
|------|--------|-------------|
| `FileOrganization` | SEQUENTIAL, INDEXED, RELATIVE, LINE_SEQUENTIAL | File organization types |
| `FileAccess` | SEQUENTIAL, RANDOM, DYNAMIC | File access modes |
| `DataUsage` | DISPLAY, BINARY, COMP, COMP-1/2/3/4/5, INDEX, POINTER | Data usage clauses |
| `PictureType` | NUMERIC, ALPHABETIC, ALPHANUMERIC, EDITED | PICTURE clause types |
| `OpenMode` | INPUT, OUTPUT, IO, EXTEND | File open modes |
| `BasicVarType` | NUMERIC, STRING, INTEGER | BASIC variable types |

## Integration with Emitters

The Business IR is designed to work with the COBOL and BASIC emitters:

```python
from ir.business import BusinessProgram, ...
from targets.business import COBOLEmitter, BASICEmitter

# Create IR
program = BusinessProgram(...)

# Emit COBOL code
cobol_emitter = COBOLEmitter()
result = cobol_emitter.emit(program.to_dict())
print(result.code)

# Emit BASIC code
basic_emitter = BASICEmitter()
result = basic_emitter.emit(program.to_dict())
print(result.code)
```

## See Also

- [Business Emitters README](../../targets/business/README.md)
- [COBOL Language Reference](https://www.ibm.com/docs/en/cobol-zos)
- [BASIC Language Reference](https://en.wikipedia.org/wiki/BASIC)
