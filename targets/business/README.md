# STUNIR Business Language Emitters

Code emitters for business-oriented programming languages.

## Overview

The Business Emitters package provides code generators for:

- **COBOL**: Enterprise business data processing with record structures
- **BASIC**: Simple programming with line numbers and interactive features

## Architecture

```
targets/business/
├── __init__.py          # Package exports
├── cobol_emitter.py     # COBOL code emitter
├── basic_emitter.py     # BASIC code emitter
└── README.md            # This file
```

## Usage

### COBOL Emitter

```python
from targets.business import COBOLEmitter

# Create emitter
emitter = COBOLEmitter(dialect='standard')

# Generate code from IR
ir = {
    'name': 'HELLO-WORLD',
    'paragraphs': [
        {
            'name': 'MAIN-PARA',
            'statements': [
                {'kind': 'display_statement', 'items': [
                    {'kind': 'literal', 'value': 'Hello, COBOL!'}
                ]},
                {'kind': 'stop_statement', 'run': True}
            ]
        }
    ]
}

result = emitter.emit(ir)
print(result.code)
```

Output:
```cobol
       IDENTIFICATION DIVISION.
       PROGRAM-ID. HELLO-WORLD.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       PROCEDURE DIVISION.
       MAIN-PARA.
           DISPLAY "Hello, COBOL!".
           STOP RUN.
```

### BASIC Emitter

```python
from targets.business import BASICEmitter

# Create emitter
emitter = BASICEmitter(dialect='standard', line_increment=10)

# Generate code from IR
ir = {
    'name': 'HELLO',
    'line_numbers': True,
    'statements': [
        {'kind': 'basic_print_screen', 'items': [
            {'value': {'kind': 'literal', 'value': 'Hello, BASIC!'}}
        ]},
        {'kind': 'basic_input_user', 'prompt': 'NAME', 'variables': ['NAME$']},
        {'kind': 'basic_print_screen', 'items': [
            {'value': {'kind': 'literal', 'value': 'Hello, '}},
            {'value': {'kind': 'identifier', 'name': 'NAME$'}}
        ]}
    ]
}

result = emitter.emit(ir)
print(result.code)
```

Output:
```basic
10 REM HELLO
20 REM
30 PRINT "Hello, BASIC!"
40 INPUT "NAME"; NAME$
50 PRINT "Hello, " NAME$
60 END
```

## COBOL Emitter Features

### Four Divisions

The COBOL emitter generates all four divisions:

| Division | Content |
|----------|---------|
| IDENTIFICATION | Program name, author, dates |
| ENVIRONMENT | Configuration, file control |
| DATA | File section, working-storage |
| PROCEDURE | Paragraphs with statements |

### File Handling

```python
ir = {
    'name': 'FILE-DEMO',
    'files': [
        {
            'name': 'CUSTOMER-FILE',
            'assign_to': 'CUSTOMER.DAT',
            'organization': 'indexed',
            'access': 'sequential',
            'record_key': 'CUST-ID',
            'file_status': 'WS-STATUS',
            'record': {
                'name': 'CUSTOMER-RECORD',
                'level': 1,
                'children': [
                    {'name': 'CUST-ID', 'level': 5, 'picture': {'pattern': '9(5)'}},
                    {'name': 'CUST-NAME', 'level': 5, 'picture': {'pattern': 'X(30)'}}
                ]
            }
        }
    ],
    'paragraphs': [...]
}
```

Generated FILE-CONTROL:
```cobol
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE
               ASSIGN TO "CUSTOMER.DAT"
               ORGANIZATION IS INDEXED
               RECORD KEY IS CUST-ID
               FILE STATUS IS WS-STATUS.
```

### Record Structures

Level numbers (01-49) are properly formatted:

```cobol
       01  EMPLOYEE-RECORD.
           05  EMP-ID            PIC 9(5).
           05  EMP-NAME          PIC X(30).
           05  EMP-ADDRESS.
               10  ADDR-STREET   PIC X(30).
               10  ADDR-CITY     PIC X(20).
               10  ADDR-STATE    PIC XX.
               10  ADDR-ZIP      PIC 9(5).
           05  EMP-SALARY        PIC 9(7)V99.
```

### PICTURE Clauses

Supported PICTURE patterns:

| Pattern | Example | Description |
|---------|---------|-------------|
| 9(n) | 9(5) | Numeric digits |
| X(n) | X(30) | Alphanumeric |
| A(n) | A(10) | Alphabetic |
| 9(n)V9(m) | 9(5)V99 | Decimal numeric |
| S9(n)V9(m) | S9(7)V99 | Signed decimal |
| Z(n)9 | ZZZ9 | Zero-suppressed |
| $Z,ZZ9.99 | $Z,ZZ9.99 | Currency edited |

### Control Flow

```python
# PERFORM VARYING
{'kind': 'perform_statement',
 'paragraph_name': 'PROCESS-TABLE',
 'varying': {
     'identifier': 'IDX',
     'from_value': {'kind': 'literal', 'value': 1},
     'by_value': {'kind': 'literal', 'value': 1},
     'until_value': {'kind': 'condition', 'left': ..., 'op': '>', 'right': ...}
 }}

# EVALUATE
{'kind': 'evaluate_statement',
 'subjects': [{'kind': 'identifier', 'name': 'STATUS-CODE'}],
 'when_clauses': [
     {'conditions': [{'kind': 'when_condition', 'value': {'kind': 'literal', 'value': 'A'}}],
      'statements': [...]},
 ],
 'when_other': [...]}
```

### Supported Dialects

| Dialect | Description |
|---------|-------------|
| standard | COBOL-85 standard |
| ibm | IBM Enterprise COBOL |
| microfocus | Micro Focus COBOL |
| gnu | GnuCOBOL |

## BASIC Emitter Features

### Line Numbers

Line numbers are automatically generated with configurable increment:

```python
emitter = BASICEmitter(line_increment=10)  # 10, 20, 30, ...
```

### DIM Statements

```python
ir = {
    'dim_statements': [
        {'variable': 'ITEMS$', 'dimensions': [100]},      # Single dimension
        {'variable': 'MATRIX', 'dimensions': [10, 10]}   # Multi-dimensional
    ]
}
```

Output:
```basic
10 DIM ITEMS$(100)
20 DIM MATRIX(10, 10)
```

### DEF FN Functions

```python
ir = {
    'def_functions': [
        {
            'name': 'SQUARE',
            'parameter': 'X',
            'expression': {
                'kind': 'binary_expr',
                'left': {'kind': 'identifier', 'name': 'X'},
                'op': 'mul',
                'right': {'kind': 'identifier', 'name': 'X'}
            }
        }
    ]
}
```

Output:
```basic
10 DEF FNSQUARE(X) = (X * X)
```

### Control Flow

```basic
' FOR...NEXT
10 FOR I = 1 TO 10 STEP 2
20   PRINT I
30 NEXT I

' WHILE...WEND
40 WHILE X < 100
50   LET X = X + 1
60 WEND

' GOTO/GOSUB
70 GOTO 100
80 GOSUB 500
90 RETURN
```

### I/O Statements

```basic
' User input
10 INPUT "Enter name: "; NAME$
20 PRINT "Hello, "; NAME$

' DATA and READ
30 DATA "Apple", 1.50, "Banana", 0.75
40 READ ITEM$, PRICE
50 RESTORE 30

' File operations
60 OPEN "DATA.TXT" FOR INPUT AS #1
70 INPUT #1, NAME$, VALUE
80 CLOSE #1
```

### Supported Dialects

| Dialect | Description |
|---------|-------------|
| standard | Classic BASIC |
| qbasic | Microsoft QBasic (block IF support) |
| gwbasic | GW-BASIC |

## EmitterResult

Both emitters return an `EmitterResult` object:

```python
result = emitter.emit(ir)

# Generated code
print(result.code)

# Manifest with metadata
print(result.manifest)
# {
#     'schema': 'stunir.codegen.cobol.v1',
#     'timestamp': 1704067200,
#     'program_name': 'PAYROLL',
#     'dialect': 'standard',
#     'code_hash': 'abc123...',
#     'code_lines': 150,
#     'files_count': 2,
#     'paragraphs_count': 5,
#     'data_items_count': 20,
#     'errors': [],
#     'warnings': []
# }
```

## Example Programs

### COBOL Payroll Processing

```python
ir = {
    'name': 'PAYROLL',
    'files': [{
        'name': 'EMPLOYEE-FILE',
        'organization': 'indexed',
        'record_key': 'EMP-ID',
        'record': {
            'name': 'EMPLOYEE-RECORD',
            'level': 1,
            'children': [
                {'name': 'EMP-ID', 'level': 5, 'picture': {'pattern': '9(5)'}},
                {'name': 'EMP-SALARY', 'level': 5, 'picture': {'pattern': '9(7)V99'}}
            ]
        }
    }],
    'data_items': [
        {'name': 'WS-TOTAL', 'level': 1, 'picture': {'pattern': '9(10)V99'}, 'value': 0},
        {'name': 'WS-EOF', 'level': 1, 'picture': {'pattern': '9'}, 'value': 0}
    ],
    'paragraphs': [
        {
            'name': 'MAIN-PARA',
            'statements': [
                {'kind': 'open_statement', 'files': [{'name': 'EMPLOYEE-FILE', 'mode': 'input'}]},
                {'kind': 'perform_statement', 'paragraph_name': 'PROCESS',
                 'until': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'WS-EOF'},
                           'op': '=', 'right': {'kind': 'literal', 'value': 1}}},
                {'kind': 'close_statement', 'files': ['EMPLOYEE-FILE']},
                {'kind': 'display_statement', 'items': [
                    {'kind': 'literal', 'value': 'Total: '},
                    {'kind': 'identifier', 'name': 'WS-TOTAL'}
                ]}
            ]
        },
        {
            'name': 'PROCESS',
            'statements': [
                {'kind': 'read_statement', 'file_name': 'EMPLOYEE-FILE',
                 'at_end': [{'kind': 'move_statement', 
                             'source': {'kind': 'literal', 'value': 1},
                             'destinations': ['WS-EOF']}],
                 'not_at_end': [{'kind': 'add_statement',
                                 'values': [{'kind': 'identifier', 'name': 'EMP-SALARY'}],
                                 'to_value': 'WS-TOTAL'}]}
            ]
        }
    ]
}
```

### BASIC Inventory System

```python
ir = {
    'name': 'INVENTORY',
    'line_numbers': True,
    'dim_statements': [
        {'variable': 'ITEM$', 'dimensions': [100]},
        {'variable': 'QTY', 'dimensions': [100]},
        {'variable': 'PRICE', 'dimensions': [100]}
    ],
    'def_functions': [
        {'name': 'VALUE', 'parameter': 'I',
         'expression': {'kind': 'binary_expr', 'op': 'mul',
                        'left': {'kind': 'identifier', 'name': 'QTY', 
                                 'subscripts': [{'kind': 'identifier', 'name': 'I'}]},
                        'right': {'kind': 'identifier', 'name': 'PRICE',
                                  'subscripts': [{'kind': 'identifier', 'name': 'I'}]}}}
    ],
    'statements': [
        {'kind': 'assignment', 'variable': 'COUNT', 'value': {'kind': 'literal', 'value': 0}},
        {'kind': 'rem_statement', 'text': 'Main Menu'},
        {'kind': 'basic_print_screen', 'items': [
            {'value': {'kind': 'literal', 'value': 'INVENTORY SYSTEM'}}]},
        {'kind': 'basic_print_screen', 'items': [
            {'value': {'kind': 'literal', 'value': '1. Add Item'}}]},
        {'kind': 'basic_print_screen', 'items': [
            {'value': {'kind': 'literal', 'value': '2. List Items'}}]},
        {'kind': 'basic_print_screen', 'items': [
            {'value': {'kind': 'literal', 'value': '3. Exit'}}]},
        {'kind': 'basic_input_user', 'prompt': 'Choice', 'variables': ['C']},
        {'kind': 'if_statement',
         'condition': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'C'},
                       'op': '=', 'right': {'kind': 'literal', 'value': 3}},
         'then_statements': [{'kind': 'end_statement'}]},
        {'kind': 'goto_statement', 'target': '30'}
    ]
}
```

## Error Handling

Both emitters track errors and warnings:

```python
result = emitter.emit(ir)

# Check for issues
if result.manifest['errors']:
    print('Errors:', result.manifest['errors'])

if result.manifest['warnings']:
    print('Warnings:', result.manifest['warnings'])
```

## See Also

- [Business IR README](../../ir/business/README.md)
- [STUNIR Documentation](../../docs/)
- [Tests](../../tests/codegen/test_business_emitters.py)
