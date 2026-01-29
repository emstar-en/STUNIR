# DO-331 Traceability Guide

## Overview

Traceability is a core requirement of DO-331. The STUNIR tools provide complete bidirectional traceability between IR elements and model elements.

## Trace Matrix Structure

```json
{
  "schema": "stunir.trace.do331.v1",
  "ir_hash": "sha256:...",
  "model_hash": "sha256:...",
  "entries": [
    {
      "trace_id": 1,
      "source_id": 100,
      "source_path": "module.function_name",
      "target_id": 200,
      "target_path": "Module::FunctionName",
      "rule": "Function to Action Definition",
      "do331_objective": "MB.3",
      "verified": false
    }
  ]
}
```

## Trace Types

| Type | Direction | Description |
|------|-----------|-------------|
| IR_To_Model | Forward | IR element to model element |
| Model_To_IR | Backward | Model element to IR source |
| Model_To_Model | Internal | Model element relationships |
| Model_To_Req | Requirements | Model to requirement mapping |

## Transformation Rules

| Rule | DO-331 Objective | Description |
|------|------------------|-------------|
| Rule_Module_To_Package | MB.2 | Module becomes package |
| Rule_Function_To_Action | MB.3 | Function becomes action def |
| Rule_Type_To_Attribute_Def | MB.2 | Type becomes attribute def |
| Rule_If_To_Decision | MB.3 | If statement becomes decision |
| Rule_State_Machine | MB.3 | State machine mapping |
| Rule_Transition | MB.3 | Transition mapping |

## Gap Analysis

The traceability framework includes gap analysis:

```ada
function Analyze_Gaps (
   Matrix : Trace_Matrix;
   IR_IDs : Element_ID_Array
) return Gap_Report;
```

Returns:
- Total IR elements
- Traced elements
- Missing traces
- Gap percentage
- Completeness status

## DO-331 Table MB-1 Compliance

The trace matrix addresses DO-331 Table MB-1 objectives:

| Objective | Description | Implementation |
|-----------|-------------|----------------|
| MB.1.1 | LLR traceable to model | trace_matrix.ir_to_model |
| MB.1.2 | Model traceable to HLR | trace_matrix.model_to_req |
| MB.1.3 | Test cases traceable | trace_matrix.test_to_model |
