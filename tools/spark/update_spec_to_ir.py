#!/usr/bin/env python3
"""Update spec_to_ir.adb to emit types and constants."""

import re

# Read the file
with open('src/ir/spec_to_ir.adb', 'r', encoding='utf-8') as f:
    content = f.read()

# Old pattern for types emission
old_types = '''      Put_Line (Output, "  \\"types\\": [],");'''

# New types emission
new_types = '''      --  Emit types array
      Put (Output, "  \\"types\\": [");
      for I in Type_Def_Index range 1 .. IR.Types.Count loop
         Put (Output, "{");
         Put (Output, "\\"name\\": \\"" & Identifier_Strings.To_String (IR.Types.Type_Defs (I).Name) & "\\"");
         
         --  Emit kind
         Put (Output, ", \\"kind\\": "");
         case IR.Types.Type_Defs (I).Kind is
            when Type_Struct => Put (Output, "struct");
            when Type_Enum => Put (Output, "enum");
            when Type_Alias => Put (Output, "alias");
            when Type_Generic => Put (Output, "generic");
         end case;
         Put (Output, "\\"");
         
         --  Emit base_type if present
         if Type_Name_Strings.Length (IR.Types.Type_Defs (I).Base_Type) > 0 then
            Put (Output, ", \\"base_type\\": \\"" & 
               Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Base_Type) & "\\"");
         end if;
         
         --  Emit fields for struct types
         if IR.Types.Type_Defs (I).Kind = Type_Struct and then IR.Types.Type_Defs (I).Fields.Count > 0 then
            Put (Output, ", \\"fields\\": [");
            for J in Type_Field_Index range 1 .. IR.Types.Type_Defs (I).Fields.Count loop
               Put (Output, "{");
               Put (Output, "\\"name\\": \\"" & 
                  Identifier_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Name) & "\\"");
               Put (Output, ", \\"type\\": \\"" & 
                  Type_Name_Strings.To_String (IR.Types.Type_Defs (I).Fields.Fields (J).Field_Type) & "\\"");
               Put (Output, "}");
               if J