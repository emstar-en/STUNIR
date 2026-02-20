"-- pattern_match.ads
-- Evaluate glob patterns against file/directory paths.
--
-- REGEX_IR_REF: schema/stunir_regex_ir_v1.dcbor.json
--              group: extraction.c_function / pattern_id: c_identifier_at_end
-- NOTE: This package implements GLOB pattern matching (*, ?, [...]), NOT
-- regular expression matching. Glob patterns are a strict subset of regular
-- languages. The regex IR documents the regex equivalents for reference.
-- See CONTRIBUTING.md for the governance rule on adding new patterns.

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package pattern_match is

   Pattern_Match_Error : exception;

   --  Returns True if Path matches the glob Pattern.
   --  Supported wildcards: * (any sequence), ? (any single char), [...] (char class).
   --  Raises Pattern_Match_Error if Pattern is malformed.
   function matches_pattern (
      path    : String;
      pattern : String
   ) return Boolean;

end pattern_match;"