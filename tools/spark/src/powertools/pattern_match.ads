"-- pattern_match.ads
-- Evaluate patterns against files/directories

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package pattern_match is

   Pattern_Match_Error : exception;

   function matches_pattern (
      path : String,
      pattern : String
   ) return Boolean;

end pattern_match;"