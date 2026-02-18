"-- path_normalize.ads
-- Normalize file paths (resolve symlinks, handle . vs /)

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package path_normalize is

   Path_Normalization_Error : exception;

   function normalize_path (path : String) return Unbounded_String;

end path_normalize;"