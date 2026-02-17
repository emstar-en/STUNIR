-- path_normalize.adb
-- Normalize file paths (resolve symlinks, handle . vs /)

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories;

package body path_normalize is

   function normalize_path (path : String) return Unbounded_String is
         normalized : Unbounded_String := To_Unbounded_String(path);
            begin
                  -- Basic normalization: replace backslashes with forward slashes on Windows
                        if Ada.Directories.Is_Windows then
                                 for i in normalized.First..normalized.Last loop
                                             if Element(normalized, i) = '\\' then
                                                            Replace_Slice(
                                                                                  normalized,
                                                                                                    Pos => i,
                                                                                                                      Length => 1,
                                                                                                                                        New_Item => '/'
                                                                                                                                                       );
                                                                                                                                                                   end if;
                                                                                                                                                                            end loop;
                                                                                                                                                                                  end if;
                                                                                                                                                                                  
                                                                                                                                                                                        -- Remove redundant slashes (e.g., "dir//file" -> "dir/file")
                                                                                                                                                                                              while To_String(normalized).Contains("//") loop
                                                                                                                                                                                                       Replace_Slice(
                                                                                                                                                                                                                    normalized,
                                                                                                                                                                                                                   Pos => To_String(normalized)'Find("//"),
                                                                                                                                                                                                                               Length => 2,
                                                                                                                                                                                                                                           New_Item => "/"
                                                                                                                                                                                                                                                    );
                                                                                                                                                                                                                                                          end loop;
                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                -- Remove trailing slash if present (except for root)
                                                                                                                                                                                                                                                                      if normalized.Length > 1 and Element(normalized, normalized.Last) = '/' then
                                                                                                                                                                                                                                                                               Delete(
                                                                                                                                                                                                                                                                                            normalized,
                                                                                                                                                                                                                                                                                           Pos => normalized.Last,
                                                                                                                                                                                                                                                                                                       Keep => normalized.Last - 1
                                                                                                                                                                                                                                                                                                                );
                                                                                                                                                                                                                                                                                                             end if;
                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                   return normalized;
                                                                                                                                                                                                                                                                                                                      exception
                                                                                                                                                                                                                                                                                                                            when others =>
                                                                                                                                                                                                                                                                                        raise Path_Normalization_Error with "Path normalization failed for: " & To_String(normalized);
   end normalize_path;
   
   end path_normalize;"