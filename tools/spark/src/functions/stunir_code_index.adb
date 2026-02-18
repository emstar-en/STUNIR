-------------------------------------------------------------------------------
--  STUNIR Code Index - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Indexes source code files and produces a SHA-256 keyed JSON manifest.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Directories;
with Ada.Strings.Unbounded;
with Ada.Streams;
with Ada.Streams.Stream_IO;
with GNAT.SHA256;
with STUNIR_String_Builder;

package body STUNIR_Code_Index is

   use Ada.Text_IO;
   use Ada.Directories;

   function Detect_Language (File_Path : String) return Language_Kind is
      Lower : String := File_Path;
   begin
      for I in Lower'Range loop
         if Lower (I) in 'A' .. 'Z' then
            Lower (I) := Character'Val (Character'Pos (Lower (I)) + 32);
         end if;
      end loop;

      if Lower'Length >= 2 and then Lower (Lower'Last - 1 .. Lower'Last) = ".c" then
         return LANG_C;
      elsif Lower'Length >= 4 and then Lower (Lower'Last - 3 .. Lower'Last) = ".cpp" then
         return LANG_CPP;
      elsif Lower'Length >= 4 and then Lower (Lower'Last - 3 .. Lower'Last) = ".adb" then
         return LANG_ADA;
      elsif Lower'Length >= 4 and then Lower (Lower'Last - 3 .. Lower'Last) = ".ads" then
         return LANG_ADA;
      elsif Lower'Length >= 3 and then Lower (Lower'Last - 2 .. Lower'Last) = ".rs" then
         return LANG_RUST;
      elsif Lower'Length >= 3 and then Lower (Lower'Last - 2 .. Lower'Last) = ".py" then
         return LANG_PYTHON;
      elsif Lower'Length >= 5 and then Lower (Lower'Last - 4 .. Lower'Last) = ".java" then
         return LANG_JAVA;
      else
         return LANG_UNKNOWN;
      end if;
   end Detect_Language;

   procedure Compute_File_Hash
     (File_Path : String;
      Hash      : out Hash_String;
      Success   : out Boolean)
   is
      use Ada.Streams;
      use Ada.Streams.Stream_IO;
      File    : Ada.Streams.Stream_IO.File_Type;
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      Buffer  : Stream_Element_Array (1 .. 8192);
      Last    : Stream_Element_Offset;
   begin
      Hash := Hash_Strings.Null_Bounded_String;
      Success := False;

      if not Exists (File_Path) then
         return;
      end if;

      Open (File, In_File, File_Path);
      while not End_Of_File (File) loop
         Read (File, Buffer, Last);
         if Last >= Buffer'First then
            GNAT.SHA256.Update (Context, Buffer (Buffer'First .. Last));
         end if;
      end loop;
      Close (File);

      declare
         Digest : constant String := GNAT.SHA256.Digest (Context);
      begin
         Hash := Hash_Strings.To_Bounded_String (Digest);
         Success := True;
      end;

   exception
      when others =>
         if Ada.Streams.Stream_IO.Is_Open (File) then
            Close (File);
         end if;
         Hash := Hash_Strings.Null_Bounded_String;
         Success := False;
   end Compute_File_Hash;

   procedure Compute_Index_Hash
     (Index  : in out Code_Index;
      Success : out Boolean)
   is
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
   begin
      Success := False;
      for I in 1 .. Index.File_Count loop
         declare
            Path_Str : constant String := Path_Strings.To_String (Index.Files (I).Relative_Path);
            Hash_Str : constant String := Hash_Strings.To_String (Index.Files (I).SHA256);
         begin
            GNAT.SHA256.Update (Context, Path_Str);
            GNAT.SHA256.Update (Context, Hash_Str);
         end;
      end loop;
      Index.Index_Hash := Hash_Strings.To_Bounded_String (GNAT.SHA256.Digest (Context));
      Success := True;
   end Compute_Index_Hash;

   procedure Write_Index_JSON
     (Index   : Code_Index;
      Output_Path : String;
      Success : out Boolean)
   is
      use STUNIR_String_Builder;
      Builder : String_Builder;
      File_Out : Ada.Text_IO.File_Type;
      function Language_To_String (Lang : Language_Kind) return String is
      begin
         case Lang is
            when LANG_C => return "LANG_C";
            when LANG_CPP => return "LANG_CPP";
            when LANG_ADA => return "LANG_ADA";
            when LANG_RUST => return "LANG_RUST";
            when LANG_PYTHON => return "LANG_PYTHON";
            when LANG_JAVA => return "LANG_JAVA";
            when others => return "LANG_UNKNOWN";
         end case;
      end Language_To_String;
   begin
      Initialize (Builder);
      Append_Line (Builder, "{");
      Append_Line (Builder, "  ""kind"": ""stunir.code_index.v1"",");
      Append_Line (Builder, "  ""root_path"": """ & Path_Strings.To_String (Index.Root_Path) & """,");
      Append_Line (Builder, "  ""index_hash"": """ & Hash_Strings.To_String (Index.Index_Hash) & """,");
      Append_Line (Builder, "  ""file_count"": " & Natural'Image (Index.File_Count) & ",");
      Append_Line (Builder, "  ""files"": [");

      for I in 1 .. Index.File_Count loop
         if I > 1 then
            Append_Line (Builder, "    ,{");
         else
            Append_Line (Builder, "    {");
         end if;

         Append_Line (Builder, "      ""relative_path"": """ & Path_Strings.To_String (Index.Files (I).Relative_Path) & """,");
         Append_Line (Builder, "      ""absolute_path"": """ & Path_Strings.To_String (Index.Files (I).Absolute_Path) & """,");
         Append_Line (Builder, "      ""sha256"": """ & Hash_Strings.To_String (Index.Files (I).SHA256) & """,");
         Append_Line (Builder, "      ""size_bytes"": " & Natural'Image (Index.Files (I).Size_Bytes) & ",");
         Append_Line (Builder, "      ""language"": """ & Language_To_String (Index.Files (I).Language) & """");
         Append_Line (Builder, "    }");
      end loop;

      Append_Line (Builder, "  ]");
      Append_Line (Builder, "}");

      Create (File_Out, Out_File, Output_Path);
      Put (File_Out, To_String (Builder));
      Close (File_Out);
      Success := True;

   exception
      when others =>
         if Is_Open (File_Out) then
            Close (File_Out);
         end if;
         Success := False;
   end Write_Index_JSON;

   procedure Index_Directory
     (Config : Index_Config;
      Result : in out Index_Result)
   is
      Search  : Search_Type;
      Dir_Ent : Directory_Entry_Type;
      Root    : constant String := Path_Strings.To_String (Config.Input_Dir);
      Base_Len : constant Natural := Root'Length;
   begin
      Result.Status := Success;
      Result.Index.Root_Path := Config.Input_Dir;
      Result.Index.File_Count := 0;

      if not Exists (Root) then
         Result.Status := Error_Input_Not_Found;
         return;
      end if;

      Start_Search (Search, Root, "*", [Directory => True, others => True]);

      while More_Entries (Search) loop
         Get_Next_Entry (Search, Dir_Ent);
         declare
            Full : constant String := Full_Name (Dir_Ent);
         begin
            if Kind (Dir_Ent) = Ordinary_File then
               if not Exists (Full) then
                  null;
               else
                  declare
                     Rel : String := Full;
                     Hash : Hash_String;
                     Ok : Boolean;
                     Skip : Boolean := False;
                  begin
                     if Full'Length > Base_Len + 1 then
                        Rel := Full (Base_Len + 2 .. Full'Last);
                     end if;

                     if Full'Length > Max_Path_Length or else Rel'Length > Max_Path_Length then
                        Skip := True;
                     end if;

                     if not Skip then
                        Compute_File_Hash (Full, Hash, Ok);
                        if not Ok then
                           Skip := True;
                        end if;
                     end if;

                     if not Skip then
                        if Result.Index.File_Count = Max_Files then
                           Result.Status := Error_Index_Overflow;
                           exit;
                        end if;

                        Result.Index.File_Count := Result.Index.File_Count + 1;
                        declare
                           Idx : constant Positive := Result.Index.File_Count;
                        begin
                           Result.Index.Files (Idx).Relative_Path := Path_Strings.To_Bounded_String (Rel);
                           Result.Index.Files (Idx).Absolute_Path := Path_Strings.To_Bounded_String (Full);
                           Result.Index.Files (Idx).SHA256 := Hash;
                           begin
                              Result.Index.Files (Idx).Size_Bytes := Natural (Size (Full));
                           exception
                              when others =>
                                 Result.Index.Files (Idx).Size_Bytes := 0;
                           end;
                           Result.Index.Files (Idx).Language := Detect_Language (Full);
                        end;
                     end if;
                  exception
                     when others =>
                        null;
                  end;
               end if;
            end if;
         end;
      end loop;

      End_Search (Search);

      declare
         Hash_OK : Boolean;
      begin
         Compute_Index_Hash (Result.Index, Hash_OK);
         if not Hash_OK then
            Result.Status := Error_Hash_Failed;
         end if;
      end;

   exception
      when others =>
         Result.Status := Error_Output_Failed;
   end Index_Directory;

   procedure Run_Code_Index is
      Config : Index_Config;
      Result : Index_Result;
      Output_OK : Boolean;
      Arg_Count : constant Natural := Ada.Command_Line.Argument_Count;
      Input_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
      Output_Path : Ada.Strings.Unbounded.Unbounded_String := Ada.Strings.Unbounded.Null_Unbounded_String;
   begin
      for I in 1 .. Arg_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument (I);
         begin
            if Arg = "--input" and then I < Arg_Count then
               Input_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            elsif Arg = "--output" and then I < Arg_Count then
               Output_Path := Ada.Strings.Unbounded.To_Unbounded_String (Ada.Command_Line.Argument (I + 1));
            end if;
         end;
      end loop;

      if Ada.Strings.Unbounded.Length (Input_Path) = 0 or else Ada.Strings.Unbounded.Length (Output_Path) = 0 then
         Put_Line ("Usage: stunir_code_index --input <source_dir> --output <index.json>");
         return;
      end if;

      Config.Input_Dir := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Input_Path));
      Config.Output_Path := Path_Strings.To_Bounded_String (Ada.Strings.Unbounded.To_String (Output_Path));

      Index_Directory (Config, Result);
      if Result.Status /= Success then
         Put_Line ("[ERROR] Code index failed");
         return;
      end if;

      Write_Index_JSON (Result.Index, Ada.Strings.Unbounded.To_String (Output_Path), Output_OK);
      if not Output_OK then
         Put_Line ("[ERROR] Failed to write index JSON");
      end if;
   end Run_Code_Index;

end STUNIR_Code_Index;
