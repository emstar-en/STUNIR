--  command_utils - Utility package for command execution

pragma SPARK_Mode (Off);

with Ada.Text_IO;
with Ada.Strings.Unbounded;
with GNAT.OS_Lib;

package body Command_Utils is

   function Get_Command_Output
     (Cmd     : String;
      Input   : String;
      Success : access Boolean) return String
   is
      use GNAT.OS_Lib;
      use Ada.Strings.Unbounded;
      
      Args      : Argument_List_Access;
      Temp_Out  : File_Descriptor;
      Temp_Name : String := "cmd_output.tmp";
      Result    : Unbounded_String := Null_Unbounded_String;
      Status    : Integer;
      File      : Ada.Text_IO.File_Type;
      Line      : String (1 .. 1024);
      Last      : Natural;
   begin
      Success.all := False;
      
      --  Create temporary file for output
      Temp_Out := Create_File (Temp_Name, Binary);
      if Temp_Out = Invalid_FD then
         return "";
      end if;
      Close (Temp_Out);
      
      --  Build command with output redirection
      declare
         Full_Cmd : constant String := Cmd & " > " & Temp_Name & " 2>&1";
      begin
         Args := Argument_String_To_List ("/bin/sh -c '" & Full_Cmd & "'");
      end;
      
      --  Spawn the process
      Status := Spawn ("sh", Args.all);
      Free (Args);
      
      --  Read output from temp file
      begin
         Ada.Text_IO.Open (File, Ada.Text_IO.In_File, Temp_Name);
         while not Ada.Text_IO.End_Of_File (File) loop
            Ada.Text_IO.Get_Line (File, Line, Last);
            if Result /= Null_Unbounded_String then
               Append (Result, ASCII.LF);
            end if;
            Append (Result, Line (1 .. Last));
         end loop;
         Ada.Text_IO.Close (File);
         Delete_File (Temp_Name, Success.all);
         Success.all := (Status = 0);
      exception
         when others =>
            if Ada.Text_IO.Is_Open (File) then
               Ada.Text_IO.Close (File);
            end if;
            Delete_File (Temp_Name, Success.all);
            return "";
      end;
      
      return To_String (Result);
   end Get_Command_Output;

end Command_Utils;