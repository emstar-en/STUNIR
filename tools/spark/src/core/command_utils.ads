--  command_utils - Utility package for command execution
--  Provides helper functions for running shell commands

pragma SPARK_Mode (Off);

with GNAT.Strings;

package Command_Utils is

   --  Execute a command with input and return output
   --  Cmd: The command to execute (will be parsed with sh -c)
   --  Input: Input to provide to the command's stdin
   --  Success: Set to True if command succeeded, False otherwise
   --  Returns: The command's stdout output
   function Get_Command_Output
     (Cmd    : String;
      Input  : String;
      Success : access Boolean) return String;

end Command_Utils;