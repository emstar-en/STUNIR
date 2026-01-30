-------------------------------------------------------------------------------
--  STUNIR Config Parser - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Config_Parser is

   --  Parse environment variable for profile
   procedure Parse_Profile_Env (
      Profile : out Build_Profile;
      Found   : out Boolean)
   is
   begin
      --  SPARK-safe stub: would check STUNIR_PROFILE env var
      Profile := Profile_Auto;
      Found := False;
   end Parse_Profile_Env;

   --  Parse configuration from environment
   procedure Parse_From_Environment (
      Config  : out Configuration;
      Success : out Boolean)
   is
      Env_Profile : Build_Profile;
      Env_Found   : Boolean;
   begin
      --  Start with defaults
      Initialize_Config (Config);
      Set_Default_Paths (Config);
      
      --  Check for profile override
      Parse_Profile_Env (Env_Profile, Env_Found);
      if Env_Found then
         Config.Profile := Env_Profile;
      end if;
      
      --  Validate
      Config.Is_Valid := Validate_Config (Config);
      Success := Config.Is_Valid;
   end Parse_From_Environment;

   --  Apply command line argument to config
   procedure Apply_Argument (
      Config : in out Configuration;
      Arg    : Parsed_Arg)
   is
   begin
      --  Handle known arguments
      if Arg.Kind = Arg_Flag then
         --  Flags
         declare
            Name_Str : constant String := 
               Arg.Name.Data (1 .. Arg.Name.Length);
         begin
            if Name_Str = "strict" then
               Config.Strict_Mode := True;
            elsif Name_Str = "verbose" or Name_Str = "v" then
               Config.Verbose := True;
            end if;
         end;
      elsif Arg.Kind = Arg_Value then
         --  Named values
         declare
            Name_Str : constant String := 
               Arg.Name.Data (1 .. Arg.Name.Length);
            Value_Str : constant String :=
               Arg.Value.Data (1 .. Arg.Value.Length);
         begin
            if Name_Str = "profile" then
               Config.Profile := String_To_Profile (Value_Str);
            elsif Name_Str = "spec-root" and Value_Str'Length <= Max_Path_String then
               Config.Spec_Root := Make_Path (Value_Str);
            elsif Name_Str = "output" and Value_Str'Length <= Max_Path_String then
               Config.Output_IR := Make_Path (Value_Str);
            end if;
         end;
      end if;
   end Apply_Argument;

   --  Apply all parsed arguments to config
   procedure Apply_All_Arguments (
      Config  : in out Configuration;
      Result  : Parse_Result)
   is
   begin
      for I in 1 .. Result.Arg_Count loop
         Apply_Argument (Config, Result.Args (I));
      end loop;
      
      --  Re-validate after applying arguments
      Config.Is_Valid := Validate_Config (Config);
   end Apply_All_Arguments;

   --  Check for flag in arguments
   function Has_Flag (
      Result : Parse_Result;
      Name   : String) return Boolean
   is
   begin
      for I in 1 .. Result.Arg_Count loop
         if Result.Args (I).Kind = Arg_Flag then
            declare
               Arg_Name : constant String :=
                  Result.Args (I).Name.Data (1 .. Result.Args (I).Name.Length);
            begin
               if Arg_Name = Name then
                  return True;
               end if;
            end;
         end if;
      end loop;
      return False;
   end Has_Flag;

   --  Get value for argument
   function Get_Value (
      Result  : Parse_Result;
      Name    : String;
      Default : String := "") return Medium_String
   is
   begin
      for I in 1 .. Result.Arg_Count loop
         if Result.Args (I).Kind = Arg_Value then
            declare
               Arg_Name : constant String :=
                  Result.Args (I).Name.Data (1 .. Result.Args (I).Name.Length);
            begin
               if Arg_Name = Name then
                  return Result.Args (I).Value;
               end if;
            end;
         end if;
      end loop;
      
      --  Return default
      if Default'Length <= Max_Medium_String then
         return Make_Medium (Default);
      else
         return Empty_Medium;
      end if;
   end Get_Value;

end Config_Parser;
