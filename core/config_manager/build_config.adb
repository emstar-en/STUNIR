-------------------------------------------------------------------------------
--  STUNIR Build Configuration - Ada SPARK Implementation
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package body Build_Config is

   --  Initialize configuration with defaults
   procedure Initialize_Config (Config : out Configuration) is
   begin
      Config := Default_Config;
   end Initialize_Config;

   --  Set default paths based on conventional layout
   procedure Set_Default_Paths (Config : in out Configuration) is
   begin
      --  spec/
      Config.Spec_Root := Make_Path ("spec");
      
      --  asm/spec_ir.json
      Config.Output_IR := Make_Path ("asm/spec_ir.json");
      
      --  asm/output.py
      Config.Output_Code := Make_Path ("asm/output.py");
      
      --  local_toolchain.lock.json
      Config.Lock_File := Make_Path ("local_toolchain.lock.json");
      
      --  Native binary path
      Config.Native_Binary := Make_Path (
         "tools/native/rust/stunir-native/target/release/stunir-native");
   end Set_Default_Paths;

   --  Validate configuration
   function Validate_Config (Config : Configuration) return Boolean is
   begin
      --  Must have spec root
      if Config.Spec_Root.Length = 0 then
         return False;
      end if;
      
      --  Must have output IR path
      if Config.Output_IR.Length = 0 then
         return False;
      end if;
      
      return True;
   end Validate_Config;

   --  Profile string conversion
   function Profile_To_String (P : Build_Profile) return String is
   begin
      case P is
         when Profile_Auto   => return "auto";
         when Profile_Native => return "native";
         when Profile_Python => return "python";
         when Profile_Shell  => return "shell";
      end case;
   end Profile_To_String;

   function String_To_Profile (S : String) return Build_Profile is
   begin
      if S = "native" then
         return Profile_Native;
      elsif S = "python" then
         return Profile_Python;
      elsif S = "shell" then
         return Profile_Shell;
      else
         return Profile_Auto;
      end if;
   end String_To_Profile;

   --  Phase string conversion
   function Phase_To_String (P : Build_Phase) return String is
   begin
      case P is
         when Phase_Discovery => return "discovery";
         when Phase_Epoch     => return "epoch";
         when Phase_Spec_Parse => return "spec_parse";
         when Phase_IR_Emit   => return "ir_emit";
         when Phase_Code_Gen  => return "code_gen";
         when Phase_Compile   => return "compile";
         when Phase_Receipt   => return "receipt";
         when Phase_Verify    => return "verify";
      end case;
   end Phase_To_String;

end Build_Config;
