--  STUNIR DO-330 Template Types Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Templates is

   --  ============================================================
   --  TQL_To_String
   --  ============================================================

   function TQL_To_String (Level : TQL_Level) return String is
   begin
      case Level is
         when TQL_1 => return "TQL-1";
         when TQL_2 => return "TQL-2";
         when TQL_3 => return "TQL-3";
         when TQL_4 => return "TQL-4";
         when TQL_5 => return "TQL-5";
      end case;
   end TQL_To_String;

   --  ============================================================
   --  DAL_To_String
   --  ============================================================

   function DAL_To_String (Level : DAL_Level) return String is
   begin
      case Level is
         when DAL_A => return "DAL-A";
         when DAL_B => return "DAL-B";
         when DAL_C => return "DAL-C";
         when DAL_D => return "DAL-D";
         when DAL_E => return "DAL-E";
      end case;
   end DAL_To_String;

   --  ============================================================
   --  Method_To_String
   --  ============================================================

   function Method_To_String (Method : Verification_Method) return String is
   begin
      case Method is
         when Test     => return "Test";
         when Analysis => return "Analysis";
         when Review   => return "Review";
         when Formal   => return "Formal";
      end case;
   end Method_To_String;

   --  ============================================================
   --  Status_To_String
   --  ============================================================

   function Status_To_String (Status : Test_Status) return String is
   begin
      case Status is
         when Not_Run  => return "Not_Run";
         when Passed   => return "Passed";
         when Failed   => return "Failed";
         when Blocked  => return "Blocked";
         when Skipped  => return "Skipped";
      end case;
   end Status_To_String;

   --  ============================================================
   --  Kind_To_Filename
   --  ============================================================

   function Kind_To_Filename (Kind : Template_Kind) return String is
   begin
      case Kind is
         when TOR_Template           => return "TOR_template.txt";
         when TQP_Template           => return "TQP_template.txt";
         when TAS_Template           => return "TAS_template.txt";
         when VCP_Template           => return "verification_template.txt";
         when CI_Template            => return "config_index_template.txt";
         when Traceability_Template  => return "traceability_template.txt";
         when Problem_Report_Template => return "problem_report_template.txt";
      end case;
   end Kind_To_Filename;

   --  ============================================================
   --  Is_Valid_Requirement_ID
   --  Validates format: TOR-XXX-NNN or similar
   --  ============================================================

   function Is_Valid_Requirement_ID (ID : String) return Boolean is
      Has_Dash   : Boolean := False;
      Has_Letter : Boolean := False;
      Has_Digit  : Boolean := False;
   begin
      if ID'Length < 3 or ID'Length > 32 then
         return False;
      end if;

      for I in ID'Range loop
         if ID (I) = '-' then
            Has_Dash := True;
         elsif ID (I) in 'A' .. 'Z' | 'a' .. 'z' then
            Has_Letter := True;
         elsif ID (I) in '0' .. '9' then
            Has_Digit := True;
         elsif ID (I) /= '_' then
            --  Invalid character
            return False;
         end if;
      end loop;

      return Has_Letter and Has_Dash;
   end Is_Valid_Requirement_ID;

   --  ============================================================
   --  Is_Valid_Test_Case_ID
   --  Validates format: TC-NNN or TC_XXX_NNN
   --  ============================================================

   function Is_Valid_Test_Case_ID (ID : String) return Boolean is
      Has_TC     : Boolean := False;
      Has_Number : Boolean := False;
   begin
      if ID'Length < 4 or ID'Length > 32 then
         return False;
      end if;

      --  Check for TC prefix
      if ID'Length >= 2 then
         if (ID (ID'First) = 'T' or ID (ID'First) = 't') and
            (ID (ID'First + 1) = 'C' or ID (ID'First + 1) = 'c')
         then
            Has_TC := True;
         end if;
      end if;

      --  Check for at least one digit
      for I in ID'Range loop
         if ID (I) in '0' .. '9' then
            Has_Number := True;
            exit;
         end if;
      end loop;

      return Has_TC and Has_Number;
   end Is_Valid_Test_Case_ID;

end Templates;
