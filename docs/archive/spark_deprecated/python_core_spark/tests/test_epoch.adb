-------------------------------------------------------------------------------
--  STUNIR Epoch Manager Tests
--  Part of Phase 2 SPARK Migration
-------------------------------------------------------------------------------

with Ada.Text_IO; use Ada.Text_IO;
with Stunir_Strings; use Stunir_Strings;
with Stunir_Hashes;  use Stunir_Hashes;
with Epoch_Types;    use Epoch_Types;
with Epoch_Selector; use Epoch_Selector;

procedure Test_Epoch is
   
   Tests_Passed : Natural := 0;
   Tests_Failed : Natural := 0;
   
   procedure Assert (Condition : Boolean; Name : String) is
   begin
      if Condition then
         Put_Line ("  [PASS] " & Name);
         Tests_Passed := Tests_Passed + 1;
      else
         Put_Line ("  [FAIL] " & Name);
         Tests_Failed := Tests_Failed + 1;
      end if;
   end Assert;
   
begin
   Put_Line ("=== Epoch Manager Tests ===");
   Put_Line ("");
   
   --  Test 1: Epoch source to string
   Put_Line ("Test Suite: Epoch Source Conversion");
   Assert (Source_To_String (Source_Zero) = "ZERO", "Zero source string");
   Assert (Source_To_String (Source_Env_Build_Epoch) = "STUNIR_BUILD_EPOCH", 
           "Build epoch source string");
   Assert (Source_To_String (Source_Derived_Spec_Digest) = "DERIVED_SPEC_DIGEST_V1",
           "Derived spec digest source string");
   Put_Line ("");
   
   --  Test 2: String to epoch source
   Put_Line ("Test Suite: String to Source Parsing");
   Assert (String_To_Source ("ZERO") = Source_Zero, "Parse zero");
   Assert (String_To_Source ("UNKNOWN_VALUE") = Source_Unknown, "Parse unknown");
   Assert (String_To_Source ("SOURCE_DATE_EPOCH") = Source_Env_Source_Date,
           "Parse source date epoch");
   Put_Line ("");
   
   --  Test 3: Epoch value parsing
   Put_Line ("Test Suite: Epoch Value Parsing");
   Assert (Parse_Epoch_Value ("0") = 0, "Parse zero value");
   Assert (Parse_Epoch_Value ("123") = 123, "Parse simple value");
   Assert (Parse_Epoch_Value ("1234567890") = 1234567890, "Parse timestamp");
   Put_Line ("");
   
   --  Test 4: Derive epoch from digest
   Put_Line ("Test Suite: Epoch From Digest");
   declare
      Digest : Hash_Hex := Zero_Hash;
   begin
      Digest.Data (1 .. 8) := "00000001";
      Assert (Derive_Epoch_From_Digest (Digest) = 1, "Derive from 00000001");
      
      Digest.Data (1 .. 8) := "0000000F";
      Assert (Derive_Epoch_From_Digest (Digest) = 15, "Derive from 0000000F");
      
      Digest.Data (1 .. 8) := "FFFFFFFF";
      Assert (Derive_Epoch_From_Digest (Digest) = 4294967295, "Derive from FFFFFFFF");
   end;
   Put_Line ("");
   
   --  Test 5: Epoch selection
   Put_Line ("Test Suite: Epoch Selection");
   declare
      Selection : Epoch_Selection;
      Spec_Root : Path_String := Make_Path ("spec");
   begin
      Select_Epoch (Spec_Root, False, Selection);
      Assert (Selection.Source /= Source_Unknown, "Selection has valid source");
      Assert (Selection.Is_Deterministic, "Selection is deterministic");
   end;
   Put_Line ("");
   
   --  Test 6: Is deterministic source
   Put_Line ("Test Suite: Deterministic Source Check");
   Assert (Is_Deterministic_Source (Source_Zero), "Zero is deterministic");
   Assert (Is_Deterministic_Source (Source_Derived_Spec_Digest), 
           "Derived is deterministic");
   Assert (not Is_Deterministic_Source (Source_Current_Time),
           "Current time is not deterministic");
   Put_Line ("");
   
   --  Summary
   Put_Line ("=== Summary ===");
   Put_Line ("Tests Passed:" & Natural'Image (Tests_Passed));
   Put_Line ("Tests Failed:" & Natural'Image (Tests_Failed));
   
end Test_Epoch;
