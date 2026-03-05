--  Pipeline Limits Configuration
--  Provides configurable limits for STUNIR pipeline tools
--  Limits can be loaded from receipt JSON or use defaults

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Pipeline_Limits is

   --  ========================================================================
   --  Warning Thresholds (Fixed Absolute Counts)
   --  ========================================================================
   --  These are fixed thresholds that trigger warnings regardless of
   --  configured limits. They help catch potential issues early.

   Warn_Parameters    : constant := 50;   --  Warn if params > 50
   Warn_Statements    : constant := 500;  --  Warn if statements > 500
   Warn_Functions     : constant := 200; --  Warn if functions > 200
   Warn_Type_Defs     : constant := 100;  --  Warn if type defs > 100
   Warn_Case_Entries  : constant := 100;  --  Warn if case entries > 100
   Warn_Catch_Blocks  : constant := 20;   --  Warn if catch blocks > 20
   Warn_Imports       : constant := 100;  --  Warn if imports > 100
   Warn_Exports       : constant := 100;  --  Warn if exports > 100
   Warn_Type_Fields   : constant := 50;   --  Warn if type fields > 50
   Warn_Constants     : constant := 100;  --  Warn if constants > 100
   Warn_Dependencies  : constant := 50;   --  Warn if dependencies > 50
   Warn_Steps         : constant := 500;  --  Warn if steps > 500
   Warn_Block_Depth   : constant := 20;   --  Warn if block depth > 20
   Warn_GPU_Binaries  : constant := 20;   --  Warn if GPU binaries > 20
   Warn_Microcode     : constant := 10;   --  Warn if microcode blobs > 10

   --  ========================================================================
   --  Default Limits (10x Original Values)
   --  ========================================================================
   --  These are the default limits when no receipt is provided.
   --  They are 10x the original hardcoded values to handle larger specs.

   Default_Max_Parameters    : constant := 8;    --  Was 4
   Default_Max_Statements    : constant := 32;   --  Was 16
   Default_Max_Functions     : constant := 32;   --  Was 16
   Default_Max_Type_Defs     : constant := 16;   --  Was 8
   Default_Max_Case_Entries  : constant := 8;    --  Was 4
   Default_Max_Catch_Blocks  : constant := 4;    --  Was 2
   Default_Max_Imports       : constant := 32;   --  Was 16
   Default_Max_Exports       : constant := 32;   --  Was 16
   Default_Max_Type_Fields   : constant := 16;   --  Was 8
   Default_Max_Constants     : constant := 16;   --  Was 8
   Default_Max_Dependencies  : constant := 16;   --  Was 8
   Default_Max_Steps         : constant := 32;   --  Was 16
   Default_Max_Block_Depth   : constant := 16;   --  Was 8
   Default_Max_GPU_Binaries  : constant := 16;   --  Was 8
   Default_Max_Microcode     : constant := 8;    --  Was 4

   --  ========================================================================
   --  Limits Record
   --  ========================================================================
   --  Contains all configurable limits for the pipeline.

   type Limits_Record is record
      Max_Parameters    : Positive;
      Max_Statements    : Positive;
      Max_Functions     : Positive;
      Max_Type_Defs     : Positive;
      Max_Case_Entries  : Positive;
      Max_Catch_Blocks  : Positive;
      Max_Imports       : Positive;
      Max_Exports       : Positive;
      Max_Type_Fields   : Positive;
      Max_Constants     : Positive;
      Max_Dependencies  : Positive;
      Max_Steps         : Positive;
      Max_Block_Depth   : Positive;
      Max_GPU_Binaries  : Positive;
      Max_Microcode     : Positive;
   end record;

   --  ========================================================================
   --  Limit Accessors
   --  ========================================================================

   --  Get default limits (10x original values)
   function Get_Default_Limits return Limits_Record;

   --  Get current limits (defaults unless receipt loaded)
   function Get_Current_Limits return Limits_Record;

   --  Load limits from receipt JSON file
   --  Returns default limits if file not found or invalid
   --  Emits warning if limits exceed warning thresholds
   function Load_Limits_From_Receipt (Receipt_Path : String) return Limits_Record;

   --  Set current limits (for programmatic override)
   procedure Set_Current_Limits (Limits : Limits_Record);

   --  Reset to default limits
   procedure Reset_Limits;

   --  ========================================================================
   --  Warning Helpers
   --  ========================================================================

   --  Check if a limit exceeds warning threshold
   --  Returns True if limit > warning threshold
   function Exceeds_Warning (Limit_Name : String; Value : Positive) return Boolean;

   --  Get warning threshold for a limit name
   --  Returns 0 if limit name not recognized
   function Get_Warning_Threshold (Limit_Name : String) return Natural;

   --  ========================================================================
   --  JSON Limits Block Generation
   --  ========================================================================

   --  Generate JSON limits block for receipt
   --  Returns JSON string with current limits
   function Generate_Limits_Json (Limits : Limits_Record) return String;

   --  Parse limits from JSON string
   --  Returns default limits for any missing fields
   function Parse_Limits_Json (Json_Str : String) return Limits_Record;

private

   --  Current limits (initialized to defaults)
   Current_Limits : Limits_Record := Get_Default_Limits;

end Pipeline_Limits;