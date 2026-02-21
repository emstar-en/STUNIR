--  STUNIR Artifact Collector Specification
--  Collects artifacts for DO-330 compliance packages
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

with Package_Types; use Package_Types;

package Artifact_Collector is

   --  Collector Configuration
   type Collector_Config is record
      Base_Dir     : Path_String;
      Base_Dir_Len : Path_Length;
      Output_Dir   : Path_String;
      Output_Len   : Path_Length;
      Compute_Hash : Boolean;
      Verify_Files : Boolean;
   end record;

   Null_Collector_Config : constant Collector_Config := (
      Base_Dir     => (others => ' '),
      Base_Dir_Len => 0,
      Output_Dir   => (others => ' '),
      Output_Len   => 0,
      Compute_Hash => True,
      Verify_Files => True
   );

   --  Initialize collector configuration
   procedure Initialize_Config
     (Config    : out Collector_Config;
      Base_Dir  : String;
      Output_Dir : String)
   with Pre => Base_Dir'Length > 0 and Base_Dir'Length <= Max_Path_Length and
               Output_Dir'Length > 0 and Output_Dir'Length <= Max_Path_Length;

   --  Collect all artifacts from base directory
   procedure Collect_All_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   with Pre => Config.Base_Dir_Len > 0;

   --  Collect source code artifacts
   procedure Collect_Source_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   with Pre => Config.Base_Dir_Len > 0;

   --  Collect test artifacts
   procedure Collect_Test_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   with Pre => Config.Base_Dir_Len > 0;

   --  Collect proof artifacts
   procedure Collect_Proof_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   with Pre => Config.Base_Dir_Len > 0;

   --  Add artifact to package
   procedure Add_Artifact
     (Comp_Pkg : in out Compliance_Package;
      Name     : String;
      Path     : String;
      Kind     : Artifact_Kind;
      Hash     : String := "";
      Size     : Natural := 0;
      Status   : out Package_Status)
   with Pre => Name'Length > 0 and Name'Length <= Max_Name_Length and
               Path'Length > 0 and Path'Length <= Max_Path_Length and
               Hash'Length <= Max_Hash_Length and
               Comp_Pkg.Artifact_Total < Max_Artifacts,
        Post => (if Status = Success then
                  Comp_Pkg.Artifact_Total = Comp_Pkg.Artifact_Total'Old + 1);

   --  Verify artifact exists and hash matches
   function Verify_Artifact (Artifact : Artifact_Entry) return Boolean;

   --  Verify all artifacts in package
   function Verify_All_Artifacts (Comp_Pkg : Compliance_Package) return Boolean;

   --  Compute SHA-256 hash of file
   procedure Compute_File_Hash
     (Path   : String;
      Hash   : out Hash_String;
      Length : out Hash_Length;
      Status : out Package_Status)
   with Pre => Path'Length > 0 and Path'Length <= Max_Path_Length;

end Artifact_Collector;
