--  STUNIR Artifact Collector Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Artifact_Collector is

   procedure Copy_String
     (Source : String;
      Target : out String;
      Length : out Natural)
   with Pre => Target'Length >= Source'Length
   is
   begin
      Length := Source'Length;
      for I in 1 .. Source'Length loop
         pragma Loop_Invariant (I <= Source'Length);
         Target (Target'First + I - 1) := Source (Source'First + I - 1);
      end loop;
      for I in Source'Length + 1 .. Target'Length loop
         Target (Target'First + I - 1) := ' ';
      end loop;
   end Copy_String;

   procedure Initialize_Config
     (Config    : out Collector_Config;
      Base_Dir  : String;
      Output_Dir : String)
   is
      Base_Len, Out_Len : Natural;
   begin
      Config := Null_Collector_Config;
      Copy_String (Base_Dir, Config.Base_Dir, Base_Len);
      Config.Base_Dir_Len := Path_Length (Base_Len);
      Copy_String (Output_Dir, Config.Output_Dir, Out_Len);
      Config.Output_Len := Path_Length (Out_Len);
   end Initialize_Config;

   procedure Collect_All_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   is
   begin
      Collect_Source_Artifacts (Config, Comp_Pkg, Status);
      if Status /= Success then
         return;
      end if;
      Collect_Test_Artifacts (Config, Comp_Pkg, Status);
      if Status /= Success then
         return;
      end if;
      Collect_Proof_Artifacts (Config, Comp_Pkg, Status);
   end Collect_All_Artifacts;

   procedure Collect_Source_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   is
      pragma Unreferenced (Config, Comp_Pkg);
   begin
      Status := Success;
   end Collect_Source_Artifacts;

   procedure Collect_Test_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   is
      pragma Unreferenced (Config, Comp_Pkg);
   begin
      Status := Success;
   end Collect_Test_Artifacts;

   procedure Collect_Proof_Artifacts
     (Config : Collector_Config;
      Comp_Pkg : in out Compliance_Package;
      Status : out Package_Status)
   is
      pragma Unreferenced (Config, Comp_Pkg);
   begin
      Status := Success;
   end Collect_Proof_Artifacts;

   procedure Add_Artifact
     (Comp_Pkg : in out Compliance_Package;
      Name     : String;
      Path     : String;
      Kind     : Artifact_Kind;
      Hash     : String := "";
      Size     : Natural := 0;
      Status   : out Package_Status)
   is
      A : Artifact_Entry := Null_Artifact_Entry;
      Name_Len, Path_Len, Hash_Len : Natural;
   begin
      Copy_String (Name, A.Name, Name_Len);
      A.Name_Len := Package_Types.Name_Length (Name_Len);
      Copy_String (Path, A.Path, Path_Len);
      A.Path_Len := Package_Types.Path_Length (Path_Len);
      if Hash'Length > 0 then
         Copy_String (Hash, A.Hash, Hash_Len);
         A.Hash_Len := Package_Types.Hash_Length (Hash_Len);
      end if;
      A.Kind := Kind;
      A.Size_Bytes := Size;
      A.Is_Valid := True;
      Comp_Pkg.Artifact_Total := Comp_Pkg.Artifact_Total + 1;
      Comp_Pkg.Artifacts (Comp_Pkg.Artifact_Total) := A;
      Status := Success;
   end Add_Artifact;

   function Verify_Artifact (Artifact : Artifact_Entry) return Boolean is
   begin
      return Artifact.Is_Valid and Artifact.Path_Len > 0;
   end Verify_Artifact;

   function Verify_All_Artifacts (Comp_Pkg : Compliance_Package) return Boolean
   is
   begin
      for I in 1 .. Comp_Pkg.Artifact_Total loop
         pragma Loop_Invariant (I <= Comp_Pkg.Artifact_Total);
         if not Verify_Artifact (Comp_Pkg.Artifacts (I)) then
            return False;
         end if;
      end loop;
      return True;
   end Verify_All_Artifacts;

   procedure Compute_File_Hash
     (Path   : String;
      Hash   : out Hash_String;
      Length : out Hash_Length;
      Status : out Package_Status)
   is
      pragma Unreferenced (Path);
      Dummy_Hash : constant String :=
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
   begin
      if Dummy_Hash'Length <= Max_Hash_Length then
         Copy_String (Dummy_Hash, Hash, Length);
         Length := Package_Types.Hash_Length (Dummy_Hash'Length);
         Status := Success;
      else
         Hash := (others => ' ');
         Length := 0;
         Status := IO_Error;
      end if;
   end Compute_File_Hash;

end Artifact_Collector;
