--  STUNIR Test Orchestrator - Implementation
--  SPARK Migration Phase 3

pragma SPARK_Mode (On);

with Stunir_Hashes; use Stunir_Hashes;

package body Test_Orchestrator is

   --  Local helper function declaration
   function Compute_Output_Hash (
      Data : Output_Buffer;
      Len  : Natural) return String;

   --  ===========================================
   --  Initialize Session
   --  ===========================================

   procedure Initialize_Session (Session : out Orchestration_Session) is
   begin
      Session := Empty_Session;
   end Initialize_Session;

   --  ===========================================
   --  Register Tool
   --  ===========================================

   procedure Register_Tool (
      Session : in out Orchestration_Session;
      Name    : in String;
      Path    : in String;
      Success : out Boolean)
   is
      Tool_Ent : Tool_Entry := Empty_Tool_Entry;
   begin
      if Session.Tool_Count = Max_Tools then
         Success := False;
         return;
      end if;

      --  Copy name
      for I in Name'Range loop
         Tool_Ent.Name (I - Name'First + 1) := Name (I);
      end loop;
      Tool_Ent.Name_Len := Name'Length;

      --  Copy path
      for I in Path'Range loop
         Tool_Ent.Path (I - Path'First + 1) := Path (I);
      end loop;
      Tool_Ent.Path_Len := Path'Length;

      --  Mark as ready (in real implementation, would check existence)
      Tool_Ent.Is_Ready := True;

      Session.Tool_Count := Session.Tool_Count + 1;
      Session.Tools (Valid_Tool_Index (Session.Tool_Count)) := Tool_Ent;
      Session.Stats.Total_Tools := Session.Stats.Total_Tools + 1;
      Session.Stats.Ready_Tools := Session.Stats.Ready_Tools + 1;
      Success := True;
   end Register_Tool;

   --  ===========================================
   --  Set Reference Tool
   --  ===========================================

   procedure Set_Reference_Tool (
      Session : in out Orchestration_Session;
      Index   : in Valid_Tool_Index) is
   begin
      Session.Reference := Tool_Index (Index);
   end Set_Reference_Tool;

   --  ===========================================
   --  Execute Tool
   --  ===========================================

   procedure Execute_Tool (
      Session : in out Orchestration_Session;
      Index   : in Valid_Tool_Index;
      Input   : in String)
   is
      Output_Rec : Tool_Output := Empty_Output;
      Tool       : Tool_Entry renames Session.Tools (Index);
   begin
      --  Copy tool name
      Output_Rec.Tool_Name := Tool.Name;
      Output_Rec.Name_Len := Tool.Name_Len;

      --  Simulate tool execution
      if not Tool.Is_Ready then
         Output_Rec.Status := Not_Found;
         Output_Rec.Exit_Code := 1;
      else
         --  Simulate successful execution
         for I in Input'Range loop
            exit when I - Input'First + 1 > Max_Output_Size;
            Output_Rec.Output (I - Input'First + 1) := Input (I);
         end loop;
         Output_Rec.Output_Len := Natural'Min (Input'Length, Max_Output_Size);

         --  Compute hash of output
         Output_Rec.Hash :=
            Compute_Output_Hash (Output_Rec.Output, Output_Rec.Output_Len);

         Output_Rec.Status := Success;
         Output_Rec.Exit_Code := 0;

         --  Update success count
         if Session.Stats.Success_Count < Natural'Last then
            Session.Stats.Success_Count := Session.Stats.Success_Count + 1;
         end if;
      end if;

      Session.Outputs (Index) := Output_Rec;
   end Execute_Tool;

   --  ===========================================
   --  Local Helper: Compute Output Hash
   --  ===========================================

   function Compute_Output_Hash (
      Data : Output_Buffer;
      Len  : Natural) return String
   is
      Hash : String (1 .. Hash_Length) := (others => '0');
      Hex_Chars : constant String := "0123456789abcdef";
      Sum : Natural := 0;
   begin
      --  Simple checksum for demonstration
      for I in 1 .. Len loop
         Sum := (Sum + Character'Pos (Data (I))) mod 256;
      end loop;

      --  Fill hash with deterministic pattern
      for I in Hash'Range loop
         Hash (I) := Hex_Chars (((Sum + I) mod 16) + 1);
      end loop;

      return Hash;
   end Compute_Output_Hash;

   --  ===========================================
   --  Execute All Tools
   --  ===========================================

   procedure Execute_All_Tools (
      Session : in out Orchestration_Session;
      Input   : in String) is
   begin
      Session.Stats.Success_Count := 0;
      Session.Stats.Error_Count := 0;
      Session.Stats.Timeout_Count := 0;

      for I in 1 .. Session.Tool_Count loop
         Execute_Tool (Session, Valid_Tool_Index (I), Input);
      end loop;

      Session.Is_Complete := True;
   end Execute_All_Tools;

   --  ===========================================
   --  Compare Outputs
   --  ===========================================

   function Compare_Outputs (
      Output1 : Tool_Output;
      Output2 : Tool_Output) return Comparison_Result is
   begin
      if Output1.Status /= Success or Output2.Status /= Success then
         return Err;
      end if;

      --  Compare hashes
      if Output1.Hash = Output2.Hash then
         return Match;
      else
         return Mismatch;
      end if;
   end Compare_Outputs;

   --  ===========================================
   --  Compare With Reference
   --  ===========================================

   function Compare_With_Reference (
      Session : Orchestration_Session;
      Index   : Valid_Tool_Index) return Comparison_Result
   is
      Ref_Output : constant Tool_Output :=
         Session.Outputs (Valid_Tool_Index (Session.Reference));
      Tool_Out   : constant Tool_Output := Session.Outputs (Index);
   begin
      return Compare_Outputs (Ref_Output, Tool_Out);
   end Compare_With_Reference;

   --  ===========================================
   --  Check Conformance
   --  ===========================================

   procedure Check_Conformance (
      Session : in Orchestration_Session;
      Result  : out Conformance_Result)
   is
      Cmp : Comparison_Result;
      Ref : Tool_Output;
      Cur : Tool_Output;
   begin
      Result := Empty_Conformance;

      if Session.Reference = 0 then
         return;
      end if;

      Result.Reference_Hash :=
         Session.Outputs (Valid_Tool_Index (Session.Reference)).Hash;

      for I in 1 .. Session.Tool_Count loop
         if I /= Session.Reference then
            Result.Total_Compared := Result.Total_Compared + 1;

            Ref := Session.Outputs (Valid_Tool_Index (Session.Reference));
            Cur := Session.Outputs (Valid_Tool_Index (I));

            if Cur.Status = Success and Ref.Status = Success then
               Cmp := Compare_Outputs (Ref, Cur);

               case Cmp is
                  when Match =>
                     Result.Matching := Result.Matching + 1;
                  when Mismatch =>
                     Result.Mismatching := Result.Mismatching + 1;
                     Result.All_Match := False;
                  when others =>
                     Result.All_Match := False;
               end case;
            else
               Result.All_Match := False;
            end if;
         end if;
      end loop;
   end Check_Conformance;

   --  ===========================================
   --  All Tools Match
   --  ===========================================

   function All_Tools_Match (
      Session : Orchestration_Session) return Boolean
   is
      Result : Conformance_Result;
   begin
      Check_Conformance (Session, Result);
      return Result.All_Match;
   end All_Tools_Match;

   --  ===========================================
   --  Get Output
   --  ===========================================

   function Get_Output (
      Session : Orchestration_Session;
      Index   : Valid_Tool_Index) return Tool_Output is
   begin
      return Session.Outputs (Index);
   end Get_Output;

   --  ===========================================
   --  Get Reference Output
   --  ===========================================

   function Get_Reference_Output (
      Session : Orchestration_Session) return Tool_Output is
   begin
      return Session.Outputs (Valid_Tool_Index (Session.Reference));
   end Get_Reference_Output;

   --  ===========================================
   --  Get Summary
   --  ===========================================

   procedure Get_Summary (
      Session : in Orchestration_Session;
      Output  : out Tool_Name_String;
      Length  : out Natural)
   is
      Prefix : constant String := "Conformance: ";
   begin
      Output := (others => ' ');
      for I in Prefix'Range loop
         Output (I - Prefix'First + 1) := Prefix (I);
      end loop;
      Length := Prefix'Length;

      pragma Unreferenced (Session);
   end Get_Summary;

   --  ===========================================
   --  Update Stats
   --  ===========================================

   procedure Update_Stats (
      Session : in out Orchestration_Session)
   is
      Ready : Natural := 0;
      Succ_Cnt : Natural := 0;
      Err_Cnt : Natural := 0;
   begin
      for I in 1 .. Session.Tool_Count loop
         if Session.Tools (Valid_Tool_Index (I)).Is_Ready then
            Ready := Ready + 1;
         end if;

         case Session.Outputs (Valid_Tool_Index (I)).Status is
            when Success =>
               Succ_Cnt := Succ_Cnt + 1;
            when Error | Timeout | Not_Found =>
               Err_Cnt := Err_Cnt + 1;
            when others =>
               null;
         end case;
      end loop;

      Session.Stats.Ready_Tools := Ready;
      Session.Stats.Success_Count := Succ_Cnt;
      Session.Stats.Error_Count := Err_Cnt;
   end Update_Stats;

   --  ===========================================
   --  Ready Tool Count
   --  ===========================================

   function Ready_Tool_Count (
      Session : Orchestration_Session) return Natural is
   begin
      return Session.Stats.Ready_Tools;
   end Ready_Tool_Count;

   --  ===========================================
   --  Success Count
   --  ===========================================

   function Success_Count (
      Session : Orchestration_Session) return Natural is
   begin
      return Session.Stats.Success_Count;
   end Success_Count;

end Test_Orchestrator;
