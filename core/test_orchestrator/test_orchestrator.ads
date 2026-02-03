--  STUNIR Test Orchestrator
--  SPARK Migration Phase 3 - Test Infrastructure
--  Coordinates cross-tool interoperability testing

pragma SPARK_Mode (On);

with Orchestrator_Types; use Orchestrator_Types;

package Test_Orchestrator is

   --  ===========================================
   --  Session Management
   --  ===========================================

   --  Initialize orchestration session
   procedure Initialize_Session (Session : out Orchestration_Session)
      with Post => Session.Tool_Count = 0 and not Session.Is_Complete;

   --  Register a tool for orchestration
   procedure Register_Tool (
      Session : in out Orchestration_Session;
      Name    : in String;
      Path    : in String;
      Success : out Boolean)
      with Pre  => Session.Tool_Count < Max_Tools and
                   Name'Length > 0 and Name'Length <= Max_Tool_Name and
                   Path'Length > 0 and Path'Length <= Max_Tool_Name,
           Post => (if Success then
                       Session.Tool_Count = Session.Tool_Count'Old + 1
                    else Session.Tool_Count = Session.Tool_Count'Old);

   --  Set reference tool for comparison
   procedure Set_Reference_Tool (
      Session : in out Orchestration_Session;
      Index   : in Valid_Tool_Index)
      with Pre  => Index <= Session.Tool_Count,
           Post => Session.Reference = Tool_Index (Index);

   --  ===========================================
   --  Tool Execution
   --  ===========================================

   --  Execute single tool
   procedure Execute_Tool (
      Session : in out Orchestration_Session;
      Index   : in Valid_Tool_Index;
      Input   : in String)
      with Pre => Index <= Session.Tool_Count and Input'Length > 0;

   --  Execute all registered tools
   procedure Execute_All_Tools (
      Session : in out Orchestration_Session;
      Input   : in String)
      with Pre  => Session.Tool_Count > 0 and Input'Length > 0,
           Post => Session.Is_Complete;

   --  ===========================================
   --  Output Comparison
   --  ===========================================

   --  Compare two tool outputs
   function Compare_Outputs (
      Output1 : Tool_Output;
      Output2 : Tool_Output) return Comparison_Result
      with Pre => Output1.Status = Success and Output2.Status = Success;

   --  Compare output with reference
   function Compare_With_Reference (
      Session : Orchestration_Session;
      Index   : Valid_Tool_Index) return Comparison_Result
      with Pre => Session.Reference > 0 and
                  Index <= Session.Tool_Count and
                  Index /= Valid_Tool_Index (Session.Reference);

   --  Check conformance of all tools
   procedure Check_Conformance (
      Session : in Orchestration_Session;
      Result  : out Conformance_Result)
      with Pre  => Session.Is_Complete and Session.Reference > 0,
           Post => Result.Total_Compared <= Natural (Session.Tool_Count);

   --  ===========================================
   --  Reporting
   --  ===========================================

   --  Check if all tools match reference
   function All_Tools_Match (
      Session : Orchestration_Session) return Boolean
      with Pre => Session.Is_Complete and Session.Reference > 0;

   --  Get tool output by index
   function Get_Output (
      Session : Orchestration_Session;
      Index   : Valid_Tool_Index) return Tool_Output
      with Pre => Index <= Session.Tool_Count;

   --  Get reference output
   function Get_Reference_Output (
      Session : Orchestration_Session) return Tool_Output
      with Pre => Session.Reference > 0;

   --  Get conformance summary
   procedure Get_Summary (
      Session : in Orchestration_Session;
      Output  : out Tool_Name_String;
      Length  : out Natural)
      with Post => Length <= Max_Tool_Name;

   --  ===========================================
   --  Statistics
   --  ===========================================

   --  Update session statistics
   procedure Update_Stats (
      Session : in out Orchestration_Session);

   --  Get ready tool count
   function Ready_Tool_Count (
      Session : Orchestration_Session) return Natural;

   --  Get success count
   function Success_Count (
      Session : Orchestration_Session) return Natural;

end Test_Orchestrator;
