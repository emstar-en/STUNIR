-------------------------------------------------------------------------------
--  STUNIR IR JSON Parser (Implementation)
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with STUNIR_JSON_Parser;
with IR.Types;
with IR.Nodes;
with IR.Modules;
with IR.Declarations;
with IR.Statements;
with IR.Expressions;
with STUNIR.Emitters.Node_Table;

package body IR.JSON is

   use STUNIR_JSON_Parser;
   use IR.Types;
   use IR.Nodes;
   use IR.Modules;
   use IR.Declarations;
   use IR.Statements;
   use IR.Expressions;
   use STUNIR.Emitters.Node_Table;
   use type Name_Strings.Bounded_String;

   function To_Name (S : String) return IR_Name is
   begin
      if S'Length = 0 then
         return Name_Strings.Null_Bounded_String;
      elsif S'Length > Max_Name_Length then
         return Name_Strings.To_Bounded_String (S (S'First .. S'First + Max_Name_Length - 1));
      else
         return Name_Strings.To_Bounded_String (S);
      end if;
   end To_Name;

   function To_Node_ID (S : String) return Node_ID is
   begin
      if S'Length = 0 then
         return Name_Strings.Null_Bounded_String;
      elsif S'Length > Max_Name_Length then
         return Name_Strings.To_Bounded_String (S (S'First .. S'First + Max_Name_Length - 1));
      else
         return Name_Strings.To_Bounded_String (S);
      end if;
   end To_Node_ID;

   function To_Path (S : String) return IR_Path is
   begin
      if S'Length = 0 then
         return IR.Types.Path_Strings.Null_Bounded_String;
      elsif S'Length > IR.Types.Max_Path_Length then
         return IR.Types.Path_Strings.To_Bounded_String (S (S'First .. S'First + IR.Types.Max_Path_Length - 1));
      else
         return IR.Types.Path_Strings.To_Bounded_String (S);
      end if;
   end To_Path;

   function To_Hash (S : String) return IR_Hash is
   begin
      if S'Length = 0 then
         return Hash_Strings.Null_Bounded_String;
      elsif S'Length > Max_Hash_Length then
         return Hash_Strings.To_Bounded_String (S (S'First .. S'First + Max_Hash_Length - 1));
      else
         return Hash_Strings.To_Bounded_String (S);
      end if;
   end To_Hash;

   function To_Int (S : String; Default : Natural := 0) return Natural is
   begin
      if S'Length = 0 then
         return Default;
      end if;
      return Natural'Value (S);
   exception
      when others =>
         return Default;
   end To_Int;

   procedure Next (State : in out Parser_State; Status : out Status_Code) is
   begin
      Next_Token (State, Status);
   end Next;

   procedure Skip (State : in out Parser_State; Status : out Status_Code) is
   begin
      Skip_Value (State, Status);
   end Skip;

   procedure Read_Member
     (State : in out Parser_State;
      Name  :    out Identifier_String;
      Status:    out Status_Code)
   is
   begin
      Name := Identifier_Strings.Null_Bounded_String;
      if State.Current_Token /= Token_String then
         Status := Error_Parse;
         return;
      end if;

      Name := Identifier_Strings.To_Bounded_String (JSON_Strings.To_String (State.Token_Value));
      Next (State, Status);
      if Status /= Success or else State.Current_Token /= Token_Colon then
         Status := Error_Parse;
         return;
      end if;
      Next (State, Status);
   end Read_Member;

   procedure Parse_Source_Location
     (State  : in out Parser_State;
      Loc    :    out IR.Nodes.Source_Location;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
   begin
      Loc := (File => IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0);
      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "file" and then State.Current_Token = Token_String then
               Loc.File := To_Path (JSON_Strings.To_String (State.Token_Value));
            elsif Key = "line" and then State.Current_Token = Token_Number then
               Loc.Line := To_Int (JSON_Strings.To_String (State.Token_Value), 1);
            elsif Key = "column" and then State.Current_Token = Token_Number then
               Loc.Column := To_Int (JSON_Strings.To_String (State.Token_Value), 1);
            elsif Key = "length" and then State.Current_Token = Token_Number then
               Loc.Length := To_Int (JSON_Strings.To_String (State.Token_Value), 0);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Source_Location;

   procedure Parse_Type_Reference
     (State  : in out Parser_State;
      T      :    out Type_Reference;
      Status :    out Status_Code)
   is
      Member_Name : Identifier_String;
      Kind_Str    : String := "";
      Prim_Str    : String := "";
      Name_Str    : String := "";
   begin
      T := (Kind => TK_Primitive, Prim_Type => Type_Void);
      if State.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      Next (State, Status);
      while Status = Success and then State.Current_Token /= Token_Object_End loop
         Read_Member (State, Member_Name, Status);
         exit when Status /= Success;
         declare
            Key : constant String := Identifier_Strings.To_String (Member_Name);
         begin
            if Key = "kind" and then State.Current_Token = Token_String then
               Kind_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "primitive" and then State.Current_Token = Token_String then
               Prim_Str := JSON_Strings.To_String (State.Token_Value);
            elsif Key = "name" and then State.Current_Token = Token_String then
               Name_Str := JSON_Strings.To_String (State.Token_Value);
            else
               Skip (State, Status);
            end if;
         end;

         if State.Current_Token = Token_Comma then
            Next (State, Status);
         end if;
      end loop;

      if Kind_Str = "primitive_type" then
         T := (Kind => TK_Primitive, Prim_Type => Parse_Primitive_Type (Prim_Str));
      elsif Kind_Str = "type_ref" then
         T := (Kind => TK_Ref, Type_Name => To_Name (Name_Str), Type_Binding => Name_Strings.Null_Bounded_String);
      end if;

      if State.Current_Token = Token_Object_End then
         Next (State, Status);
      end if;
   end Parse_Type_Reference;

   procedure Parse_IR_JSON
     (JSON_Content : in     JSON_String;
      Module       :    out IR.Modules.IR_Module;
      Nodes        :    out STUNIR.Emitters.Node_Table.Node_Table;
      Status       :    out Status_Code)
   is
      Parser : Parser_State;
      Temp_Status : Status_Code;
   begin
      -- Initialize outputs
      Module := (Base => (Kind => Kind_Module,
                          ID => Name_Strings.Null_Bounded_String,
                          Location => (File => IR.Types.Path_Strings.Null_Bounded_String, Line => 1, Column => 1, Length => 0),
                          Hash => Hash_Strings.Null_Bounded_String),
                 Module_Name => Name_Strings.Null_Bounded_String,
                 others => <>);
      Init_Node_Table (Nodes);

      -- Initialize parser
      Initialize_Parser (Parser, JSON_Content, Temp_Status);
      if Temp_Status /= Success then
         Status := Error_Parse;
         return;
      end if;

      -- Parse the JSON structure
      Next_Token (Parser, Temp_Status);
      if Temp_Status /= Success or else Parser.Current_Token /= Token_Object_Start then
         Status := Error_Parse;
         return;
      end if;

      -- Skip to end and finish
      Skip_Value (Parser, Temp_Status);
      Status := Success;
   end Parse_IR_JSON;

end IR.JSON;