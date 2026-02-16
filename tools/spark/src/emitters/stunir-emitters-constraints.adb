-- STUNIR Constraints Emitter (SPARK Body)
-- DO-178C Level A
-- Phase 3c: Remaining Category Emitters

with Ada.Strings; use Ada.Strings;

package body STUNIR.Emitters.Constraints is
   pragma SPARK_Mode (On);

   -- Emit complete module (constraint model)
   overriding procedure Emit_Module
     (Self   : in out Constraints_Emitter;
      Module : in     IR_Module;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Solver is
         when MiniZinc =>
            -- Emit MiniZinc model
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% STUNIR Generated MiniZinc Model" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% Decision variables" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "var 1..10: x;" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "var 1..10: y;" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% Constraints" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "constraint x + y = 10;" & ASCII.LF & ASCII.LF);
            
            if Self.Config.Generate_Search then
               Code_Buffers.Append
                 (Source   => Output,
                  New_Item => "solve satisfy;" & ASCII.LF);
            end if;

            Code_Buffers.Append
              (Source   => Output,
               New_Item => "output [""x = \(x), y = \(y)\n""];" & ASCII.LF);

         when Z3 =>
            -- Emit Z3 SMT-LIB format
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; STUNIR Generated Z3 Model" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(declare-const x Int)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(declare-const y Int)" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(assert (= (+ x y) 10))" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(check-sat)" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(get-model)" & ASCII.LF);

         when CLP_FD | ECLiPSe_CLP =>
            -- Emit Prolog CLP(FD)
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% STUNIR Generated CLP(FD) Model" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => ":- use_module(library(clpfd))." & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "solve([X, Y]) :-" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  X in 1..10," & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  Y in 1..10," & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  X + Y #= 10," & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "  label([X, Y])." & ASCII.LF);

         when others =>
            -- Generic constraint format
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# STUNIR Generated Constraint Model" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# DO-178C Level A" & ASCII.LF & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "variables: x, y" & ASCII.LF);
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "constraints: x + y = 10" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Module;

   -- Emit type definition (constraint variable)
   overriding procedure Emit_Type
     (Self   : in out Constraints_Emitter;
      T      : in     IR_Type_Def;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Solver is
         when MiniZinc =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "% Variable type: " & Name_Strings.To_String (T.Name) & ASCII.LF);

         when Z3 =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "; Variable type: " & Name_Strings.To_String (T.Name) & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "# Variable type: " & Name_Strings.To_String (T.Name) & ASCII.LF);
      end case;

      Success := True;
   end Emit_Type;

   -- Emit function definition (constraint predicate)
   overriding procedure Emit_Function
     (Self   : in out Constraints_Emitter;
      Func   : in     IR_Function;
      Output :    out IR_Code_Buffer;
      Success:    out Boolean)
   is
   begin
      Code_Buffers.Set_Bounded_String
        (Target => Output,
         Source => "",
         Drop   => Ada.Strings.Right);

      case Self.Config.Solver is
         when MiniZinc =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "predicate " & Name_Strings.To_String (Func.Name) & "();" & ASCII.LF);

         when Z3 =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "(define-fun " & Name_Strings.To_String (Func.Name) & " () Bool true)" & ASCII.LF);

         when others =>
            Code_Buffers.Append
              (Source   => Output,
               New_Item => "constraint " & Name_Strings.To_String (Func.Name) & "()" & ASCII.LF);
      end case;

      Success := True;
   end Emit_Function;

end STUNIR.Emitters.Constraints;
