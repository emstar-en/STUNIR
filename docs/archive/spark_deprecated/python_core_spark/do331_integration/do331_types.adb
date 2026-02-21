--  STUNIR DO-331 Integration Types Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body DO331_Types is

   function Status_Message (Status : DO331_Status) return String is
   begin
      case Status is
         when Success             => return "Operation successful";
         when Model_Not_Found     => return "Model not found";
         when Invalid_Model       => return "Invalid model format";
         when Transform_Failed    => return "Transformation failed";
         when Coverage_Incomplete => return "Coverage data incomplete";
         when Trace_Missing       => return "Traceability link missing";
         when IO_Error            => return "I/O error occurred";
      end case;
   end Status_Message;

   function DAL_Name (Level : DAL_Level) return String is
   begin
      case Level is
         when DAL_A => return "A";
         when DAL_B => return "B";
         when DAL_C => return "C";
         when DAL_D => return "D";
         when DAL_E => return "E";
      end case;
   end DAL_Name;

   function Model_Kind_Name (Kind : Model_Kind) return String is
   begin
      case Kind is
         when Block_Model        => return "Block";
         when Activity_Model     => return "Activity";
         when StateMachine_Model => return "StateMachine";
         when Sequence_Model     => return "Sequence";
         when Requirement_Model  => return "Requirement";
         when Package_Model      => return "Package";
         when Class_Model        => return "Class";
         when Interface_Model    => return "Interface";
      end case;
   end Model_Kind_Name;

end DO331_Types;
