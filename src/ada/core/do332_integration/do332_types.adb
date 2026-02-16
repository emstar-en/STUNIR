--  STUNIR DO-332 Integration Types Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body DO332_Types is

   function Status_Message (Status : DO332_Status) return String is
   begin
      case Status is
         when Success            => return "Analysis successful";
         when Class_Not_Found    => return "Class not found";
         when Invalid_Hierarchy  => return "Invalid class hierarchy";
         when Depth_Exceeded     => return "Inheritance depth exceeded limit";
         when Unsafe_Polymorphism=> return "Unsafe polymorphic call detected";
         when High_Coupling      => return "High coupling detected";
         when Analysis_Failed    => return "OOP analysis failed";
         when IO_Error           => return "I/O error occurred";
      end case;
   end Status_Message;

   function Inheritance_Name (Kind : Inheritance_Kind) return String is
   begin
      case Kind is
         when Single_Inheritance       => return "single";
         when Multiple_Inheritance     => return "multiple";
         when Interface_Implementation => return "interface";
         when No_Inheritance           => return "none";
      end case;
   end Inheritance_Name;

   function Method_Kind_Name (Kind : Method_Kind) return String is
   begin
      case Kind is
         when Virtual_Method     => return "virtual";
         when Non_Virtual_Method => return "non-virtual";
         when Abstract_Method    => return "abstract";
         when Override_Method    => return "override";
         when Static_Method      => return "static";
      end case;
   end Method_Kind_Name;

end DO332_Types;
