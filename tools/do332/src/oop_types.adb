--  STUNIR DO-332 OOP Types Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body OOP_Types is

   --  ============================================================
   --  DAL Requirements Implementation
   --  ============================================================

   function Get_DAL_Requirements (Level : DAL_Level) return DAL_Requirements is
   begin
      case Level is
         when DAL_A =>
            return (
               Requires_Inheritance_Analysis  => True,
               Requires_Diamond_Detection     => True,
               Requires_Polymorphism_Analysis => True,
               Requires_LSP_Checking          => True,
               Requires_Dispatch_Analysis     => True,
               Requires_Dispatch_Timing       => True,
               Requires_Coupling_Analysis     => True,
               Requires_Lifecycle_Analysis    => True
            );

         when DAL_B =>
            return (
               Requires_Inheritance_Analysis  => True,
               Requires_Diamond_Detection     => True,
               Requires_Polymorphism_Analysis => True,
               Requires_LSP_Checking          => True,
               Requires_Dispatch_Analysis     => True,
               Requires_Dispatch_Timing       => False,
               Requires_Coupling_Analysis     => True,
               Requires_Lifecycle_Analysis    => True
            );

         when DAL_C =>
            return (
               Requires_Inheritance_Analysis  => True,
               Requires_Diamond_Detection     => True,
               Requires_Polymorphism_Analysis => True,
               Requires_LSP_Checking          => False,
               Requires_Dispatch_Analysis     => False,
               Requires_Dispatch_Timing       => False,
               Requires_Coupling_Analysis     => True,
               Requires_Lifecycle_Analysis    => True
            );

         when DAL_D =>
            return (
               Requires_Inheritance_Analysis  => True,
               Requires_Diamond_Detection     => False,
               Requires_Polymorphism_Analysis => True,
               Requires_LSP_Checking          => False,
               Requires_Dispatch_Analysis     => False,
               Requires_Dispatch_Timing       => False,
               Requires_Coupling_Analysis     => False,
               Requires_Lifecycle_Analysis    => False
            );

         when DAL_E =>
            return (
               Requires_Inheritance_Analysis  => False,
               Requires_Diamond_Detection     => False,
               Requires_Polymorphism_Analysis => False,
               Requires_LSP_Checking          => False,
               Requires_Dispatch_Analysis     => False,
               Requires_Dispatch_Timing       => False,
               Requires_Coupling_Analysis     => False,
               Requires_Lifecycle_Analysis    => False
            );
      end case;
   end Get_DAL_Requirements;

end OOP_Types;
