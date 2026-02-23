-------------------------------------------------------------------------------
--  STUNIR IR Root Package
--  DO-178C Level A Compliant
--  SPARK 2014 Mode
--
--  This is the root package for the STUNIR Intermediate Representation
--  (IR) hierarchy. It provides version information and serves as the parent
--  for all IR child packages.
--
--  The IR is the core data structure used throughout the STUNIR
--  pipeline for representing typed, verifiable code across multiple target
--  languages.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
--
--  Safety: This package is marked as Pure and operates in SPARK_Mode,
--          ensuring no side effects and formal verification compatibility.
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

package IR with
   Pure,
   SPARK_Mode => On
is
   --  IR version constants
   --  These define the current version of the IR schema

   --  Major version number (breaking changes)
   IR_Version_Major : constant := 1;

   --  Minor version number (new features, backward compatible)
   IR_Version_Minor : constant := 0;

   --  Patch version number (bug fixes)
   IR_Version_Patch : constant := 0;

   --  Full schema version string in semver format
   Schema_Version : constant String := "1.0.0";

end IR;
