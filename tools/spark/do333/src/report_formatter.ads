--  STUNIR DO-333 Report Formatter
--  Report formatting utilities
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0
--
--  This package provides report formatting utilities:
--  - File output
--  - Format conversion
--  - Header/footer generation
--
--  DO-333 Objective: FM.6 (Certification Evidence)

pragma SPARK_Mode (On);

with Evidence_Generator; use Evidence_Generator;

package Report_Formatter is

   --  ============================================================
   --  Constants
   --  ============================================================

   Max_Header_Size  : constant := 4096;
   Max_Footer_Size  : constant := 1024;
   Max_Title_Length : constant := 256;

   --  ============================================================
   --  Report Header
   --  ============================================================

   subtype Title_String is String (1 .. Max_Title_Length);

   type Report_Header is record
      Title       : Title_String;
      Title_Len   : Natural;
      Timestamp   : String (1 .. 19);  --  YYYY-MM-DD HH:MM:SS
      Tool_Ver    : String (1 .. 16);
      Ver_Len     : Natural;
   end record;

   Default_Header : constant Report_Header := (
      Title     => (others => ' '),
      Title_Len => 0,
      Timestamp => "2026-01-29 00:00:00",
      Tool_Ver  => "DO333 v1.0      ",
      Ver_Len   => 10
   );

   --  ============================================================
   --  Write Result
   --  ============================================================

   type Write_Result is (
      Write_Success,
      Write_Error_IO,
      Write_Error_Path,
      Write_Error_Permission
   );

   --  ============================================================
   --  Operations
   --  ============================================================

   --  Generate text header
   procedure Generate_Text_Header
     (Header  : Report_Header;
      Content : out String;
      Length  : out Natural)
   with
      Pre => Content'Length >= Max_Header_Size;

   --  Generate HTML header
   procedure Generate_HTML_Header
     (Header  : Report_Header;
      Content : out String;
      Length  : out Natural)
   with
      Pre => Content'Length >= Max_Header_Size;

   --  Generate footer
   procedure Generate_Footer
     (Format  : Output_Format;
      Content : out String;
      Length  : out Natural)
   with
      Pre => Content'Length >= Max_Footer_Size;

   --  Create header with title
   procedure Create_Header
     (Title  : String;
      Header : out Report_Header)
   with
      Pre => Title'Length <= Max_Title_Length;

end Report_Formatter;
