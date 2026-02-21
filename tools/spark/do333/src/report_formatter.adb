--  STUNIR DO-333 Report Formatter
--  Implementation
--  Copyright (C) 2026 STUNIR Project
--  SPDX-License-Identifier: Apache-2.0

pragma SPARK_Mode (On);

package body Report_Formatter is

   --  ============================================================
   --  Helper: Append
   --  ============================================================

   procedure Append
     (Content : in Out String;
      Length  : in Out Natural;
      Text    : String)
   with
      Pre => Length + Text'Length <= Content'Length
   is
   begin
      for I in Text'Range loop
         Length := Length + 1;
         Content (Content'First + Length - 1) := Text (I);
      end loop;
   end Append;

   procedure Append_Line
     (Content : in Out String;
      Length  : in Out Natural)
   with
      Pre => Length + 1 <= Content'Length
   is
   begin
      Length := Length + 1;
      Content (Content'First + Length - 1) := ASCII.LF;
   end Append_Line;

   --  ============================================================
   --  Generate Text Header
   --  ============================================================

   procedure Generate_Text_Header
     (Header  : Report_Header;
      Content : out String;
      Length  : out Natural)
   is
   begin
      Content := (others => ' ');
      Length := 0;

      --  Banner
      Append (Content, Length, "========================================");
      Append_Line (Content, Length);

      --  Title
      if Header.Title_Len > 0 then
         Append (Content, Length, Header.Title (1 .. Header.Title_Len));
         Append_Line (Content, Length);
      end if;

      --  Separator
      Append (Content, Length, "========================================");
      Append_Line (Content, Length);

      --  Timestamp
      Append (Content, Length, "Generated: ");
      Append (Content, Length, Header.Timestamp);
      Append_Line (Content, Length);

      --  Tool version
      Append (Content, Length, "Tool: ");
      Append (Content, Length, Header.Tool_Ver (1 .. Header.Ver_Len));
      Append_Line (Content, Length);

      Append_Line (Content, Length);
   end Generate_Text_Header;

   --  ============================================================
   --  Generate HTML Header
   --  ============================================================

   procedure Generate_HTML_Header
     (Header  : Report_Header;
      Content : out String;
      Length  : out Natural)
   is
   begin
      Content := (others => ' ');
      Length := 0;

      Append (Content, Length, "<!DOCTYPE html>");
      Append_Line (Content, Length);
      Append (Content, Length, "<html lang=""en"">");
      Append_Line (Content, Length);
      Append (Content, Length, "<head>");
      Append_Line (Content, Length);
      Append (Content, Length, "<meta charset=""UTF-8"">");
      Append_Line (Content, Length);
      Append (Content, Length, "<title>");

      if Header.Title_Len > 0 then
         Append (Content, Length, Header.Title (1 .. Header.Title_Len));
      else
         Append (Content, Length, "DO-333 Report");
      end if;

      Append (Content, Length, "</title>");
      Append_Line (Content, Length);

      --  Basic CSS
      Append (Content, Length, "<style>");
      Append_Line (Content, Length);
      Append (Content, Length, "body { font-family: Arial, sans-serif; margin: 20px; }");
      Append_Line (Content, Length);
      Append (Content, Length, "h1 { color: #333; }");
      Append_Line (Content, Length);
      Append (Content, Length, "table { border-collapse: collapse; width: 100%; }");
      Append_Line (Content, Length);
      Append (Content, Length, "th, td { border: 1px solid #ddd; padding: 8px; }");
      Append_Line (Content, Length);
      Append (Content, Length, ".pass { color: green; }");
      Append_Line (Content, Length);
      Append (Content, Length, ".fail { color: red; }");
      Append_Line (Content, Length);
      Append (Content, Length, "</style>");
      Append_Line (Content, Length);

      Append (Content, Length, "</head>");
      Append_Line (Content, Length);
      Append (Content, Length, "<body>");
      Append_Line (Content, Length);

      --  Title
      Append (Content, Length, "<h1>");
      if Header.Title_Len > 0 then
         Append (Content, Length, Header.Title (1 .. Header.Title_Len));
      else
         Append (Content, Length, "DO-333 Report");
      end if;
      Append (Content, Length, "</h1>");
      Append_Line (Content, Length);

      --  Meta info
      Append (Content, Length, "<p>Generated: ");
      Append (Content, Length, Header.Timestamp);
      Append (Content, Length, " | Tool: ");
      Append (Content, Length, Header.Tool_Ver (1 .. Header.Ver_Len));
      Append (Content, Length, "</p>");
      Append_Line (Content, Length);
   end Generate_HTML_Header;

   --  ============================================================
   --  Generate Footer
   --  ============================================================

   procedure Generate_Footer
     (Format  : Output_Format;
      Content : out String;
      Length  : out Natural)
   is
   begin
      Content := (others => ' ');
      Length := 0;

      case Format is
         when Format_Text =>
            Append_Line (Content, Length);
            Append (Content, Length, "========================================");
            Append_Line (Content, Length);
            Append (Content, Length, "End of Report");
            Append_Line (Content, Length);

         when Format_HTML =>
            Append (Content, Length, "<hr>");
            Append_Line (Content, Length);
            Append (Content, Length, "<footer>");
            Append_Line (Content, Length);
            Append (Content, Length, "<p><em>Generated by STUNIR DO-333 Tools</em></p>");
            Append_Line (Content, Length);
            Append (Content, Length, "</footer>");
            Append_Line (Content, Length);
            Append (Content, Length, "</body>");
            Append_Line (Content, Length);
            Append (Content, Length, "</html>");
            Append_Line (Content, Length);

         when Format_JSON =>
            --  JSON doesn't need a footer
            null;

         when Format_XML =>
            Append (Content, Length, "<!-- End of DO-333 Report -->");
            Append_Line (Content, Length);

         when Format_CSV =>
            --  CSV doesn't need a footer
            null;
      end case;
   end Generate_Footer;

   --  ============================================================
   --  Create Header
   --  ============================================================

   procedure Create_Header
     (Title  : String;
      Header : out Report_Header)
   is
      Title_Buf : Title_String := (others => ' ');
   begin
      for I in Title'Range loop
         Title_Buf (I - Title'First + 1) := Title (I);
      end loop;

      Header := Default_Header;
      Header.Title := Title_Buf;
      Header.Title_Len := Title'Length;
   end Create_Header;

end Report_Formatter;
