-------------------------------------------------------------------------------
--  STUNIR JSON - Ada SPARK Implementation
--  PRIMARY IMPLEMENTATION (Ada SPARK is the default language for STUNIR tools)
--
--  Lightweight JSON file IO utilities for SPARK tools.
--
--  Copyright (c) 2026 STUNIR Project
--  License: MIT
-------------------------------------------------------------------------------

pragma SPARK_Mode (On);

with Ada.Streams;
with Ada.Streams.Stream_IO;
with Ada.Directories;
with GNAT.SHA256;

package body STUNIR_JSON is

   use Ada.Streams;

   procedure Read_JSON_File
     (Path    : String;
      Content : out JSON_String;
      Success : out Boolean)
   is
      File    : Ada.Streams.Stream_IO.File_Type;
      Buffer  : Ada.Streams.Stream_Element_Array (1 .. 8192);
      Last    : Ada.Streams.Stream_Element_Offset;
      Current : String := "";
   begin
      Content := JSON_Buffers.Null_Bounded_String;
      Success := False;

      if not Ada.Directories.Exists (Path) then
         return;
      end if;

      Ada.Streams.Stream_IO.Open (File, Ada.Streams.Stream_IO.In_File, Path);

      while not Ada.Streams.Stream_IO.End_Of_File (File) loop
         Ada.Streams.Stream_IO.Read (File, Buffer, Last);
         if Last >= Buffer'First then
            declare
               Chunk : String (1 .. Integer (Last));
            begin
               for I in Buffer'First .. Last loop
                  Chunk (Integer (I)) := Character'Val (Buffer (I));
               end loop;
               if Current'Length + Chunk'Length <= Max_JSON_Size then
                  Current := Current & Chunk;
               end if;
            end;
         end if;
      end loop;

      Ada.Streams.Stream_IO.Close (File);

      if Current'Length > 0 then
         Content := JSON_Buffers.To_Bounded_String (Current);
         Success := True;
      end if;

   exception
      when others =>
         if Ada.Streams.Stream_IO.Is_Open (File) then
            Ada.Streams.Stream_IO.Close (File);
         end if;
         Content := JSON_Buffers.Null_Bounded_String;
         Success := False;
   end Read_JSON_File;

   procedure Write_JSON_File
     (Path    : String;
      Content : String;
      Success : out Boolean)
   is
      File : Ada.Streams.Stream_IO.File_Type;
   begin
      Success := False;

      if Content'Length = 0 or Content'Length > Max_JSON_Size then
         return;
      end if;

      Ada.Streams.Stream_IO.Create (File, Ada.Streams.Stream_IO.Out_File, Path);
      declare
         Buffer : Ada.Streams.Stream_Element_Array (1 .. Content'Length);
      begin
         for I in Content'Range loop
            Buffer (Ada.Streams.Stream_Element_Offset (I)) := Ada.Streams.Stream_Element (Character'Pos (Content (I)));
         end loop;
         Ada.Streams.Stream_IO.Write (File, Buffer);
      end;
      Ada.Streams.Stream_IO.Close (File);
      Success := True;

   exception
      when others =>
         if Ada.Streams.Stream_IO.Is_Open (File) then
            Ada.Streams.Stream_IO.Close (File);
         end if;
         Success := False;
   end Write_JSON_File;

   procedure Compute_File_Hash
     (Path    : String;
      Hash    : out Hash_String;
      Success : out Boolean)
   is
      File    : Ada.Streams.Stream_IO.File_Type;
      Context : GNAT.SHA256.Context := GNAT.SHA256.Initial_Context;
      Buffer  : Ada.Streams.Stream_Element_Array (1 .. 8192);
      Last    : Ada.Streams.Stream_Element_Offset;
   begin
      Hash := Hash_Strings.Null_Bounded_String;
      Success := False;

      if not Ada.Directories.Exists (Path) then
         return;
      end if;

      Ada.Streams.Stream_IO.Open (File, Ada.Streams.Stream_IO.In_File, Path);
      while not Ada.Streams.Stream_IO.End_Of_File (File) loop
         Ada.Streams.Stream_IO.Read (File, Buffer, Last);
         if Last >= Buffer'First then
            GNAT.SHA256.Update (Context, Buffer (Buffer'First .. Last));
         end if;
      end loop;
      Ada.Streams.Stream_IO.Close (File);

      declare
         Digest : constant String := GNAT.SHA256.Digest (Context);
      begin
         Hash := Hash_Strings.To_Bounded_String (Digest);
         Success := True;
      end;

   exception
      when others =>
         if Ada.Streams.Stream_IO.Is_Open (File) then
            Ada.Streams.Stream_IO.Close (File);
         end if;
         Hash := Hash_Strings.Null_Bounded_String;
         Success := False;
   end Compute_File_Hash;

end STUNIR_JSON;
