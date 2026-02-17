pragma Warnings (Off);
pragma Ada_95;
pragma Source_File_Name (ada_main, Spec_File_Name => "b__stunir_ir_to_code_main.ads");
pragma Source_File_Name (ada_main, Body_File_Name => "b__stunir_ir_to_code_main.adb");
pragma Suppress (Overflow_Check);
with Ada.Exceptions;

package body ada_main is

   E070 : Short_Integer; pragma Import (Ada, E070, "system__os_lib_E");
   E016 : Short_Integer; pragma Import (Ada, E016, "ada__exceptions_E");
   E012 : Short_Integer; pragma Import (Ada, E012, "system__soft_links_E");
   E010 : Short_Integer; pragma Import (Ada, E010, "system__exception_table_E");
   E035 : Short_Integer; pragma Import (Ada, E035, "ada__containers_E");
   E066 : Short_Integer; pragma Import (Ada, E066, "ada__io_exceptions_E");
   E025 : Short_Integer; pragma Import (Ada, E025, "ada__numerics_E");
   E007 : Short_Integer; pragma Import (Ada, E007, "ada__strings_E");
   E053 : Short_Integer; pragma Import (Ada, E053, "ada__strings__maps_E");
   E056 : Short_Integer; pragma Import (Ada, E056, "ada__strings__maps__constants_E");
   E040 : Short_Integer; pragma Import (Ada, E040, "interfaces__c_E");
   E019 : Short_Integer; pragma Import (Ada, E019, "system__exceptions_E");
   E081 : Short_Integer; pragma Import (Ada, E081, "system__object_reader_E");
   E047 : Short_Integer; pragma Import (Ada, E047, "system__dwarf_lines_E");
   E101 : Short_Integer; pragma Import (Ada, E101, "system__soft_links__initialize_E");
   E034 : Short_Integer; pragma Import (Ada, E034, "system__traceback__symbolic_E");
   E105 : Short_Integer; pragma Import (Ada, E105, "ada__strings__utf_encoding_E");
   E113 : Short_Integer; pragma Import (Ada, E113, "ada__tags_E");
   E005 : Short_Integer; pragma Import (Ada, E005, "ada__strings__text_buffers_E");
   E189 : Short_Integer; pragma Import (Ada, E189, "gnat_E");
   E144 : Short_Integer; pragma Import (Ada, E144, "ada__streams_E");
   E180 : Short_Integer; pragma Import (Ada, E180, "system__file_control_block_E");
   E150 : Short_Integer; pragma Import (Ada, E150, "system__finalization_root_E");
   E142 : Short_Integer; pragma Import (Ada, E142, "ada__finalization_E");
   E177 : Short_Integer; pragma Import (Ada, E177, "system__file_io_E");
   E184 : Short_Integer; pragma Import (Ada, E184, "system__storage_pools_E");
   E162 : Short_Integer; pragma Import (Ada, E162, "ada__strings__unbounded_E");
   E125 : Short_Integer; pragma Import (Ada, E125, "ada__calendar_E");
   E136 : Short_Integer; pragma Import (Ada, E136, "ada__calendar__time_zones_E");
   E186 : Short_Integer; pragma Import (Ada, E186, "ada__text_io_E");
   E193 : Short_Integer; pragma Import (Ada, E193, "gnat__secure_hashes_E");
   E200 : Short_Integer; pragma Import (Ada, E200, "gnat__secure_hashes__sha2_common_E");
   E195 : Short_Integer; pragma Import (Ada, E195, "gnat__secure_hashes__sha2_32_E");
   E191 : Short_Integer; pragma Import (Ada, E191, "gnat__sha256_E");
   E182 : Short_Integer; pragma Import (Ada, E182, "system__regexp_E");
   E123 : Short_Integer; pragma Import (Ada, E123, "ada__directories_E");
   E221 : Short_Integer; pragma Import (Ada, E221, "stunir__semantic_ir_E");
   E188 : Short_Integer; pragma Import (Ada, E188, "stunir_json_utils_E");
   E223 : Short_Integer; pragma Import (Ada, E223, "stunir_string_builder_E");
   E119 : Short_Integer; pragma Import (Ada, E119, "stunir_ir_to_code_E");

   Sec_Default_Sized_Stacks : array (1 .. 1) of aliased System.Secondary_Stack.SS_Stack (System.Parameters.Runtime_Default_Sec_Stack_Size);

   Local_Priority_Specific_Dispatching : constant String := "";
   Local_Interrupt_States : constant String := "";

   Is_Elaborated : Boolean := False;

   procedure finalize_library is
   begin
      declare
         procedure F1;
         pragma Import (Ada, F1, "ada__directories__finalize_body");
      begin
         E123 := E123 - 1;
         F1;
      end;
      declare
         procedure F2;
         pragma Import (Ada, F2, "ada__directories__finalize_spec");
      begin
         F2;
      end;
      E182 := E182 - 1;
      declare
         procedure F3;
         pragma Import (Ada, F3, "system__regexp__finalize_spec");
      begin
         F3;
      end;
      E191 := E191 - 1;
      declare
         procedure F4;
         pragma Import (Ada, F4, "gnat__sha256__finalize_spec");
      begin
         F4;
      end;
      E186 := E186 - 1;
      declare
         procedure F5;
         pragma Import (Ada, F5, "ada__text_io__finalize_spec");
      begin
         F5;
      end;
      E162 := E162 - 1;
      declare
         procedure F6;
         pragma Import (Ada, F6, "ada__strings__unbounded__finalize_spec");
      begin
         F6;
      end;
      declare
         procedure F7;
         pragma Import (Ada, F7, "system__file_io__finalize_body");
      begin
         E177 := E177 - 1;
         F7;
      end;
      declare
         procedure Reraise_Library_Exception_If_Any;
            pragma Import (Ada, Reraise_Library_Exception_If_Any, "__gnat_reraise_library_exception_if_any");
      begin
         Reraise_Library_Exception_If_Any;
      end;
   end finalize_library;

   procedure adafinal is
      procedure s_stalib_adafinal;
      pragma Import (Ada, s_stalib_adafinal, "system__standard_library__adafinal");

      procedure Runtime_Finalize;
      pragma Import (C, Runtime_Finalize, "__gnat_runtime_finalize");

   begin
      if not Is_Elaborated then
         return;
      end if;
      Is_Elaborated := False;
      Runtime_Finalize;
      s_stalib_adafinal;
   end adafinal;

   type No_Param_Proc is access procedure;
   pragma Favor_Top_Level (No_Param_Proc);

   procedure adainit is
      Main_Priority : Integer;
      pragma Import (C, Main_Priority, "__gl_main_priority");
      Time_Slice_Value : Integer;
      pragma Import (C, Time_Slice_Value, "__gl_time_slice_val");
      WC_Encoding : Character;
      pragma Import (C, WC_Encoding, "__gl_wc_encoding");
      Locking_Policy : Character;
      pragma Import (C, Locking_Policy, "__gl_locking_policy");
      Queuing_Policy : Character;
      pragma Import (C, Queuing_Policy, "__gl_queuing_policy");
      Task_Dispatching_Policy : Character;
      pragma Import (C, Task_Dispatching_Policy, "__gl_task_dispatching_policy");
      Priority_Specific_Dispatching : System.Address;
      pragma Import (C, Priority_Specific_Dispatching, "__gl_priority_specific_dispatching");
      Num_Specific_Dispatching : Integer;
      pragma Import (C, Num_Specific_Dispatching, "__gl_num_specific_dispatching");
      Main_CPU : Integer;
      pragma Import (C, Main_CPU, "__gl_main_cpu");
      Interrupt_States : System.Address;
      pragma Import (C, Interrupt_States, "__gl_interrupt_states");
      Num_Interrupt_States : Integer;
      pragma Import (C, Num_Interrupt_States, "__gl_num_interrupt_states");
      Unreserve_All_Interrupts : Integer;
      pragma Import (C, Unreserve_All_Interrupts, "__gl_unreserve_all_interrupts");
      Exception_Tracebacks : Integer;
      pragma Import (C, Exception_Tracebacks, "__gl_exception_tracebacks");
      Detect_Blocking : Integer;
      pragma Import (C, Detect_Blocking, "__gl_detect_blocking");
      Default_Stack_Size : Integer;
      pragma Import (C, Default_Stack_Size, "__gl_default_stack_size");
      Default_Secondary_Stack_Size : System.Parameters.Size_Type;
      pragma Import (C, Default_Secondary_Stack_Size, "__gnat_default_ss_size");
      Bind_Env_Addr : System.Address;
      pragma Import (C, Bind_Env_Addr, "__gl_bind_env_addr");
      Interrupts_Default_To_System : Integer;
      pragma Import (C, Interrupts_Default_To_System, "__gl_interrupts_default_to_system");

      procedure Runtime_Initialize (Install_Handler : Integer);
      pragma Import (C, Runtime_Initialize, "__gnat_runtime_initialize");

      Finalize_Library_Objects : No_Param_Proc;
      pragma Import (C, Finalize_Library_Objects, "__gnat_finalize_library_objects");
      Binder_Sec_Stacks_Count : Natural;
      pragma Import (Ada, Binder_Sec_Stacks_Count, "__gnat_binder_ss_count");
      Default_Sized_SS_Pool : System.Address;
      pragma Import (Ada, Default_Sized_SS_Pool, "__gnat_default_ss_pool");

   begin
      if Is_Elaborated then
         return;
      end if;
      Is_Elaborated := True;
      Main_Priority := -1;
      Time_Slice_Value := -1;
      WC_Encoding := 'b';
      Locking_Policy := ' ';
      Queuing_Policy := ' ';
      Task_Dispatching_Policy := ' ';
      Priority_Specific_Dispatching :=
        Local_Priority_Specific_Dispatching'Address;
      Num_Specific_Dispatching := 0;
      Main_CPU := -1;
      Interrupt_States := Local_Interrupt_States'Address;
      Num_Interrupt_States := 0;
      Unreserve_All_Interrupts := 0;
      Exception_Tracebacks := 1;
      Detect_Blocking := 0;
      Default_Stack_Size := 33554432;

      ada_main'Elab_Body;
      Default_Secondary_Stack_Size := System.Parameters.Runtime_Default_Sec_Stack_Size;
      Binder_Sec_Stacks_Count := 1;
      Default_Sized_SS_Pool := Sec_Default_Sized_Stacks'Address;

      Runtime_Initialize (1);

      Finalize_Library_Objects := finalize_library'access;

      Ada.Exceptions'Elab_Spec;
      System.Soft_Links'Elab_Spec;
      System.Exception_Table'Elab_Body;
      E010 := E010 + 1;
      Ada.Containers'Elab_Spec;
      E035 := E035 + 1;
      Ada.Io_Exceptions'Elab_Spec;
      E066 := E066 + 1;
      Ada.Numerics'Elab_Spec;
      E025 := E025 + 1;
      Ada.Strings'Elab_Spec;
      E007 := E007 + 1;
      Ada.Strings.Maps'Elab_Spec;
      E053 := E053 + 1;
      Ada.Strings.Maps.Constants'Elab_Spec;
      E056 := E056 + 1;
      Interfaces.C'Elab_Spec;
      E040 := E040 + 1;
      System.Exceptions'Elab_Spec;
      E019 := E019 + 1;
      System.Object_Reader'Elab_Spec;
      E081 := E081 + 1;
      System.Dwarf_Lines'Elab_Spec;
      E047 := E047 + 1;
      System.Os_Lib'Elab_Body;
      E070 := E070 + 1;
      System.Soft_Links.Initialize'Elab_Body;
      E101 := E101 + 1;
      E012 := E012 + 1;
      System.Traceback.Symbolic'Elab_Body;
      E034 := E034 + 1;
      E016 := E016 + 1;
      Ada.Strings.Utf_Encoding'Elab_Spec;
      E105 := E105 + 1;
      Ada.Tags'Elab_Spec;
      Ada.Tags'Elab_Body;
      E113 := E113 + 1;
      Ada.Strings.Text_Buffers'Elab_Spec;
      E005 := E005 + 1;
      Gnat'Elab_Spec;
      E189 := E189 + 1;
      Ada.Streams'Elab_Spec;
      E144 := E144 + 1;
      System.File_Control_Block'Elab_Spec;
      E180 := E180 + 1;
      System.Finalization_Root'Elab_Spec;
      E150 := E150 + 1;
      Ada.Finalization'Elab_Spec;
      E142 := E142 + 1;
      System.File_Io'Elab_Body;
      E177 := E177 + 1;
      System.Storage_Pools'Elab_Spec;
      E184 := E184 + 1;
      Ada.Strings.Unbounded'Elab_Spec;
      E162 := E162 + 1;
      Ada.Calendar'Elab_Spec;
      Ada.Calendar'Elab_Body;
      E125 := E125 + 1;
      Ada.Calendar.Time_Zones'Elab_Spec;
      E136 := E136 + 1;
      Ada.Text_Io'Elab_Spec;
      Ada.Text_Io'Elab_Body;
      E186 := E186 + 1;
      E193 := E193 + 1;
      E200 := E200 + 1;
      E195 := E195 + 1;
      Gnat.Sha256'Elab_Spec;
      E191 := E191 + 1;
      System.Regexp'Elab_Spec;
      E182 := E182 + 1;
      Ada.Directories'Elab_Spec;
      Ada.Directories'Elab_Body;
      E123 := E123 + 1;
      E221 := E221 + 1;
      Stunir_Json_Utils'Elab_Spec;
      E188 := E188 + 1;
      E223 := E223 + 1;
      Stunir_Ir_To_Code'Elab_Spec;
      E119 := E119 + 1;
   end adainit;

   procedure Ada_Main_Program;
   pragma Import (Ada, Ada_Main_Program, "_ada_stunir_ir_to_code_main");

   function main
     (argc : Integer;
      argv : System.Address;
      envp : System.Address)
      return Integer
   is
      procedure Initialize (Addr : System.Address);
      pragma Import (C, Initialize, "__gnat_initialize");

      procedure Finalize;
      pragma Import (C, Finalize, "__gnat_finalize");
      SEH : aliased array (1 .. 2) of Integer;

      Ensure_Reference : aliased System.Address := Ada_Main_Program_Name'Address;
      pragma Volatile (Ensure_Reference);

   begin
      if gnat_argc = 0 then
         gnat_argc := argc;
         gnat_argv := argv;
      end if;
      gnat_envp := envp;

      Initialize (SEH'Address);
      adainit;
      Ada_Main_Program;
      adafinal;
      Finalize;
      return (gnat_exit_status);
   end;

--  BEGIN Object file/option list
   --   C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\stunir.o
   --   C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\stunir-semantic_ir.o
   --   C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\stunir_json_utils.o
   --   C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\stunir_string_builder.o
   --   C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\stunir_ir_to_code.o
   --   C:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\stunir_ir_to_code_main.o
   --   -LC:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\
   --   -LC:\Users\MSTAR\AppData\Roaming\AbacusAI\Agent Workspaces\tools\spark\obj\
   --   -LC:/users/mstar/appdata/local/alire/cache/toolchains/gnat_native_15.2.1_346e2e00/lib/gcc/x86_64-w64-mingw32/15.2.0/adalib/
   --   -static
   --   -lgnat
   --   -Wl,--stack=0x2000000
--  END Object file/option list   

end ada_main;
