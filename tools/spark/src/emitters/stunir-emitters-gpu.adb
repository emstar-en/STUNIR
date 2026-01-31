-- STUNIR GPU Emitter (Body)
-- DO-178C Level A
-- Phase 3a: Core Category Emitters

package body STUNIR.Emitters.GPU is
   pragma SPARK_Mode (On);

   function Get_Platform_Name (Platform : GPU_Platform) return String is
   begin
      case Platform is
         when Platform_CUDA   => return "CUDA";
         when Platform_OpenCL => return "OpenCL";
         when Platform_Metal  => return "Metal";
         when Platform_ROCm   => return "ROCm";
         when Platform_Vulkan => return "Vulkan";
      end case;
   end Get_Platform_Name;

   function Get_Kernel_Prefix (Platform : GPU_Platform) return String is
   begin
      case Platform is
         when Platform_CUDA   => return "__global__";
         when Platform_OpenCL => return "__kernel";
         when Platform_Metal  => return "kernel";
         when Platform_ROCm   => return "__global__";
         when Platform_Vulkan => return "";
      end case;
   end Get_Kernel_Prefix;

   overriding procedure Emit_Module
     (Self    : in out GPU_Emitter;
      Module  : in     IR_Module;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
   begin
      Success := False;
      Initialize (Gen);

      Append_Line (Gen, "/* STUNIR Generated GPU Code */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "/* Platform: " & Get_Platform_Name (Self.Config.Platform) & " */", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Platform-specific includes
      case Self.Config.Platform is
         when Platform_CUDA =>
            Append_Line (Gen, "#include <cuda_runtime.h>", Line_Success);
            if not Line_Success then return; end if;
         when Platform_OpenCL =>
            Append_Line (Gen, "#include <CL/cl.h>", Line_Success);
            if not Line_Success then return; end if;
         when Platform_Metal =>
            Append_Line (Gen, "#include <metal_stdlib>", Line_Success);
            if not Line_Success then return; end if;
         when Platform_ROCm =>
            Append_Line (Gen, "#include <hip/hip_runtime.h>", Line_Success);
            if not Line_Success then return; end if;
         when Platform_Vulkan =>
            Append_Line (Gen, "#version 450", Line_Success);
            if not Line_Success then return; end if;
      end case;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      -- Emit types
      for I in 1 .. Module.Type_Cnt loop
         pragma Loop_Invariant (I <= Module.Type_Cnt);

         declare
            Type_Output : IR_Code_Buffer;
            Type_Success : Boolean;
         begin
            Emit_Type (Self, Module.Types (I), Type_Output, Type_Success);
            if Type_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Type_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      -- Emit functions as kernels
      for I in 1 .. Module.Func_Cnt loop
         pragma Loop_Invariant (I <= Module.Func_Cnt);

         declare
            Kernel_Output : IR_Code_Buffer;
            Kernel_Success : Boolean;
         begin
            Emit_Kernel (Self, Module.Functions (I), Kernel_Output, Kernel_Success);
            if Kernel_Success then
               Append_Raw (Gen, Code_Buffers.To_String (Kernel_Output), Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Module;

   overriding procedure Emit_Type
     (Self    : in out GPU_Emitter;
      T       : in     IR_Type_Def;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Type_Name : constant String := Get_Type_Name (T);
   begin
      Success := False;
      Initialize (Gen);

      Append_Line (Gen, "typedef struct {", Line_Success);
      if not Line_Success then return; end if;

      Increase_Indent (Gen);

      for I in 1 .. T.Field_Cnt loop
         pragma Loop_Invariant (I <= T.Field_Cnt);

         declare
            Field_Name : constant String := Name_Strings.To_String (T.Fields (I).Name);
            Field_Type : constant String := Type_Strings.To_String (T.Fields (I).Type_Ref);
         begin
            Append_Line (Gen, Field_Type & " " & Field_Name & ";", Line_Success);
            if not Line_Success then return; end if;
         end;
      end loop;

      Decrease_Indent (Gen);

      Append_Line (Gen, "} " & Type_Name & ";", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Type;

   overriding procedure Emit_Function
     (Self    : in out GPU_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
   begin
      Emit_Kernel (Self, Func, Output, Success);
   end Emit_Function;

   procedure Emit_Kernel
     (Self    : in out GPU_Emitter;
      Func    : in     IR_Function;
      Output  :    out IR_Code_Buffer;
      Success :    out Boolean)
   is
      Gen : Code_Generator;
      Line_Success : Boolean;
      Func_Name : constant String := Get_Function_Name (Func);
      Return_Type : constant String := Type_Strings.To_String (Func.Return_Type);
      Kernel_Prefix : constant String := Get_Kernel_Prefix (Self.Config.Platform);
   begin
      Success := False;
      Initialize (Gen);

      -- Kernel signature
      declare
         Signature : String := Kernel_Prefix & " " & Return_Type & " " & Func_Name & "(";
      begin
         for I in 1 .. Func.Arg_Cnt loop
            pragma Loop_Invariant (I <= Func.Arg_Cnt);

            declare
               Arg_Type : constant String := Type_Strings.To_String (Func.Args (I).Type_Ref);
               Arg_Name : constant String := Name_Strings.To_String (Func.Args (I).Name);
            begin
               if I > 1 then
                  Signature := Signature & ", ";
               end if;
               Signature := Signature & Arg_Type & " " & Arg_Name;
            end;
         end loop;

         Signature := Signature & ") {";
         Append_Line (Gen, Signature, Line_Success);
         if not Line_Success then return; end if;
      end;

      Increase_Indent (Gen);

      -- Thread indexing
      case Self.Config.Platform is
         when Platform_CUDA | Platform_ROCm =>
            Append_Line (Gen, "int idx = blockIdx.x * blockDim.x + threadIdx.x;", Line_Success);
            if not Line_Success then return; end if;
         when Platform_OpenCL =>
            Append_Line (Gen, "int idx = get_global_id(0);", Line_Success);
            if not Line_Success then return; end if;
         when Platform_Metal =>
            Append_Line (Gen, "uint idx = thread_position_in_grid;", Line_Success);
            if not Line_Success then return; end if;
         when Platform_Vulkan =>
            Append_Line (Gen, "uint idx = gl_GlobalInvocationID.x;", Line_Success);
            if not Line_Success then return; end if;
      end case;

      -- Kernel body
      for I in 1 .. Func.Stmt_Cnt loop
         pragma Loop_Invariant (I <= Func.Stmt_Cnt);

         declare
            Stmt_Data : constant String := Code_Buffers.To_String (Func.Statements (I).Data);
         begin
            if Stmt_Data'Length > 0 then
               Append_Line (Gen, Stmt_Data, Line_Success);
               if not Line_Success then return; end if;
            end if;
         end;
      end loop;

      Decrease_Indent (Gen);
      Append_Line (Gen, "}", Line_Success);
      if not Line_Success then return; end if;

      Append_Line (Gen, "", Line_Success);
      if not Line_Success then return; end if;

      Get_Output (Gen, Output);
      Success := True;
   end Emit_Kernel;

   procedure Emit_Memory_Transfer
     (Self    : in out GPU_Emitter;
      Gen     : in out Code_Generator;
      Success :    out Boolean)
   is
      Line_Success : Boolean;
   begin
      Success := False;

      Append_Line (Gen, "/* Memory Transfer */", Line_Success);
      if not Line_Success then return; end if;

      case Self.Config.Platform is
         when Platform_CUDA =>
            Append_Line (Gen, "cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);", Line_Success);
            if not Line_Success then return; end if;
         when Platform_OpenCL =>
            Append_Line (Gen, "clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);", Line_Success);
            if not Line_Success then return; end if;
         when others =>
            Append_Line (Gen, "/* Platform-specific memory transfer */", Line_Success);
            if not Line_Success then return; end if;
      end case;

      Success := True;
   end Emit_Memory_Transfer;

end STUNIR.Emitters.GPU;
