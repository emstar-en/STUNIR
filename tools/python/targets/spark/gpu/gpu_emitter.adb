--  STUNIR GPU Emitter - Ada SPARK Implementation

pragma SPARK_Mode (On);

package body GPU_Emitter is

   New_Line : constant Character := ASCII.LF;

   procedure Append_Safe (
      Content : in out Content_String;
      Text    : in String;
      Status  : out Emitter_Status)
   is
   begin
      if Content_Strings.Length (Content) + Text'Length > Max_Content_Length then
         Status := Error_Buffer_Overflow;
      else
         Content_Strings.Append (Content, Text);
         Status := Success;
      end if;
   end Append_Safe;

   procedure Emit_Kernel (
      Name      : in Identifier_String;
      Content   : out Content_String;
      Config    : in GPU_Config;
      Status    : out Emitter_Status)
   is
      Kernel_Name : constant String := Identifier_Strings.To_String (Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Backend is
         when CUDA =>
            Append_Safe (Content,
               "/* STUNIR Generated CUDA Kernel */" & New_Line &
               "__global__ void " & Kernel_Name & "(" &
               "float* input, float* output, int n) {" & New_Line &
               "    int idx = blockIdx.x * blockDim.x + threadIdx.x;" & New_Line &
               "    if (idx < n) {" & New_Line &
               "        output[idx] = input[idx];" & New_Line &
               "    }" & New_Line &
               "}" & New_Line, Status);

         when OpenCL =>
            Append_Safe (Content,
               "/* STUNIR Generated OpenCL Kernel */" & New_Line &
               "__kernel void " & Kernel_Name & "(" &
               "__global float* input, __global float* output, int n) {" & New_Line &
               "    int idx = get_global_id(0);" & New_Line &
               "    if (idx < n) {" & New_Line &
               "        output[idx] = input[idx];" & New_Line &
               "    }" & New_Line &
               "}" & New_Line, Status);

         when Metal =>
            Append_Safe (Content,
               "/* STUNIR Generated Metal Kernel */" & New_Line &
               "kernel void " & Kernel_Name & "(" &
               "device float* input [[buffer(0)]]," & New_Line &
               "device float* output [[buffer(1)]]," & New_Line &
               "uint idx [[thread_position_in_grid]]) {" & New_Line &
               "    output[idx] = input[idx];" & New_Line &
               "}" & New_Line, Status);

         when Vulkan_Compute =>
            Append_Safe (Content,
               "/* STUNIR Generated Vulkan Compute Shader */" & New_Line &
               "#version 450" & New_Line &
               "layout(local_size_x = 256) in;" & New_Line &
               "layout(binding = 0) buffer Input { float data[]; } input_buf;" & New_Line &
               "layout(binding = 1) buffer Output { float data[]; } output_buf;" & New_Line &
               "void main() {" & New_Line &
               "    uint idx = gl_GlobalInvocationID.x;" & New_Line &
               "    output_buf.data[idx] = input_buf.data[idx];" & New_Line &
               "}" & New_Line, Status);
      end case;
   end Emit_Kernel;

   procedure Emit_Host_Code (
      Name      : in Identifier_String;
      Content   : out Content_String;
      Config    : in GPU_Config;
      Status    : out Emitter_Status)
   is
      Kernel_Name : constant String := Identifier_Strings.To_String (Name);
   begin
      Content := Content_Strings.Null_Bounded_String;
      Status := Success;

      case Config.Backend is
         when CUDA =>
            Append_Safe (Content,
               "/* STUNIR Generated CUDA Host Code */" & New_Line &
               "#include <cuda_runtime.h>" & New_Line &
               "extern __global__ void " & Kernel_Name & "(float*, float*, int);" & New_Line &
               "void launch_" & Kernel_Name & "(float* d_in, float* d_out, int n) {" & New_Line &
               "    dim3 block(256);" & New_Line &
               "    dim3 grid((n + 255) / 256);" & New_Line &
               "    " & Kernel_Name & "<<<grid, block>>>(d_in, d_out, n);" & New_Line &
               "}" & New_Line, Status);
         when others =>
            Append_Safe (Content,
               "/* Host code for " & GPU_Backend'Image (Config.Backend) & " */" & New_Line, Status);
      end case;
   end Emit_Host_Code;

end GPU_Emitter;
