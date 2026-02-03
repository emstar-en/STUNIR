--  STUNIR GPU Emitter - Ada SPARK Specification
--  Emit CUDA/OpenCL code for GPU targets
--  DO-178C Level A compliant

pragma SPARK_Mode (On);

with Emitter_Types; use Emitter_Types;

package GPU_Emitter is

   type GPU_Backend is (CUDA, OpenCL, Metal, Vulkan_Compute);

   type GPU_Config is record
      Backend       : GPU_Backend;
      Block_Size_X  : Positive;
      Block_Size_Y  : Positive;
      Block_Size_Z  : Positive;
      Shared_Memory : Natural;
   end record;

   Default_Config : constant GPU_Config := (
      Backend       => CUDA,
      Block_Size_X  => 256,
      Block_Size_Y  => 1,
      Block_Size_Z  => 1,
      Shared_Memory => 0
   );

   procedure Emit_Kernel (
      Name      : in Identifier_String;
      Content   : out Content_String;
      Config    : in GPU_Config;
      Status    : out Emitter_Status)
      with Pre => Identifier_Strings.Length (Name) > 0;

   procedure Emit_Host_Code (
      Name      : in Identifier_String;
      Content   : out Content_String;
      Config    : in GPU_Config;
      Status    : out Emitter_Status);

end GPU_Emitter;
