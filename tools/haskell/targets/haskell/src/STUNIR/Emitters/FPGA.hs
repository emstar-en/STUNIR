{-# LANGUAGE OverloadedStrings #-}

-- | FPGA HDL emitters
module STUNIR.Emitters.FPGA
  ( emitVerilog
  , emitVHDL
  , HDLLanguage(..)
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import STUNIR.Emitters.Types

-- | HDL language
data HDLLanguage
  = Verilog
  | VHDL
  | SystemVerilog
  deriving (Show, Eq)

-- | Emit Verilog code
emitVerilog :: Text -> EmitterResult Text
emitVerilog moduleName = Right $ T.unlines
  [ "// STUNIR Generated Verilog"
  , "// Module: " <> moduleName
  , "// Generator: Haskell Pipeline"
  , ""
  , "module " <> moduleName <> " ("
  , "    input wire clk,"
  , "    input wire rst,"
  , "    input wire [31:0] data_in,"
  , "    output reg [31:0] data_out"
  , ");"
  , ""
  , "    always @(posedge clk or posedge rst) begin"
  , "        if (rst)"
  , "            data_out <= 32'h0;"
  , "        else"
  , "            data_out <= data_in;"
  , "    end"
  , ""
  , "endmodule"
  ]

-- | Emit VHDL code
emitVHDL :: Text -> EmitterResult Text
emitVHDL moduleName = Right $ T.unlines
  [ "-- STUNIR Generated VHDL"
  , "-- Module: " <> moduleName
  , "-- Generator: Haskell Pipeline"
  , ""
  , "library IEEE;"
  , "use IEEE.STD_LOGIC_1164.ALL;"
  , "use IEEE.NUMERIC_STD.ALL;"
  , ""
  , "entity " <> moduleName <> " is"
  , "    Port ("
  , "        clk : in STD_LOGIC;"
  , "        rst : in STD_LOGIC;"
  , "        data_in : in STD_LOGIC_VECTOR(31 downto 0);"
  , "        data_out : out STD_LOGIC_VECTOR(31 downto 0)"
  , "    );"
  , "end " <> moduleName <> ";"
  , ""
  , "architecture Behavioral of " <> moduleName <> " is"
  , "begin"
  , "    process(clk, rst)"
  , "    begin"
  , "        if rst = '1' then"
  , "            data_out <= (others => '0');"
  , "        elsif rising_edge(clk) then"
  , "            data_out <= data_in;"
  , "        end if;"
  , "    end process;"
  , "end Behavioral;"
  ]
