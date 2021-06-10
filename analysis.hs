{-# LANGUAGE TupleSections #-}

module Main where

-- base
import Data.Foldable (traverse_)
-- proto-lens
import Data.ProtoLens (decodeMessageOrDie)
-- microlens
import Lens.Micro ((^..), traversed)
-- bytestring
import qualified Data.ByteString as B (readFile)
-- optparse-applicative
import Options.Applicative ((<**>), Parser, argument, str, metavar, helper, briefDesc, info, execParser)

import qualified Proto.Results as P
import qualified Proto.Results_Fields as P 
import Lib

type Payoffs = (Double, Double)
type Index = (Int, Int)

args :: Parser String
args = argument str (metavar "COMMAND")

main :: IO ()
main = do
  command <- execParser $ info (args <**> helper) briefDesc
  case command of "writeMeans" -> traverse_ writeMean files
                  "writeSems" -> traverse_ writeSem files
                  _ -> error $ "Invalid command: " ++ command

loadMsg :: String -> IO P.AllResult
loadMsg path = do
  bs <- B.readFile path
  return $ decodeMessageOrDie bs

getMeanMatrix :: P.AllResult -> [[Double]]
getMeanMatrix r = r ^.. P.meanMatrix . traversed . P.row

getSemMatrix :: P.AllResult -> [[Double]]
getSemMatrix r = r ^.. P.semMatrix . traversed . P.row

files :: [String]
files = ["os-pf", "os-ps", "os-pv", "ov-pf", "ov-ps", "ov-pv"]

writeMean :: String -> IO ()
writeMean name = do
  msg <- loadMsg ("output/" ++ name ++ ".bin")
  writeFile ("output/mean-" ++ name ++ ".txt") . matrixToStr 4 $ getMeanMatrix msg

writeSem :: String -> IO ()
writeSem name = do
  msg <- loadMsg ("output/" ++ name ++ ".bin")
  writeFile ("output/sem-" ++ name ++ ".txt") . matrixToStr 4 $ getSemMatrix msg
