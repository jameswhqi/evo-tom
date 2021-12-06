{-# LANGUAGE TupleSections #-}

module Main where

-- base
import Data.Foldable (traverse_)
-- proto-lens
import Data.ProtoLens (decodeMessageOrDie)
-- microlens
import Lens.Micro ((^..), traversed, (&))
-- bytestring
import qualified Data.ByteString as B (readFile)
-- optparse-applicative
import Options.Applicative ((<**>), Parser, argument, str, metavar, helper, briefDesc, info, execParser)
import Data.Matrix (fromLists, Matrix, getElem, nrows, ncols, splitBlocks, getRow, rowVector, (<->), colVector, getCol, (<|>))
import Formatting (formatToString, fixed)

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
                  "writePlots" -> traverse_ writePlot files
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

writePlot :: String -> IO ()
writePlot name = do
  msg <- loadMsg ("output/" ++ name ++ ".bin")
  let mat1 = getMeanMatrix msg
        & fromLists
        & mapApply (iterateN 3 . dupRowCol) [1,5]
        & mapApply dupRowCol [10,14,18,22]
      string1 = unlines [unwords [show $ fi j - (0.5::Double), show $ fi i - (0.5::Double), formatToString (fixed 4) (getElem i j mat1)] | i <- [1..nrows mat1], j <- [1..ncols mat1]]
  writeFile ("output/fullplot-" ++ name ++ ".txt") $ "x y z\n" ++ string1
  let mat2 = getMeanMatrix msg
        & fromLists
        & xformMeanMatrix
      string2 = unlines [unwords [show $ fi j - (0.5::Double), show $ fi i - (0.5::Double), formatToString (fixed 4) (getElem i j mat2)] | i <- [1..nrows mat2], j <- [1..ncols mat2]]
  writeFile ("output/sumplot-" ++ name ++ ".txt") $ "x y z\n" ++ string2

mapApply :: (a -> b -> b) -> [a] -> b -> b
mapApply f l x = foldl (&) x $ map f l

iterateN :: Int -> (a -> a) -> a -> a
iterateN n f x = iterate f x !! n

dupRow :: Int -> Matrix a -> Matrix a
dupRow i m = let (tl, _, bl, _) = splitBlocks i (ncols m) m
                 row = rowVector $ getRow i tl
             in tl <-> row <-> bl
  
dupCol :: Int -> Matrix a -> Matrix a
dupCol i m = let (tl, tr, _, _) = splitBlocks (nrows m) i m
                 col = colVector $ getCol i tl
             in tl <|> col <|> tr

dupRowCol :: Int -> Matrix a -> Matrix a
dupRowCol i = dupRow i . dupCol i
