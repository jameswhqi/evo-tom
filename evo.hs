{-# LANGUAGE TupleSections #-}

module Main where

-- base
import Data.List (transpose, foldl')
-- vector
import Data.Vector (Vector, fromList, toList)
import qualified Data.Vector as V (map, zipWith, sum)
-- matrix
import Data.Matrix (Matrix, fromLists, toLists, multStd2, colVector, scaleRow, combineRows, minorMatrix)
import qualified Data.Matrix as M (transpose)
-- microlens
import Lens.Micro ((^..), traversed, (&))
-- bytestring
import qualified Data.ByteString as B (readFile)
-- proto-lens
import Data.ProtoLens (decodeMessageOrDie)
-- formatting
import Formatting (formatToString, fixed)
-- optparse-applicative
import Options.Applicative ((<**>), Parser, argument, str, metavar, helper, briefDesc, info, execParser)

import qualified Proto.Results as P
import qualified Proto.Results_Fields as P
import Lib

type Result = [(Double, Vector Double)]

interval :: Int
interval = 2
timestep :: Double
timestep = 0.5

args :: Parser String
args = argument str (metavar "ENV")

main :: IO ()
main = do
  env <- execParser $ info (args <**> helper) briefDesc
  msg <- loadMsg $ "output/"++env++".bin"
  let m = xformMeanMatrix . getMeanMatrix $ msg
      result = evolve m . normalizeDist $ fromList [1,1,1,1,1,1]
  plotResult env interval result
  writeResult env interval result

loadMsg :: String -> IO P.AllResult
loadMsg path = do
  bs <- B.readFile path
  return $ decodeMessageOrDie bs

getMeanMatrix :: P.AllResult -> Matrix Double
getMeanMatrix r = fromLists $ r ^.. P.meanMatrix . traversed . P.row

xformMeanMatrix :: Matrix Double -> Matrix Double
xformMeanMatrix m = foldl' (&) m
  $ [f 0.5 i | f <- [scaleRow, scaleCol], i <- [4,7,10,13]]
  ++ [f i 0.25 (i+j) | f <- [combineRows, combineCols], i <- [4,7,10,13], j <- [-1,1]]
  ++ [minorMatrix i i | i <- [3,4,4,5,5,6,6,7]]

evolve :: Matrix Double -> Vector Double -> Result
evolve m initial = takeWhile continue . zip [0,timestep..] $ iterate (step m) initial
  where continue (i, _) = i < 250

step :: Matrix Double -> Vector Double -> Vector Double
step m v = let p = map sum . toLists $ m `multStd2` colVector v
               pMean = sum . zipWith (*) p $ toList v
               delta = zipWith (*) (toList v) $ map (subtract pMean) p
           in normalizeDist . V.zipWith (+) v . fromList $ map (*timestep) delta

normalizeDist :: Vector Double -> Vector Double
normalizeDist v = V.map (/V.sum v) v

plotResult :: String -> Int -> Result -> IO ()
plotResult env intrvl = plotLinesWithX ("output/evo-"++env++".pdf") ["random", "tft", "num", "fbm", "rl", "tom"] . toLines . every intrvl

toLines :: Result -> [[(Double, Double)]]
toLines = transpose . map f
  where f (i, v) = map (i,) $ toList v

writeResult :: String -> Int -> Result -> IO ()
writeResult env intrvl = writeFile ("output/evo-"++env++".txt") . ("i r tft num fbm rl tom\n" ++) . unlines . map f . every intrvl
  where f (i, v) = unwords . ([formatToString (fixed 1) i] ++) . map (formatToString (fixed 4)) $ toList v

-- matrix operations
scaleCol :: Num a => a -> Int -> Matrix a -> Matrix a
scaleCol a i = M.transpose . scaleRow a i . M.transpose

combineCols :: Num a => Int -> a -> Int -> Matrix a -> Matrix a
combineCols i a j = M.transpose . combineRows i a j . M.transpose
