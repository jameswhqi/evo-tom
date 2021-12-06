{-# LANGUAGE TupleSections #-}

module Main where

-- base
import Data.List (transpose)
-- vector
-- vector
import Data.Vector (Vector, fromList, toList, singleton)
import qualified Data.Vector as V (map, zipWith, sum)
-- matrix
import Data.Matrix (Matrix, fromLists, toLists, multStd2, colVector)
-- microlens
import Lens.Micro ((^..), traversed)
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
import Data.Foldable (traverse_)
import System.Random.MWC (initialize)
import Control.Monad (replicateM)

type Result = [(Double, Vector Double)]

timestep :: Double
timestep = 0.5

args :: Parser String
args = argument str (metavar "COMMAND")

main :: IO ()
main = do
  command <- execParser $ info (args <**> helper) briefDesc
  case command of "process" -> traverse_ doProcess [0..5]
                  "robust" -> traverse_ doRobust [0..5]
                  _ -> error $ "Invalid command: " ++ command

doProcess :: Int -> IO ()
doProcess i = do
  let env = files !! i
  msg <- loadMsg $ "output/"++env++".bin"
  let m = xformMeanMatrix . getMeanMatrix $ msg
      result = evolve m (ends !! i) . normalizeDist $ fromList [1,1,1,1,1,1]
  plotResult env (intervals !! i) result
  writeResult env (intervals !! i) result

doRobust :: Int -> IO ()
doRobust i = do
  let env = files !! i
  msg <- loadMsg $ "output/"++env++".bin"
  let m = xformMeanMatrix . getMeanMatrix $ msg
  gen <- initialize (singleton 3509)
  results <- replicateM 1000 $ do
    dist <- replicateM 6 $ unifD gen
    let result = evolve m 500 . normalizeDist . fromList $ dist
    return . snd . last $ result
  let meanResults = map mean . transpose . map toList $ results
  writeFile ("output/robust-"++env++".txt") . ("x y\n" ++) . unlines . zipWith f [(1::Int)..] $ meanResults
    where f a v = unwords [show a, formatToString (fixed 4) v]

intervals :: [Int]
intervals = [1, 2, 2, 2, 2, 2]

ends :: [Double]
ends = [50, 250, 250, 250, 250, 250]

files :: [String]
files = ["os-pf", "os-ps", "os-pv", "ov-pf", "ov-ps", "ov-pv"]

loadMsg :: String -> IO P.AllResult
loadMsg path = do
  bs <- B.readFile path
  return $ decodeMessageOrDie bs

getMeanMatrix :: P.AllResult -> Matrix Double
getMeanMatrix r = fromLists $ r ^.. P.meanMatrix . traversed . P.row

evolve :: Matrix Double -> Double -> Vector Double -> Result
evolve m end initial = takeWhile continue . zip [0,timestep..] $ iterate (step m) initial
  where continue (i, _) = i < end

step :: Matrix Double -> Vector Double -> Vector Double
step m v = let p = map sum . toLists $ m `multStd2` colVector v
               pMean = sum . zipWith (*) p $ toList v
               delta = zipWith (*) (toList v) $ map (subtract pMean) p
           in normalizeDist . V.zipWith (+) v . fromList $ map (*timestep) delta

normalizeDist :: Vector Double -> Vector Double
normalizeDist v = V.map (/V.sum v) v

plotResult :: String -> Int -> Result -> IO ()
plotResult env intrvl = plotLinesWithX ("output/evo-"++env++".pdf") ["random", "tft", "rl", "num", "fbm", "tom"] . toLines . every intrvl

toLines :: Result -> [[(Double, Double)]]
toLines = transpose . map f
  where f (i, v) = map (i,) $ toList v

writeResult :: String -> Int -> Result -> IO ()
writeResult env intrvl = writeFile ("output/evo-"++env++".txt") . ("i r tft rl num fbm tom\n" ++) . unlines . map f . every intrvl
  where f (i, v) = unwords . ([formatToString (fixed 1) i] ++) . map (formatToString (fixed 4)) $ toList v
