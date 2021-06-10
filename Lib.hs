{-# LANGUAGE TupleSections #-}

module Lib where

-- base
import Data.List (genericLength, intercalate)
-- mwc-random
import System.Random.MWC (uniform, uniformR, GenIO)
-- microlens
import Lens.Micro ((^..), traversed, _1, _2, (&), (%~), to, set, mapped)
-- containers
import Data.Map.Strict (Map, fromList, (!))
import Data.Set (Set, fromAscList, findMin, deleteMin, deleteFindMax, size)
import qualified Data.Set as S (map)
-- formatting
import Formatting (formatToString, fixed)
-- Chart
import Graphics.Rendering.Chart.Easy (plot, line, def, PlotValue, opaque, AlphaColour, setColors, plot_lines_style, line_width)
-- Chart-cairo
import Graphics.Rendering.Chart.Backend.Cairo (toFile, FileOptions(..), FileFormat(PDF))
-- colour
import Data.Colour.CIE (cieLAB)
import Data.Colour.CIE.Illuminant (d65)
import Data.Colour.SRGB (toSRGB, RGB(..))

comb :: [a] -> [(a, a)]
comb [] = []
comb xss@(x:xs) = map (x,) xss ++ comb xs

combInv :: Int -> Int
combInv l = round $ (sqrt (1 + 8 * (fi l :: Double)) - 1) / 2

combDiff :: [a] -> [(a, a)]
combDiff [_] = []
combDiff (x:xs) = map (x,) xs ++ combDiff xs
combDiff [] = error "Argument of combDiff can't be empty"

unifD :: GenIO -> IO Double
unifD = uniform

unifB :: GenIO -> IO Int
unifB = uniformR (0, 1)

normalize :: [Double] -> [Double]
normalize v = map (/rms w) w
  where w = map (subtract $ mean v) v

noNaN :: [Double] -> [Double]
noNaN = filter (not . isNaN)

mean :: [Double] -> Double
mean v = sum (noNaN v) / genericLength (noNaN v)

rms :: [Double] -> Double
rms v = sqrt $ sum (map (^(2::Int)) (noNaN v)) / genericLength (noNaN v)

std :: [Double] -> Double
std v = sqrt $ sum ((^(2::Int)) . subtract (mean v) <$> noNaN v) / (genericLength (noNaN v) - 1)

sem :: [Double] -> Double
sem v = sqrt $ sum ((^(2::Int)) . subtract (mean v) <$> noNaN v) / (genericLength (noNaN v) - 1) / genericLength (noNaN v)

foldPairs :: ([a] -> b) -> [(a, a)] -> (b, b)
foldPairs f ps = (f $ ps^..traversed._1, f $ ps^..traversed._2)

meanPairs, semPairs :: [(Double, Double)] -> (Double, Double)
meanPairs = foldPairs mean
semPairs = foldPairs sem

compareSnd :: Ord b => (a, b) -> (a, b) -> Ordering
compareSnd (_, x) (_, y) = compare x y

every :: Int -> [a] -> [a]
every _ [] = []
every n (x:xs) = x : every n (drop (n-1) xs)

samplePairWoR :: GenIO -> Int -> IO (Int, Int)
samplePairWoR g l = do
  i <- uniformR (0, l-1) g
  j <- uniformR (0, l-2) g
  let k = if j >= i
          then j + 1
          else j
  return $ if i < k then (i, k) else (k, i)

samplePairWR :: GenIO -> [Double] -> IO (Int, Int)
samplePairWR g ws = do
  let cumws = scanl1 (+) $ map (/sum ws) ws
      f a = length $ takeWhile (< a) cumws
  i <- unifD g
  j <- unifD g
  return (f i, f j)

samplePairWRUnif :: GenIO -> Int -> IO (Int, Int)
samplePairWRUnif g l = do
  i <- uniformR (0, l-1) g
  j <- uniformR (0, l-1) g
  return (i, j)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = hs : chunksOf n ts
  where (hs, ts) = splitAt n xs


createMap :: [(Double, Double)] -> Map (Int, Int) Double
createMap ps = let indices = comb [1..combInv (length ps)]
               in fromList $ concatMap f $ zip indices ps
  where f ((i, j), (p0, p1)) =
          if i == j
          then [((i, j), (p0 + p1) / 2)]
          else [((i, j), p0), ((j, i), p1)]

toMatrix :: [(Double, Double)] -> [[Double]]
toMatrix ps = let mp = createMap ps
                  d = combInv (length ps)
                  mx = [[(x, y) | y <- [1..d]] | x <- [1..d]]
              in mx & traversed
                    . traversed
                    %~ (mp !)

matrixToStr :: Int -> [[Double]] -> String
matrixToStr prec m = intercalate "\n" $
                     m ^.. traversed
                         . to f
  where f l = unwords $
              l ^.. traversed
                  . to (formatToString (fixed prec))


plotLines :: String -> [String] -> [[Double]] -> IO ()
plotLines f ns ps = toFile def f $ do
  setColors colorMap
  sequence_ $ plot <$> zipWith ($) (line <$> ns) (pure . zip xs <$> ps)
    where xs = [1..] :: [Double]

fi :: (Integral a, Num b) => a -> b
fi = fromIntegral

initSet :: Set (Double, Double)
initSet = fromAscList $ filter f [(x, y) | x <- [-100,-95..100], y <- [-100,-95..100]]
  where f (x, y) = inrange . toSRGB $ cieLAB d65 60 x y
        inrange (RGB r g b) = all (\a -> a >= 0 && a <= 1) [r, g, b]

abSeq :: [(Double, Double)]
abSeq = map (head . fst) . takeWhile f $ iterate step ([findMin initSet], deleteMin initSet)
  where step (l, s) = let ds = S.map minD s
                          minD p = (minimum $ map (distSq p) l, p)
                          distSq (x1, y1) (x2, y2) = (x1-x2)^(2::Int) + (y1-y2)^(2::Int)
                          ((_, p'), s') = deleteFindMax ds
                      in (p':l, S.map snd s')
        f (_, s) = size s > 0

colorMap :: [AlphaColour Double]
colorMap = cycle $ map f abSeq
  where f (x, y) = opaque $ cieLAB d65 60 x y

plotLinesWithX :: PlotValue x => String -> [String] -> [[(x, Double)]] -> IO ()
plotLinesWithX f ns ps = toFile (FileOptions (800,600) PDF) f $ do
  setColors colorMap
  sequence_ $ plot . set (mapped.plot_lines_style.line_width) 2 <$> zipWith line ns (pure <$> ps)
