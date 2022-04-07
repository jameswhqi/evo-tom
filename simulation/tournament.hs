{-# LANGUAGE TupleSections, DeriveGeneric, DeriveAnyClass #-}

module Main where

-- base
import Prelude hiding (pi)
import Control.Monad (replicateM, foldM)
import Data.List (sort, foldl', transpose, maximumBy)
import Data.Tuple (swap)
import Text.Printf (printf)
import System.CPUTime (getCPUTime)
import System.IO (hFlush, stdout)
import Control.Exception (evaluate)
import GHC.Generics (Generic)
-- mwc-random
import System.Random.MWC (initialize, GenIO)
-- vector
import Data.Vector.Unboxed (singleton)
-- deepseq
import Control.DeepSeq (NFData, rnf, ($!!))
-- proto-lens
import Data.ProtoLens (defMessage, encodeMessage)
-- microlens
import Lens.Micro ((&), (.~), traversed, (^..), _2, to)
-- bytestring
import qualified Data.ByteString as B
-- containers
import Data.Map.Strict (Map, (!), fromList, toList, adjust, fromAscList, toAscList)
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IM
-- statistics
import Statistics.Distribution (cumulative, complCumulative, logDensity)
import Statistics.Distribution.Normal (normalDistr, NormalDistribution)
-- optparse-applicative
import Options.Applicative ((<**>), Parser, argument, str, metavar, helper, briefDesc, info, execParser)

import qualified Proto.Results as P
import qualified Proto.Results_Fields as P
import Lib


args :: Parser String
args = argument str (metavar "ENV")

main :: IO ()
main = do
  env <- execParser $ info (args <**> helper) briefDesc
  let (otype, ptype) = case env of "os-pf" -> (OStable, PFixed)
                                   "os-ps" -> (OStable, PStable)
                                   "os-pv" -> (OStable, PVar)
                                   "ov-pf" -> (OVar, PFixed)
                                   "ov-ps" -> (OVar, PStable)
                                   "ov-pv" -> (OVar, PVar)
                                   _ -> error $ "Invalid environment: " ++ env
  start <- getCPUTime
  gen <- initialize (singleton 3513)
  result <- runTournament gen otype ptype
  putChar '\n'
  putStrLn . matrixToStr 3 $ meanMatrix result
  putChar '\n'
  putStrLn . matrixToStr 3 $ semMatrix result
  B.writeFile ("output/"++env++".bin") . encodeMessage $ genMessage result
  end <- getCPUTime
  let diff = fromIntegral (end - start) / (10^(12 :: Int))
  printf "\nComputation time: %0.3f sec\n" (diff :: Double)

type Choice = Int
type Choices = (Choice, Choice)
type Payoffs = (Double, Double)
type Role = Int

type Posterior = [PostEntry]
type PostEntry = (PostPoint, Double)
type PostPoint = (Double, Double)

type QTables = [(Int, QTable)]
type QTable = Map QKey Double
type QKey = (QState, Choice)
type QState = [Choices] -- (choice_self, choice_opp), latest first

data Agent = Random
           | TfT { _lastOppChoice :: Choice }
           | NUM { _lambda :: Double }
           | ToM { _lambda :: Double
                 , _posterior :: Posterior
                 }
           | RL { _lambda :: Double
                , _history :: QState
                , _tables :: QTables
                }
           | FBM { _lambda :: Double }
           deriving (Generic, NFData, Eq, Ord, Show)

type Id = Int
type Sample = Bool
type Pair = ((Id, Agent), (Id, Agent))
type Game = [Double]
data OppStability = OStable | OVar
data PayStability = PFixed | PStable | PVar
data GameResult = GameResult { _game :: Game
                             , _choices :: Choices
                             , _agents :: Pair
                             , _payoffs :: Payoffs
                             }
                | GameResult0 { _payoffs :: Payoffs }
                deriving (Generic, NFData, Show)
data PairGamesResult = PairGamesResult [GameResult] Pair
                     | PairGamesResult0 Payoffs
                     deriving (Generic, NFData)
type PairResult = (Pair, [PairGamesResult])
data OVarGamesResult = OVarGamesResult [GameResult] [Agent] [Agent]
                     | OVarGamesResult0 [Payoffs]
                     deriving (Generic, NFData, Show)
data AllResult = OStableResult [PairResult]
               | OVarResult [OVarGamesResult]


mus, sigmaExps :: [Double]
mus = [-2,-1.8..2]
sigmaExps = map (+ logBase 2 0.75) [-3..3]

muPrior, sigmaExpPrior :: NormalDistribution
muPrior = normalDistr 0 1
sigmaExpPrior = normalDistr (logBase 2 0.75) 1.5

initialPosterior :: Posterior
initialPosterior = [((m, 2**s), logDensity muPrior m + logDensity sigmaExpPrior s) | m <- mus, s <- sigmaExps]

initialTables :: [(Int, QTable)]
initialTables = [(l, initialTable l) | l <- [0..3]]

initialTable :: Int -> QTable
initialTable l = fromList [((s, a), 0) | s <- possibleStates l, a <- [0, 1]]

possibleStates :: Int -> [QState]
possibleStates l = replicateM l [(x, y) | x <- [0, 1], y <- [0, 1]]

agents :: [Agent]
agents = [ Random, TfT 0 ]
  ++ map (\l -> RL l [] initialTables) [-1..1]
  ++ map (\l -> NUM l) [-1..1]
  ++ map (\l -> FBM l) [-1..1]
  ++ map (\l -> ToM l initialPosterior) [-1..1]
agentNums :: [Int]
agentNums = [4,4,1,2,1,1,2,1,1,2,1,1,2,1]
agents' :: [Agent]
agents' = concat $ zipWith replicate agentNums agents

pairs :: [Pair]
pairs = comb $ zip [0..] agents

runTournament :: GenIO -> OppStability -> PayStability -> IO AllResult
runTournament gen OStable gs = OStableResult <$> mapM (runPair gen gs) pairs
runTournament gen OVar gs = do
  let (n, n0) = case gs of PFixed -> (0, 5)
                           PStable -> (0, 500)
                           PVar -> (0, 500)
  r <- replicateM n $ runGamesOV gen gs True
  r0 <- fmap concat . replicateM 100 $ do
    r0' <- replicateM n0 $ runGamesOV gen gs False
    logProgress r0'
    return r0'
  return $ OVarResult (r ++ r0)

runPair :: GenIO -> PayStability -> Pair -> IO (Pair, [PairGamesResult])
runPair gen gs p = do
  let (n, n0) = case gs of PFixed -> (0, 100)
                           PStable -> (0, 10000)
                           PVar -> (0, 100)
  r <- replicateM n $ runGamesOS gen gs True p
  r0 <- replicateM n0 $ runGamesOS gen gs False p
  logProgress (r, r0)
  return (p, r ++ r0)

runGamesOS :: GenIO -> PayStability -> Sample -> Pair -> IO PairGamesResult
runGamesOS gen gs sp p = do
  game <- case gs of PStable -> genGame gen
                     _ -> return []
  let gameGen = case gs of PFixed -> return pd
                           PStable -> return game
                           PVar -> genGame gen
  (rs, p') <- foldM (folderOS gen gameGen sp) ([], p) [1..100]
  return $!! if sp
             then PairGamesResult (reverse rs) p'
             else PairGamesResult0 . meanPairs . map _payoffs $ rs

folderOS :: GenIO -> IO Game -> Sample -> ([GameResult], Pair) -> Int -> IO ([GameResult], Pair)
folderOS gen gameGen sp (rs, p) _ = do
  game <- gameGen
  (r, p') <- runGame gen updateOS game sp p
  return (r:rs, p')

runGamesOV :: GenIO -> PayStability -> Sample -> IO OVarGamesResult
runGamesOV gen gs sp = do
  game <- case gs of PStable -> genGame gen
                     _ -> return []
  let gameGen = case gs of PFixed -> return pd
                           PStable -> return game
                           PVar -> genGame gen
      t = [1..100*length agents']  -- folded structure
      am = IM.fromAscList $ zip [0..] agents'  -- agent map
  result <- if sp
            then do (rs, (as0, as1)) <- foldM (folderOVSp gen gameGen) ([], (am, am)) t
                    let ea = map snd . IM.toAscList  -- extract agents
                    return $ OVarGamesResult (reverse rs) (ea as0) (ea as1)
            else do let pm = fromAscList . map (,[]) $ comb [0..length agents-1]  -- payoffs map
                    (rs, _) <- foldM (folderOV gen gameGen) (pm, (am, am)) t
                    return $ OVarGamesResult0 . map (meanPairs . snd) $ toAscList rs
  return $!! result

type AccOVSp = ([GameResult], (IntMap Agent, IntMap Agent))
folderOVSp :: GenIO -> IO Game -> AccOVSp -> Int -> IO AccOVSp
folderOVSp gen gameGen (rs, (as0, as1)) _ = do
  game <- gameGen
  (i, j) <- samplePairWRUnif gen (length agents')
  (r, ((_, ai), (_, aj))) <- runGame gen updateOV game True ((i, as0 IM.! i), (j, as1 IM.! j))
  return (r:rs, (IM.insert i ai as0, IM.insert j aj as1))

type AccOV = (Map (Id, Id) [Payoffs], (IntMap Agent, IntMap Agent))
folderOV :: GenIO -> IO Game -> AccOV -> Int -> IO AccOV
folderOV gen gameGen (rs, (as0, as1)) _ = do
  game <- gameGen
  (i, j) <- samplePairWRUnif gen (length agents')
  (GameResult0 (pi, pj), ((_, ai), (_, aj))) <- runGame gen updateOV game False ((i, as0 IM.! i), (j, as1 IM.! j))
  let i' = reverseId i
      j' = reverseId j
      rs' = if i' <= j'
            then adjust ((pi, pj):) (i', j') rs
            else adjust ((pj, pi):) (j', i') rs
  return (rs', (IM.insert i ai as0, IM.insert j aj as1))

reverseId :: Int -> Int
reverseId i = length . takeWhile (<= i) $ scanl1 (+) agentNums

logProgress :: NFData a => a -> IO ()
logProgress a = do
  evaluate (rnf a)
  putChar '.'
  hFlush stdout

type Updater = Agent -> Game -> Role -> Choices -> Agent
runGame :: GenIO -> Updater -> Game -> Sample -> Pair -> IO (GameResult, Pair)
runGame g u game sp p@((i0, a0), (i1, a1)) = do
  c0 <- choose g a0 game 0
  c1 <- choose g a1 game 1
  let cs = (c0, c1)
      a0' = u a0 game 0 cs
      a1' = u a1 game 1 cs
      r = if sp
          then GameResult game cs p (payoffs game cs)
          else GameResult0 (payoffs game cs)
  return (r, ((i0, a0'), (i1, a1')))

choose :: GenIO -> Agent -> Game -> Role -> IO Choice
choose g Random _ _ = unifB g
choose _ (TfT c) _ _ = return c
choose _ (NUM l) m r = do
  let [p0,p1,p2,p3,p4,p5,p6,p7] = if r == 0
                                  then m
                                  else transposeGame m
  return $ if (p1+p3)*l+p0+p2 > (p5+p7)*l+p4+p6 then 0 else 1
choose _ a@ToM{} m r = do
  let (u0, u1) = tomUtil m r a
  return $ if u0 > u1 then 0 else 1
choose g a@RL{} _ _ = do
  let (u0, u1) = qUtil a
  if u0 == u1
  then unifB g
  else return $ if u0 > u1 then 0 else 1
choose _ (FBM l) m r = do
  let (u0, u1) = fbmUtil m r l
  return $ if u0 > u1 then 0 else 1

-- likelihood of choosing 0
tomLambdaLlhd :: Game -> Role -> PostPoint -> (Double, Double)
tomLambdaLlhd g r (mu, sigma) =
  let (cl, o) = cLambda g (1-r)
      [p0, p1] = (\f -> f (normalDistr mu sigma) cl) <$> [cumulative, complCumulative]
  in if o == LT then (p0, p1) else (p1, p0)

cLambda :: Game -> Role -> (Double, Ordering) -- LT: when lambda < cLambda, choose 0
cLambda g r = let [p0,p1,p2,p3,p4,p5,p6,p7] = if r == 0
                                              then g
                                              else transposeGame g
                  cl = (p0+p2-p4-p6)/(p5+p7-p1-p3)
                  o = compare (p1+p3) (p5+p7)
              in (cl, o)

transposeGame :: Game -> Game
transposeGame = map snd . sort . zip ([1,0,5,4,3,2,7,6] :: [Int])

tomUtil :: Game -> Role -> Agent -> (Double, Double)
tomUtil g r (ToM l p) = ((p0+p1*l)*c0+(p2+p3*l)*c1, (p4+p5*l)*c0+(p6+p7*l)*c1)
  where [p0,p1,p2,p3,p4,p5,p6,p7] = if r == 0
                                    then g
                                    else transposeGame g
        (c0, c1) = tomLambdaLlhd g r pp
        (pp, _) = maximumBy compareSnd p
tomUtil _ _ _ = error "tomUtil must be applied to a ToM agent"

fbmUtil :: Game -> Role -> Double -> (Double, Double)
fbmUtil g r l = ((p0+p1*l)*c0+(p2+p3*l)*c1, (p4+p5*l)*c0+(p6+p7*l)*c1)
  where [p0,p1,p2,p3,p4,p5,p6,p7] = if r == 0
                                    then g
                                    else transposeGame g
        (c0, c1) = tomLambdaLlhd g r (0,0.75)


qLRBase, qLRStep, qDiscount :: Double
qLRBase = 0.05
qLRStep = 0.25
qDiscount = 0.8

qUtil :: Agent -> (Double, Double)
qUtil (RL _ h ts) = let ts' = filter (\(len, _) -> len <= length h) ts
                        [q0, q1] = map mean . transpose $ getQValues h <$> ts'
                    in (q0, q1)
qUtil _ = error "qUtil must be applied to an RL agent"


getQValues :: QState -> (Int, QTable) -> [Double]
getQValues h (l, t) = [t ! (take l h, c) | c <- [0, 1]]

updateOS :: Updater
updateOS (TfT _) _ r (c0, c1) = TfT $ if r == 0 then c1 else c0
updateOS (ToM l p) g r cs = ToM l p'
  where p' = updatePostEntry g r cs <$> p
updateOS (RL l h ts) g r cs@(c0, c1) = RL l h' ts'
  where h' = cs':h
        cs' = if r == 0 then cs else swap cs
        css = if r == 0 then [(0, c1), (1, c1)] else [(c0, 0), (c0, 1)]
        pss = map (payoffs g) css
        rewards = map (util . if r == 0 then id else swap) pss
        util (p0, p1) = p0 + l*p1
        ts' = (\(len, t) -> (len, updateTable h rewards cs' len t)) <$> ts
updateOS a _ _ _ = a

updateOV :: Updater
updateOV a@(ToM _ _) _ _ _ = a
updateOV a b c d = updateOS a b c d

updatePostEntry :: Game -> Role -> Choices -> PostEntry -> PostEntry
updatePostEntry g r (c0, c1) (pp, v) = (pp, v + if log p' < -745 then -745 else log p')
  where c = if r == 0 then c1 else c0
        (p0, p1) = tomLambdaLlhd g r pp
        p' = if c == 0 then p0 else p1

updateTable :: QState -> [Double] -> Choices -> Int -> QTable -> QTable
updateTable h rewards (_, c1) l t =
  if l > length h
  then t
  else let futures = map (\c -> maximum $ getQValues ((c, c1):h) (l, t)) [0, 1]
           updateEntry t' c = adjust f (take l h, c) t'
             where f q = q + lr * (rewards!!c + qDiscount * futures!!c - q)
                   lr = qLRBase + qLRStep * fi l
       in foldl' updateEntry t [0, 1]

payoffs :: Game -> Choices -> Payoffs
payoffs g (c0, c1) = (g!!(cell * 2), g!!(cell * 2 + 1))
  where cell = c0 * 2 + c1

pd :: Game
pd = normalize4 [3,0,5,1]

gameSpace :: String
gameSpace = "all"

genGame :: GenIO -> IO Game
genGame = case gameSpace of "vpd" -> genVpd
                            "sym" -> genSym
                            "all" -> genAll
                            _ -> error $ "Unknow game space: " ++ gameSpace

genVpd :: GenIO -> IO Game
genVpd g = do
  v <- replicateM 4 $ unifD g
  let [b, d, a, c] = sort v
  if b + c < a * 2
    then return $ normalize4 [a,b,c,d]
    else genVpd g

genSym :: GenIO -> IO Game
genSym g = do
  v <- replicateM 4 $ unifD g
  return $ normalize4 v

genAll :: GenIO -> IO Game
genAll g = do
  v <- replicateM 8 $ unifD g
  return $ normalize v

normalize4 :: Game -> Game
normalize4 v = [a,a,b,c,c,b,d,d]
  where [a,b,c,d] = normalize v


-- print results
foldPairPayoffs :: ([Double] -> Double) -> [PairResult] -> [Payoffs]
foldPairPayoffs f rs = rs ^.. traversed . _2 . to (foldPairs f . map fromPairGamesResult)

fromPairGamesResult :: PairGamesResult -> Payoffs
fromPairGamesResult (PairGamesResult rs _) = meanPairs $ map _payoffs rs
fromPairGamesResult (PairGamesResult0 p) = p

foldOVarPayoffs :: ([Double] -> Double) -> [OVarGamesResult] -> [Payoffs]
foldOVarPayoffs f rs = map (foldPairs f) . transpose $ map fromOVarGamesResult rs

fromOVarGamesResult :: OVarGamesResult -> [Payoffs]
fromOVarGamesResult (OVarGamesResult rs _ _) = map (meanPairs . snd) $ toAscList tallied
  where tallied = foldl' f (fromAscList $ map (,[]) (comb [0..length agents-1])) rs
        f as (GameResult _ _ ((i, _), (j, _)) (pi, pj)) =
          let i' = reverseId i
              j' = reverseId j
          in if i' <= j'
             then adjust ((pi, pj):) (i', j') as
             else adjust ((pj, pi):) (j', i') as
        f _ (GameResult0 _) = error "OVarGamesResult can only contain GameResult, not GameResult0"
fromOVarGamesResult (OVarGamesResult0 rs) = rs

resultMatrix :: ([Double] -> Double) -> AllResult -> [[Double]]
resultMatrix f (OStableResult rs) = toMatrix . foldPairPayoffs f $ rs
resultMatrix f (OVarResult rs) = toMatrix . foldOVarPayoffs f $ rs

meanMatrix, semMatrix :: AllResult -> [[Double]]
meanMatrix = resultMatrix mean
semMatrix = resultMatrix sem


-- protobuf
genMessage :: AllResult -> P.AllResult
genMessage rs = defMessage
  & P.allResult .~ f1 rs
  & P.meanMatrix .~ map makeRow (meanMatrix rs)
  & P.semMatrix .~ map makeRow (semMatrix rs)
  where f1 :: AllResult -> P.AllResultOneof
        f1 (OStableResult r) = defMessage & P.oStableResult .~ f2 r
        f1 (OVarResult r) = defMessage & P.oVarResult .~ f3 r
        f2 :: [PairResult] -> P.OStableResult
        f2 r = defMessage & P.pairResults .~ map f4 r
        f3 :: [OVarGamesResult] -> P.OVarResult
        f3 r = defMessage
          & P.oVarAgents .~ map makeAgent agents
          & P.oVarGamesResults .~ map f5 r
        f4 :: PairResult -> P.PairResult
        f4 (p, r) = defMessage
          & P.pair .~ makePair p
          & P.pairGamesResults .~ map f6 r
        f5 :: OVarGamesResult -> P.OVarGamesResultOneof
        f5 (OVarGamesResult r ai aj) = defMessage & P.oVarGamesResult .~ f7 r ai aj
        f5 (OVarGamesResult0 r) = defMessage & P.oVarGamesResult0 .~ f8 r
        f6 :: PairGamesResult -> P.PairGamesResultOneof
        f6 (PairGamesResult r p) = defMessage & P.pairGamesResult .~ f9 r p
        f6 (PairGamesResult0 p) = defMessage & P.pairMeanPayoffs .~ makePayoffs p
        f7 :: [GameResult] -> [Agent] -> [Agent] -> P.OVarGamesResult
        f7 r ai aj = defMessage
          & P.oVarGameResults .~ map makeGameResult r
          & P.finalAgentsi .~ map makeAgent ai
          & P.finalAgentsj .~ map makeAgent aj
        f8 :: [Payoffs] -> P.OVarGamesResult0
        f8 r = defMessage & P.oVarPayoffs .~ map makePayoffs r
        f9 :: [GameResult] -> Pair -> P.PairGamesResult
        f9 r p = defMessage
          & P.pairGameResults .~ map makeGameResult r
          & P.finalPair .~ makePair p

makeRow :: [Double] -> P.Row
makeRow r = defMessage & P.row .~ r

makePair :: Pair -> P.Pair
makePair (a0, a1) = defMessage
  & P.agent0 .~ makeIndexedAgent a0
  & P.agent1 .~ makeIndexedAgent a1

makePairS :: Pair -> P.Pair
makePairS (a0, a1) = defMessage
  & P.agent0 .~ makeIndexedAgentS a0
  & P.agent1 .~ makeIndexedAgentS a1

makePayoffs :: Payoffs -> P.Payoffs
makePayoffs (p0, p1) = defMessage
  & P.payoff0 .~ p0
  & P.payoff1 .~ p1

makeIndexedAgent :: (Id, Agent) -> P.IndexedAgent
makeIndexedAgent (i, a) = defMessage
  & P.id .~ fi i
  & P.iagent .~ makeAgent a

makeIndexedAgentS :: (Id, Agent) -> P.IndexedAgent
makeIndexedAgentS (i, a) = defMessage
  & P.id .~ fi i
  & P.iagent .~ makeAgentS a

makeAgent :: Agent -> P.Agent
makeAgent Random = defMessage & P.random .~ defMessage
makeAgent (TfT c) = defMessage & P.tft .~ m
  where m = defMessage & P.lastOppChoice .~ fi c
makeAgent (NUM l) = defMessage & P.num .~ m
  where m = defMessage & P.lambda .~ l
makeAgent (ToM l p) = defMessage & P.tom .~ m
  where m = defMessage
          & P.lambda .~ l
          & P.posterior .~ map makePostEntry p
makeAgent (RL l h ts) = defMessage & P.rl .~ m
  where m = defMessage
          & P.lambda .~ l
          & P.history .~ map makeChoices h
          & P.tables .~ map makeQTable ts
makeAgent (FBM l) = defMessage & P.fbm .~ m
  where m = defMessage & P.lambda .~ l

makePostEntry :: PostEntry -> P.PostEntry
makePostEntry (pp, v) = defMessage
  & P.postPoint .~ makePostPoint pp
  & P.postValue .~ v

makePostPoint :: PostPoint -> P.PostPoint
makePostPoint (mu, sigma) = defMessage
  & P.mu .~ mu
  & P.sigma .~ sigma

makeAgentS :: Agent -> P.Agent
makeAgentS (ToM _ p) = defMessage & P.tomS .~ m
  where m = defMessage & P.postEntryS .~ makePostEntry (maximumBy compareSnd p)
makeAgentS (RL _ _ ts) = defMessage & P.rlS .~ m
  where m = defMessage & P.qStd .~ map qDiff ts
makeAgentS a = makeAgent a

qStd :: (Int, QTable) -> Double
qStd (_, t) = std (toList t ^.. traversed . _2)

qDiff :: (Int, QTable) -> Double
qDiff (l, t) = rms . map (\s -> t ! (s, 0) - t ! (s, 1)) $ possibleStates l

makeGameResult :: GameResult -> P.GameResult
makeGameResult (GameResult g cs a ps) = defMessage
  & P.game .~ g
  & P.choices .~ makeChoices cs
  & P.agents .~ makePairS a
  & P.payoffs .~ makePayoffs ps
makeGameResult _ = error "makeGameResult can only accept GameResult, not GameResult0"

makeChoices :: Choices -> P.Choices
makeChoices (c0, c1) = defMessage
  & P.choice0 .~ fi c0
  & P.choice1 .~ fi c1

makeQTable :: (Int, QTable) -> P.QTable
makeQTable (l, t) = defMessage
  & P.stateLength .~ fi l
  & P.table .~ map makeQEntry (toList t)

makeQEntry :: (QKey, Double) -> P.QEntry
makeQEntry (k, q) = defMessage
  & P.qKey .~ makeQKey k
  & P.qValue .~ q

makeQKey :: QKey -> P.QKey
makeQKey (s, c) = defMessage
  & P.qState .~ map makeChoices s
  & P.choice .~ fi c
