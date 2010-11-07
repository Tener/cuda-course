module Main where

import System.Environment
import System.Directory
import System.FilePath
import System.Random
import System.Process
import System.IO

import Data.List.Split

import Control.Applicative
import Control.Monad

import Text.Printf
import Text.EditDistance

import qualified Codec.Binary.UTF8.String as UTF8
{-

Format porcelany:

TOTAL=CPU=1305.641968
CPU;alsdkjal;alaska;4
CPU;qwlkejql;alej;5
TOTAL=GPU=216.992004
GPU;alsdkjal;a;999
GPU;qwlkejql;a;999

-}

runTest nazwaProgramu slownik slowa = do
  -- uruchamiamy proces
  cont <- readProcess nazwaProgramu (map UTF8.encodeString $ slownik:slowa) ""
  -- parsujemy wyjście programu
  let oneProc :: [String] -> ( (String,[(String,String,Int)]), [String] )
      oneProc [] = error "null args"
      oneProc (header:rest) =
          let ["TOTAL", proc, cnt', time] = splitOn "=" header
              cnt = read cnt' :: Int
              (results,rest') = splitAt cnt rest
              res = map oneResult results
          in
              ((proc, res),rest')
      oneResult line = 
          let [_proc,w1,w2,dist] = splitOn ";" line
          in
              (w1,w2,read dist :: Int)

  let (p1,p2_unparsed) = oneProc (lines cont)
      (p2,nothing) = oneProc p2_unparsed

  case nothing of
    [] -> return ()
    _ -> error "Nie sparsowane do końca wejście"

--  return (p1,p2)

  let formatBatch (proc, res) = (map (\(w1,w2,dist) -> printf "%s ==> %s -> %s : %d %s" proc w1 w2 dist (show $ validate w1 w2 dist)) res)
      formatOneResult proc (w1,w2,dist) (dist2,match) = printf "%s ==> %s -> %s : %d(%d) %s" proc w1 w2 dist dist2 (show match)

  let checkResults (proc,res) = do
        let errors = [ (w1,formatOneResult proc r v) | r@(w1,w2,dist) <- res, v@(dist2,match) <- [validate w1 w2 dist], not match]
        unless (null errors) (do
                               putStrLn "Error occured:"
                               mapM_ putStrLn (map snd errors)
                               putStrLn "Try it yourself:"
                               let unsetEnv = "env --unset=PORCELAIN "
                               putStrLn (unsetEnv ++ unwords ([nazwaProgramu,slownik]++ map fst errors))
                               error "fail"
                             )
  mapM_ checkResults [p1,p2]



validate input output distance = let dist' = levenshteinDistance (EditCosts 1 1 1 1) input output in (dist', dist' == distance)

main = do
  [slownik, nazwaProgramu] <- getArgs

  -- test ze słownika -- TODO

  -- test losowy
  letters <- randomRs ('a','z') <$> newStdGen
  wordLenghts <- randomRs (3,10) <$> newStdGen
  let batchCount = 100
      batchSize = 28
      batches = take batchCount $ splitEvery batchSize (cut letters wordLenghts)

      cut xs (z:zs) = let (a,b) = splitAt z xs in a : cut b zs

  mapM_ (\b -> runTest slownik nazwaProgramu b >> putStr "." >> hFlush stdout) batches

  -- test wybranych kombinacji słów
  runTest slownik nazwaProgramu ["testowość", "przewlokły", "woretewr", "żywotnikowiek", "gżegżułka"]

  return ()
