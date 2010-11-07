module Main where

import System.Environment
import System.Directory
import System.FilePath
import System.Process
import System.IO
import Data.List.Split

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

  print nothing

--  return (p1,p2)

  let formatBatch (proc, res) = unlines (map (\(w1,w2,dist) -> printf "%s ==> %s -> %s : %d %s" proc w1 w2 dist (show $ validate w1 w2 dist)) res)
  putStrLn $ formatBatch p1
  putStrLn $ formatBatch p2



validate input output distance = let dist' = levenshteinDistance (EditCosts 1 1 1 1) input output in (dist', dist' == distance)

main = do
  -- test ze słownika
  [slownik, nazwaProgramu] <- getArgs

  print =<< runTest slownik nazwaProgramu ["testowość", "przewlokły", "woretewr", "żywotnikowiek", "gżegżułka"]

  -- test losowy

  -- test wybranych kombinacji słów

  return ()