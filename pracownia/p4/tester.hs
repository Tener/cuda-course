module Main where

import System.Environment
import System.Posix.Env
import System.Random
import System.Process
import System.IO

import Data.List.Split

import Control.Applicative
import Control.Monad

import Text.Printf
import Text.EditDistance

import qualified Codec.Binary.UTF8.String as UTF8
import qualified Data.ByteString as ByteString
import qualified Data.ByteString.Lazy as ByteStringL
import qualified Codec.Text.IConv as IConv

runTest :: FilePath -> String -> [String] -> IO ()
runTest nazwaProgramu slownik slowa = do
  -- uruchamiamy proces
  appendFile "run-test-dump.sh" (printf "%s %s\n" nazwaProgramu (unwords (slownik:slowa)))
  putStrLn $  (printf "%s %s\n" nazwaProgramu (unwords (slownik:slowa)) :: String)
  cont <- readProcess nazwaProgramu (map UTF8.encodeString $ slownik:slowa) ""
  -- parsujemy wyjście programu
  let oneProc :: [String] -> ( (String,[(String,String,Int)]), [String] )
      oneProc [] = error "null args"
      oneProc (header:rest) =
          let ["TOTAL", pro, cnt', _time] = splitOn "=" header
              cnt = read cnt' :: Int
              (results,rest') = splitAt cnt rest
              res = map oneResult results
          in
              ((pro, res),rest')
      oneResult line =
          let [_proc,w1,w2,dist] = splitOn ";" line
          in
              (w1,w2,read dist :: Int)

  let (p1,p2_unparsed) = oneProc (lines cont)
      (p2,nothing) = oneProc p2_unparsed

  case nothing of
    [] -> return ()
    _ -> error "Nie sparsowane do końca wejście"

  let formatOneResult pro (w1,w2,dist) (dist2,match) = printf "%s ==> %s -> %s : %d(%d) %s" pro w1 w2 dist dist2 (show match)

  let checkResults (pro,res) = do
        let errors = [ (w1,formatOneResult pro r v) | r@(w1,w2,dist) <- res, v@(_dist2,match) <- [validate w1 w2 dist], not match]
        unless (null errors) (do
                               putStrLn "Error occured:"
                               mapM_ putStrLn (map snd errors)
                               putStrLn "Try it yourself:"
                               let unsetEnv = "env --unset=PORCELAIN "
                               putStrLn (unsetEnv ++ unwords ([nazwaProgramu,slownik]++ map fst errors))
                               error "fail"
                             )
  mapM_ checkResults [p1,p2]



validate :: String -> String -> Int -> (Int, Bool)
validate input output distance = let dist' = levenshteinDistance (EditCosts 1 1 1 1) input output in (dist', dist' == distance)

wczytajSlownik :: FilePath -> IO [ByteStringL.ByteString]
wczytajSlownik zrodlo = (ByteStringL.split (fromIntegral $ fromEnum '\n')
                       . IConv.convert "ISO-8859-2" "UTF-8"
                       . ByteStringL.fromChunks
                       . (:[]))
                     <$> ByteString.readFile zrodlo

main :: IO ()
main = do
  -- zmienne środowiskowe wymagane przez tester
  setEnv "PORCELAIN" "1" True 
  setEnv "SKIP_CPU" "1" True

  let numMachines = 1

  [nazwaProgramu, plikSlownika] <- getArgs
  -- test ze słownika
  let slownikowy = do
         slownik <- wczytajSlownik plikSlownika
         let dlugSlownika = length slownik
         putStrLn (printf "Wczytano %d słów ze słownika %s" dlugSlownika plikSlownika)
         wordsTaken <- randomRs (0, dlugSlownika-1) <$> newStdGen
         let batchCount = 10 `div` numMachines
             batchSize = 10
             batches = take batchCount $ splitEvery batchSize (map (\i -> UTF8.decode $ ByteStringL.unpack $ slownik !! i) wordsTaken)
         mapM_ (\b -> runTest nazwaProgramu plikSlownika b >> putStr "." >> hFlush stdout) batches
         putStr "\nTEST SŁOWNIKOWY OK\n"

  -- test losowy
  let losowy = do
         letters <- randomRs ('a','z') <$> newStdGen
         wordLenghts <- randomRs (1,14) <$> newStdGen
         let batchCount = 10 `div` numMachines
             batchSize = 10
             batches = take batchCount $ splitEvery batchSize (cut letters wordLenghts)

             cut xs (z:zs) = let (a,b) = splitAt z xs in a : cut b zs
             cut _ _ = error "what?"

         mapM_ (\b -> runTest nazwaProgramu plikSlownika b >> putStr "." >> hFlush stdout) batches

         putStr "\nTEST LOSOWY OK\n"
  -- test wybranych kombinacji słów
  let wybrane = do
         runTest nazwaProgramu plikSlownika ["testowość", "przewlokły", "woretewr", "żywotnikowiek", "gżegżułka"]
         runTest nazwaProgramu plikSlownika ["iydnez"]
         putStr "TEST KOMBINACJI OK\n"

  -- uruchomienie wcześniej zdefiniowanych testów
  slownikowy
  losowy
  wybrane
