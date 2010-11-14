module Main where

import System.Environment
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

wczytajSlownik :: FilePath -> IO [ByteStringL.ByteString]
wczytajSlownik zrodlo = (ByteStringL.split (fromIntegral $ fromEnum '\n')
                       . IConv.convert "ISO-8859-2" "UTF-8"
                       . ByteStringL.fromChunks
                       . (:[]))
                     <$> ByteString.readFile zrodlo

main = do
  [nazwaProgramu, plikSlownika] <- getArgs
  -- test ze słownika
  let slownik = do
         slownik <- wczytajSlownik plikSlownika
         let dlugSlownika = length slownik
         putStrLn (printf "Wczytano %d słów ze słownika %s" dlugSlownika plikSlownika)
         wordsTaken <- randomRs (0, dlugSlownika-1) <$> newStdGen
         let batchCount = 10
             batchSize = 10
             batches = take batchCount $ splitEvery batchSize (map (\i -> UTF8.decode $ ByteStringL.unpack $ slownik !! i) wordsTaken)
         mapM_ (\b -> runTest nazwaProgramu plikSlownika b >> putStr "." >> hFlush stdout) batches
         putStr "\nTEST SŁOWNIKOWY OK\n"

  -- test losowy
  let losowy = do
         letters <- randomRs ('a','z') <$> newStdGen
         wordLenghts <- randomRs (1,14) <$> newStdGen
         let batchCount = 10
             batchSize = 10
             batches = take batchCount $ splitEvery batchSize (cut letters wordLenghts)

             cut xs (z:zs) = let (a,b) = splitAt z xs in a : cut b zs

         mapM_ (\b -> runTest nazwaProgramu plikSlownika b >> putStr "." >> hFlush stdout) batches

         putStr "\nTEST LOSOWY OK\n"
  -- test wybranych kombinacji słów
  let wybrane = do
         runTest nazwaProgramu plikSlownika ["testowość", "przewlokły", "woretewr", "żywotnikowiek", "gżegżułka"]
         runTest nazwaProgramu plikSlownika ["iydnez"]
         putStr "TEST KOMBINACJI OK\n"

  -- uruchomienie wcześniej zdefiniowanych testów
  slownik
  losowy
  wybrane
