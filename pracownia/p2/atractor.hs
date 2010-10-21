module Main where

import Codec.Image.PPM
import Control.Applicative
import Control.Monad
import Data.Complex
--import Data.List.Split
import System.Environment
import System.Process

type FL = Float
type CX = Complex FL

splitEvery :: Int -> [a] -> [[a]]
splitEvery n [] = []
splitEvery n xs = let (prev,rest) = splitAt n xs in prev : splitEvery n rest

numToChar :: Int -> Char
numToChar 1 = '1'
numToChar 2 = '2'
numToChar 3 = '3'
numToChar 4 = '4'
numToChar 5 = '5'

numToCol :: Int -> (Int,Int,Int)
numToCol 1 = (0,0,0)
numToCol 2 = (0,153,0)
numToCol 3 = (0,76,153)
numToCol 4 = (153,0,153)
numToCol 5 = (153,76,0)

step :: CX -> CX
step z = (3*z + (z**(-3)))/4

epsilon :: FL
epsilon = 0.001

maxCount :: Int
maxCount = 1000

dist :: CX -> CX -> FL
dist (x1 :+ y1) (x2 :+ y2) = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) ** 0.5

evalPoint :: FL -> FL -> Int
evalPoint x y = go maxCount (x :+ y)
    where go 0 _ = 1
          go n pnt | (1 - magnitude pnt) > epsilon = go (n-1) (step pnt)
                   | dist pnt (1 :+ 0)  <= epsilon = 2
                   | dist pnt (0 :+ 1)  <= epsilon = 3
                   | dist pnt ((-1) :+ 0) <= epsilon = 4
                   | dist pnt (0 :+ (-1)) <= epsilon = 5
                   | otherwise = go (n-1) (step pnt)

main :: IO ()
main = do
  [n,m] <- map read <$> getArgs
  let s = 2 :: FL
      d_x = (2 * s / fromIntegral (m :: Int))
      d_y = (2 * s / fromIntegral (n :: Int))

      m_x x = d_x * (fromIntegral (x::Int)) - s
      m_y y = d_y * (fromIntegral (y::Int)) - s

      solution = splitEvery m [ evalPoint (m_x x) (m_y y) | x <- [1..m], y <- [1..n] ]

      txt = unlines (map (map numToChar) solution)
      img = ppm (map (map numToCol) solution)

  when (n + m < 100) (putStr txt)
  writeFile "out.ppm" img
  _ <- system "convert out.ppm out.png"

  return ()