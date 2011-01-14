module EditDistanceMany

where

import Data.Vector
import Data.Word
import Text.EditDistance (EditCosts(..))
import Prelude hiding (length)

--levenshteinDistance :: EditCosts -> String -> String -> Int

editDistance :: (Eq a) => EditCosts -> (Vector a) -> (Vector a) -> Int
editDistance costs vecA vecB = let startA = enumFromN (0::Int) (length vecA)
                                   startB = enumFromN (0::Int) (length vecB)
                               in
                                   zipDiagonal costs startA startB 

zipDiagonal costs vA vB | length vA == 2 &&
                          length vB == 2     = minV vA vB
                        | otherwise          = let vA' = zipWith min (
                                                   vB' = undefined
                                               in
                                                 zipDiagonal costs vA' vB'

minV _ _ = 0