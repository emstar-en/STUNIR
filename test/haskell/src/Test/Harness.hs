{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module      : Test.Harness
Description : Core test harness for STUNIR conformance testing
Copyright   : (c) STUNIR Authors, 2026
License     : MIT

This module provides the core test harness infrastructure for running
STUNIR conformance tests in a deterministic and reproducible manner.
-}

module Test.Harness
    ( -- * Test Types
      TestResult(..)
    , TestCase(..)
    , TestSuite(..)
    , TestReport(..)
    , TestOutcome(..)
      -- * Running Tests
    , runTest
    , runTestSuite
    , runAllSuites
      -- * Assertions
    , assertEqual
    , assertBool
    , assertJust
    , assertRight
    , assertDeterministic
      -- * Test Construction
    , testCase
    , testSuite
      -- * Reporting
    , formatReport
    , summarizeResults
    , exitWithReport
    ) where

import Control.Exception (SomeException, try, evaluate)
import Control.DeepSeq (NFData, force)
import Control.Monad (forM, when)
import Data.Aeson (ToJSON(..), encode)
import Data.ByteString.Lazy (ByteString)
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Data.Time.Clock (UTCTime, getCurrentTime, diffUTCTime)
import System.Exit (exitWith, ExitCode(..))
import System.IO (hPutStrLn, stderr)

-- | Outcome of a single test
data TestOutcome
    = TestPassed
    | TestFailed Text
    | TestSkipped Text
    | TestError Text
    deriving (Show, Eq)

-- | Result of running a single test case
data TestResult = TestResult
    { testResultName     :: !Text
    , testResultOutcome  :: !TestOutcome
    , testResultDuration :: !Double  -- seconds
    , testResultDetails  :: !(Maybe Text)
    } deriving (Show)

-- | A single test case
data TestCase = TestCase
    { testCaseName        :: !Text
    , testCaseDescription :: !Text
    , testCaseAction      :: IO TestOutcome
    }

-- | A test suite containing multiple test cases
data TestSuite = TestSuite
    { testSuiteName  :: !Text
    , testSuiteTests :: ![TestCase]
    }

-- | Complete test report
data TestReport = TestReport
    { reportSuites   :: ![(Text, [TestResult])]
    , reportPassed   :: !Int
    , reportFailed   :: !Int
    , reportSkipped  :: !Int
    , reportErrors   :: !Int
    , reportDuration :: !Double
    , reportTime     :: !UTCTime
    } deriving (Show)

-- | Run a single test case
runTest :: TestCase -> IO TestResult
runTest TestCase{..} = do
    startTime <- getCurrentTime
    result <- try (testCaseAction) :: IO (Either SomeException TestOutcome)
    endTime <- getCurrentTime
    let duration = realToFrac (diffUTCTime endTime startTime)
    case result of
        Left ex -> return TestResult
            { testResultName = testCaseName
            , testResultOutcome = TestError (T.pack $ show ex)
            , testResultDuration = duration
            , testResultDetails = Just $ "Exception: " <> T.pack (show ex)
            }
        Right outcome -> return TestResult
            { testResultName = testCaseName
            , testResultOutcome = outcome
            , testResultDuration = duration
            , testResultDetails = Nothing
            }

-- | Run a complete test suite
runTestSuite :: TestSuite -> IO (Text, [TestResult])
runTestSuite TestSuite{..} = do
    TIO.putStrLn $ "\n=== Running: " <> testSuiteName <> " ==="
    results <- forM testSuiteTests $ \tc -> do
        result <- runTest tc
        printResult result
        return result
    return (testSuiteName, results)

-- | Run all test suites and generate report
runAllSuites :: [TestSuite] -> IO TestReport
runAllSuites suites = do
    startTime <- getCurrentTime
    suiteResults <- mapM runTestSuite suites
    endTime <- getCurrentTime
    
    let allResults = concatMap snd suiteResults
        (passed, failed, skipped, errors) = countOutcomes allResults
        duration = realToFrac (diffUTCTime endTime startTime)
    
    return TestReport
        { reportSuites = suiteResults
        , reportPassed = passed
        , reportFailed = failed
        , reportSkipped = skipped
        , reportErrors = errors
        , reportDuration = duration
        , reportTime = startTime
        }

-- | Count test outcomes
countOutcomes :: [TestResult] -> (Int, Int, Int, Int)
countOutcomes = foldr count (0, 0, 0, 0)
  where
    count r (p, f, s, e) = case testResultOutcome r of
        TestPassed     -> (p + 1, f, s, e)
        TestFailed _   -> (p, f + 1, s, e)
        TestSkipped _  -> (p, f, s + 1, e)
        TestError _    -> (p, f, s, e + 1)

-- | Print a single test result
printResult :: TestResult -> IO ()
printResult TestResult{..} = do
    let symbol = case testResultOutcome of
            TestPassed     -> "\x2713"  -- checkmark
            TestFailed _   -> "\x2717"  -- X
            TestSkipped _  -> "\x25CB"  -- circle
            TestError _    -> "\x2757"  -- exclamation
        msg = case testResultOutcome of
            TestFailed m   -> " - " <> m
            TestSkipped m  -> " - " <> m
            TestError m    -> " - " <> m
            _              -> ""
    TIO.putStrLn $ "  " <> symbol <> " " <> testResultName <> msg

-- | Assert equality
assertEqual :: (Eq a, Show a) => Text -> a -> a -> IO TestOutcome
assertEqual label expected actual
    | expected == actual = return TestPassed
    | otherwise = return $ TestFailed $ 
        label <> ": expected " <> T.pack (show expected) <>
        ", got " <> T.pack (show actual)

-- | Assert boolean condition
assertBool :: Text -> Bool -> IO TestOutcome
assertBool label True  = return TestPassed
assertBool label False = return $ TestFailed label

-- | Assert Just value
assertJust :: Text -> Maybe a -> IO TestOutcome
assertJust _     (Just _)  = return TestPassed
assertJust label Nothing  = return $ TestFailed $ label <> ": expected Just, got Nothing"

-- | Assert Right value
assertRight :: Show e => Text -> Either e a -> IO TestOutcome
assertRight _     (Right _) = return TestPassed
assertRight label (Left e)  = return $ TestFailed $ 
    label <> ": expected Right, got Left " <> T.pack (show e)

-- | Assert deterministic output (run N times and compare)
assertDeterministic :: (Eq a, Show a) => Int -> IO a -> IO TestOutcome
assertDeterministic n action = do
    results <- sequence $ replicate n action
    case results of
        [] -> return $ TestFailed "No results"
        (x:xs) -> if all (== x) xs
            then return TestPassed
            else return $ TestFailed $ "Non-deterministic output across " <> 
                T.pack (show n) <> " runs"

-- | Construct a test case
testCase :: Text -> Text -> IO TestOutcome -> TestCase
testCase name desc action = TestCase name desc action

-- | Construct a test suite
testSuite :: Text -> [TestCase] -> TestSuite
testSuite name tests = TestSuite name tests

-- | Format a report for display
formatReport :: TestReport -> Text
formatReport TestReport{..} = T.unlines
    [ ""
    , "==============================================="
    , "           STUNIR CONFORMANCE REPORT          "
    , "==============================================="
    , ""
    , "Results:"
    , "  Passed:  " <> T.pack (show reportPassed)
    , "  Failed:  " <> T.pack (show reportFailed)
    , "  Skipped: " <> T.pack (show reportSkipped)
    , "  Errors:  " <> T.pack (show reportErrors)
    , ""
    , "Total:     " <> T.pack (show $ reportPassed + reportFailed + reportSkipped + reportErrors)
    , "Duration:  " <> T.pack (show reportDuration) <> "s"
    , ""
    , "Status:    " <> if reportFailed + reportErrors == 0 then "PASSED" else "FAILED"
    , "==============================================="
    ]

-- | Generate summary statistics
summarizeResults :: TestReport -> (Int, Int, Double)
summarizeResults TestReport{..} = 
    (reportPassed, reportPassed + reportFailed + reportSkipped + reportErrors, 
     if reportFailed + reportErrors == 0 then 100.0 
     else fromIntegral reportPassed / fromIntegral (reportPassed + reportFailed) * 100.0)

-- | Exit with appropriate code based on report
exitWithReport :: TestReport -> IO ()
exitWithReport report = do
    TIO.putStrLn $ formatReport report
    exitWith $ if reportFailed report + reportErrors report == 0
        then ExitSuccess
        else ExitFailure 1
