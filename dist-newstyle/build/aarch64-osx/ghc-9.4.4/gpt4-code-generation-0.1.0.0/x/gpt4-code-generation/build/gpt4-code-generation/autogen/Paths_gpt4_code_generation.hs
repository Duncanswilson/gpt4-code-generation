{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -w #-}
module Paths_gpt4_code_generation (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where


import qualified Control.Exception as Exception
import qualified Data.List as List
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude


#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir `joinFileName` name)

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath



bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath
bindir     = "/Users/d/.cabal/bin"
libdir     = "/Users/d/.cabal/lib/aarch64-osx-ghc-9.4.4/gpt4-code-generation-0.1.0.0-inplace-gpt4-code-generation"
dynlibdir  = "/Users/d/.cabal/lib/aarch64-osx-ghc-9.4.4"
datadir    = "/Users/d/.cabal/share/aarch64-osx-ghc-9.4.4/gpt4-code-generation-0.1.0.0"
libexecdir = "/Users/d/.cabal/libexec/aarch64-osx-ghc-9.4.4/gpt4-code-generation-0.1.0.0"
sysconfdir = "/Users/d/.cabal/etc"

getBinDir     = catchIO (getEnv "gpt4_code_generation_bindir")     (\_ -> return bindir)
getLibDir     = catchIO (getEnv "gpt4_code_generation_libdir")     (\_ -> return libdir)
getDynLibDir  = catchIO (getEnv "gpt4_code_generation_dynlibdir")  (\_ -> return dynlibdir)
getDataDir    = catchIO (getEnv "gpt4_code_generation_datadir")    (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "gpt4_code_generation_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "gpt4_code_generation_sysconfdir") (\_ -> return sysconfdir)




joinFileName :: String -> String -> FilePath
joinFileName ""  fname = fname
joinFileName "." fname = fname
joinFileName dir ""    = dir
joinFileName dir fname
  | isPathSeparator (List.last dir) = dir ++ fname
  | otherwise                       = dir ++ pathSeparator : fname

pathSeparator :: Char
pathSeparator = '/'

isPathSeparator :: Char -> Bool
isPathSeparator c = c == '/'
