ThreadEnv <- rlang::env(n.threads = NULL)


setThreadEnv <- function(n.threads) {
  ThreadEnv$n.threads <- n.threads
}


resetThreadEnv <- function() {
  setThreadEnv(NULL)
}


setThreads <- function(n) {
  if (is.null(n)) k <- getDefaultThreads()
  else if (is.numeric(n)) k <- getThreadsN(n)
  else if (is.character(n)) {
    k <- switch(n, 
                "default" = getDefaultThreads(),
                "max" = getMaxThreads(),
                "min" = getMinThreads(),
                stop2("Invalid string specifying number of threads"))
  } else stop2("Invalid number of threads, must be integer, NULL, or character")

  RcppParallel::setThreadOptions(numThreads = k)
  ThreadEnv$n.threads <- k
}


resetThreads <- function() {
  resetThreadEnv()
  RcppParallel::setThreadOptions(numThreads = getDefaultThreads())
}


getDefaultThreads <- function() {
  defaultCRAN <- 2
  getThreadsN(defaultCRAN)
}


getMaxThreads <- function() {
  parallel::detectCores()
}


getMinThreads <- function() {
  getThreadsN(n = 1)
}


getThreadsN <- function(n) {
  ncores <- parallel::detectCores()
  ifelse(n >= ncores, ncores, n) 
}
