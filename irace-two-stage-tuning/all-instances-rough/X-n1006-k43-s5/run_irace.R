library(irace)

message("Starting irace...")
scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())
# checkIraceScenario(scenario = scenario)
irace_main(scenario = scenario)

# load results
if (file.exists("irace.Rdata")) {
  load("irace.Rdata")

  best <- iraceResults$allConfigurations[tail(iraceResults$iterationElites, 1)[[1]][1], ]
  write.table(best, file = "best-config.txt", sep = "\t", quote = FALSE, row.names = FALSE)

}
