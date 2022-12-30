library(here)
rmarkdown::render(paste0(here(), "/notebooks/analysis.Rmd"), "html_document")
