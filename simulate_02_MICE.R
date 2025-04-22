#!/usr/bin/env Rscript

# Load necessary libraries
library(tableone)
library(mice)
library(parallel)
library(tidyverse)
library(miceadds)
library(Hmisc)
library(DT)
library(ggplot2)
library(dplyr)
library(reshape2)
library(gridExtra)
library(arrow)
library(data.table)

set.seed(101)
################################################################################ Parse command-line arguments for file path and results directory
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("Please provide the data file path and the target directory as arguments.")
  }
  data_path <- args[1]
  results_folder <- args[2]
  
  # Load the data
  block_full <- read_csv(data_path)
  block_full$Site <- as.integer(as.factor(block_full$Site))
  #block_full <- block_full[,c(1:5,7,dim(block_full)[2])]

remove_names <- c("ID", "Age", "Site")
block <- block_full[!(rownames(block_full) %in% remove_names), !(colnames(block_full) %in% remove_names)]

################################################################################ Define numeric columns and create a count matrix
cols <- colnames(block %>% select_if(is.numeric))
n <- length(cols)
count_matrix <- matrix(0, nrow = n, ncol = n, dimnames = list(cols, cols))

# Identify columns with at least one group with all NA values
empty_colnames <- sapply(block_full, function(col) {
  tapply(col, block_full[["Site"]], function(x) all(is.na(x)))
})
cols_with_all_na <- names(apply(empty_colnames, 2, any))[apply(empty_colnames, 2, any)]
zero_na_cols <- colnames(block)[colSums(is.na(block)) == 0]
hcols <- setdiff(setdiff(names(block), cols_with_all_na), zero_na_cols)

# Count non-NA pairs for numeric column pairs
count_non_na_pairs <- function(i, j) {
  sum(complete.cases(block[, c(i, j)]))
}
count_matrix <- outer(seq_len(n), seq_len(n), Vectorize(function(i, j) count_non_na_pairs(i, j)))
count_matrix_m <- melt(as.matrix(count_matrix))

# Compute correlation matrix, correcting for zero-variance columns
zero_variance_cols <- sapply(block %>% select_if(is.numeric), function(x) var(x, na.rm = TRUE) == 0)
cor_df <- cor(block %>% select_if(is.numeric), use = "pairwise.complete.obs")

# Apply thresholding to the count matrix
threshold <- 10
cor_df_th <- cor_df * ifelse(count_matrix < threshold, 0, 1)
cor_df_th[is.na(cor_df_th)] <- 0
diag(cor_df_th) <- 0
cor_df1 <- melt(cor_df_th)

# Function to set top `n` correlations
set_correlations <- function(cor_matrix, n = NULL, threshold = NULL) {
  result_matrix <- matrix(0, nrow = nrow(cor_matrix), ncol = ncol(cor_matrix))
  
  process_row <- function(row, n, threshold) {
    if (!is.null(n)) {
      indices <- order(row, decreasing = TRUE)[1:n]
      top_n_vector <- rep(0, length(row))
      top_n_vector[indices] <- 1
      return(top_n_vector)
    } else if (!is.null(threshold)) {
      row[row > threshold] <- 1
    }
    return(row)
  }
  result_matrix <- t(apply(cor_matrix, 1, process_row, n = n, threshold = threshold))
  return(result_matrix)
}
pred <- set_correlations(abs(cor_df_th), n = threshold)

################################################################################ Edit the prediction matrix
colnames(pred) = rownames(pred)
df_pred <- as.data.table(pred)
df_pred[, Age := 1]
setcolorder(df_pred, c("Age", setdiff(names(df_pred), "Age")))
df_pred[, Site := 0]

new_row <- as.list(rep(0, ncol(df_pred)))
df_pred <- rbind(new_row, df_pred, new_row)
pred_final <- as.data.frame(df_pred)
rownames(pred_final) = colnames(pred_final)
pred_final[hcols, "Site"] <- 0

################################################################################ Impute missing data using MICE
imp0 <- mice(block_full[,2:ncol(block_full)], print = FALSE, maxit = 0)
# imp0$pred <- pred_final

# Define methods appropriately
# method <- imp0$method
# method[cols_with_all_na] <- "pmm"
# method[hcols] <- "pmm" #"norm"#"2lonly.mean" #

imp <- mice(block_full[,2:ncol(block_full)], 
            print = FALSE, 
            pred = imp0$pred, 
            method = imp0$method, 
            maxit = 20, 
            m = 10)
completed_data <- mice::complete(imp, "long", include = FALSE)
completed_data$Site <- as.factor(completed_data$Site)

################################################################################ Save the imputed results
# Create the results folder if it doesn't exist
if (!dir.exists(results_folder)) {
  dir.create(results_folder)
}

  # Define the parquet file name
  file_name <- basename(data_path)
  file_name_no_ext <- sub("\\_blocks.csv$", "", file_name)
  parquet_file <- file.path(results_folder, paste0(file_name_no_ext, "_imputed.parquet"))
  
  # Save the subdataset as a parquet file
  write_parquet(completed_data, parquet_file)
  
  # Optionally print the status of the saved file
  cat("Saved imputed dataset", file_name, "to", parquet_file, "\n")



# Loop over the unique values of the `.imp` variable (representing imputations)
# for (imp in unique(completed_data$.imp)) {
  
#   # Filter the data for the current imputation
#   subdataset <- subset(completed_data, .imp == imp)
  
#   # Define the seed folder (e.g., "seed_1", "seed_2", ..., "seed_10")
#   seed_folder <- file.path(results_folder, paste0("seed_", imp))
  
#   # Create the seed folder if it doesn't exist
#   if (!dir.exists(seed_folder)) {
#     dir.create(seed_folder)
#   }
  
#   # Define the parquet file name (e.g., "imputation_1.parquet", "imputation_2.parquet")
#   file_name <- basename(data_path)
#   file_name_no_ext <- sub("\\_blocks.csv$", "", file_name)
#   parquet_file <- file.path(seed_folder, paste0(file_name_no_ext, "_imputed.parquet"))
  
#   # Save the subdataset as a parquet file
#   write_parquet(subdataset, parquet_file)
  
#   # Optionally print the status of the saved file
#   cat("Saved subdataset for imputation", imp, "to", parquet_file, "\n")
# }