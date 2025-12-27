# ==============================================================================
# TITANIC V13: SURGICAL CORRECTION
# Goal: Resolve the 7-row disagreement between V4 (0.789) and V11 (0.787).
# Method: Consensus Voting using V3, V4, V11, V12.
# ==============================================================================

library(dplyr)
library(stringr)

# ==============================================================================
# 1. LOAD SUBMISSIONS
# ==============================================================================
cat("Loading submissions...\n")
dfs <- list()
dfs$v4  <- read.csv("c:/Git/kaggle-titanic-competition/submission_v4.csv")
dfs$v11 <- read.csv("c:/Git/kaggle-titanic-competition/submission_v11_seed_avg.csv")
dfs$v12 <- read.csv("c:/Git/kaggle-titanic-competition/submission_v12_robust.csv")
dfs$v3  <- read.csv("c:/Git/kaggle-titanic-competition/submission_v3.csv") # Rule-based

# Sanity Check: Ensure all aligned
stopifnot(all(dfs$v4$PassengerId == dfs$v11$PassengerId))

# ==============================================================================
# 2. IDENTIFY MISMATCHES
# ==============================================================================
# We care primarily where V4 (Best) != V11 (Most Stable)
mismatch_idx <- which(dfs$v4$Survived != dfs$v11$Survived)
ids_mismatch <- dfs$v4$PassengerId[mismatch_idx]

cat("\nFound", length(mismatch_idx), "mismatches between V4 and V11.\n")
cat("Passenger IDs:", paste(ids_mismatch, collapse=", "), "\n")

# ==============================================================================
# 3. DIAGNOSE PATIENT (Load Data for Context)
# ==============================================================================
test  <- read.csv("c:/Git/kaggle-titanic-competition/test.csv",  stringsAsFactors = FALSE)
patients <- test %>% filter(PassengerId %in% ids_mismatch)

# Extract Title for verification
patients$Title <- str_extract(patients$Name, "[a-zA-Z]+\\.")

cat("\n--- DIAGNOSIS ---\n")
print(patients %>% select(PassengerId, Pclass, Sex, Age, SibSp, Parch, Fare, Title))

# ==============================================================================
# 4. SURGICAL VOTE
# ==============================================================================
cat("\n--- VOTING ---\n")
# Create Voting DataFrame
votes <- data.frame(
  PassengerId = ids_mismatch,
  V4 = dfs$v4$Survived[mismatch_idx],
  V11 = dfs$v11$Survived[mismatch_idx],
  V12 = dfs$v12$Survived[mismatch_idx],
  V3 = dfs$v3$Survived[mismatch_idx]
)

# Calculate Mean Vote
votes$MeanVote <- rowMeans(votes[, -1])
votes$FinalDecision <- ifelse(votes$MeanVote >= 0.5, 1, 0)

print(votes)

# ==============================================================================
# 5. ASSEMBLE FINAL SUBMISSION
# ==============================================================================
# Start with V11 (Stability Base)
final_submission <- dfs$v11

# Overwrite with Surgical Decisions
for(i in 1:nrow(votes)) {
  pid <- votes$PassengerId[i]
  decision <- votes$FinalDecision[i]
  
  idx <- which(final_submission$PassengerId == pid)
  original <- final_submission$Survived[idx]
  
  if (original != decision) {
    cat(sprintf("Overriding PID %d: %d -> %d\n", pid, original, decision))
    final_submission$Survived[idx] <- decision
  }
}

cat("\nTotal Overrides Applied:", sum(final_submission$Survived != dfs$v11$Survived), "\n")
write.csv(final_submission, "c:/Git/kaggle-titanic-competition/submission_v13_surgical.csv", row.names = FALSE)

cat("Submission saved to submission_v13_surgical.csv\n")
