
library(caret)
library(dplyr)

# Load the saved environment or re-run minimal check if environment not saved (it wasn't).
# So we need to re-load predictions if possible.
# Since I didn't save the .RData, I have to assume the previous run's output is the truth:
# "residual deviance on 3559 degrees of freedom" -> 3564 rows.
# 3564 / 891 = 4.

# Why 4? 
# Maybe I ran 4 models? No, this is per-model.
# Maybe I used repeat CV? No, number=5.
# Maybe tuneLength/tuneGrid?
# Caret keeps predictions for ALL tuning parameters if returnData=TRUE or something?
# Default is ONLY best. But `savePredictions="final"` keeps only best.
# UNLESS... 

cat("Debugging OOF sizes...\n")
# I will rewrite the generation script to be EXPLICIT and check sizes.
# And I will use a simple Weighted Average for V9-Fix.

