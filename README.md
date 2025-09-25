# Hyperclusters

This public repository contains all code for the project entitled: "Catastrophic “hyperclustering” and recurrent losses: diagnosing U.S. flood insurance insolvency triggers" published in npj Natural Hazards. Specifically, the repository contains all code and associated data for our hyperclustering and recurrent loss analysis. We refer to the USA_return_periods repository for code regarding precipitation return period calculations.

NOTE: This repository and associated code (including the README) is regularly updated and being refined to enhance user experience for easy implementation. For inquiries about access or code use, please reach out directly to an3232@columbia.edu to get the most up-to-date files. Thank you!

## Abstract
Although a cornerstone of U.S. flood risk preparedness since 1968, the National Flood Insurance Program (NFIP), is burdened by insolvency. Despite pricing and risk assessment reforms, systemic failures persist, resulting in the accumulation of billions in federal debt. This study presents an interdisciplinary framework integrating qualitative synthesis, unsupervised machine learning, and game theory to diagnose triggers of insolvency. We identify catastrophic “hyperclustering” as large-scale flood events spanning days to weeks and induced by a common hydrometeorological driver, which dominate claim volumes often in regions of high asset density. We find chronic annual losses arise from recurrent claims, emphasizing the need for proactive managed retreat from high-risk areas. Our findings support targeted NFIP reform and broader risk management, particularly as climate extremes intensify the homeowners’ insurance crisis. We argue that long-term resilience requires aligning financial, structural, and non-structural interventions with distinct regional risk patterns—whether driven by hyperclustering, recurrent losses, or both.

## Publication Link
https://doi.org/10.1038/s44304-025-00136-w

# Repository Contents

## Code

### ST_Cluster.py
This contains all functions for our spatiotemporal clustering analysis and subsequent plotting.

### Getis_Ord.py
This file contains our function for Getis-Ord Hotspot analysis.

### Aggregate_Risk_Weights.py
This contains all functions for our aggregate risk weight index calculations.

### CPI_Adjust.ipynb
This notebook performs operations to adjust claims for inflation before conducting further analysis.

### 2025_Cluster_Update.ipynb
This is our latest code execution of our spatiotemporal clustering analysis. Note the code allows for generation of a range of clusters based on varying parameters. Due to file size constraints, here we provide results for 3 degrees lat/long, 5 days, and 7 minimum clustered points, in line with the main text of the manuscript and our sensitivity with disaster declarations (see the USA_Return_Periods repository for more info). If additional sensitivities are of interest, we recommend re-running this file with the parameters of interest on the latest file of FEMA's redacted claims dataset, or reach out directly for sensitivity data.

### F1 - F7.ipynb
These files are the associated code and analysis for each figure provided in the main text.

## Data
Two post-processed datasets are included in this repository. We plan to periodically update our clustered claims data  as we continue to work with updated FEMA NFIP claims datasets. Please check back periodically for updated files, or reach out directly to an3232@columbia.edu to check for more recent versions. The current version of files is provided in the file name, and will be updated below.

### 'cluster_update_{date}.csv'
This contains a compressed csv for all clustered claims from our analysis. Due to the large file size, only columns 'dateOfLoss', 'latitude', 'longitude', 'cluster', and 'id' are provided (which allows for merging back with the original dataset from OpenFEMA). Please note that we are periodically updating this file as new claims are released. Thus, the version of clusters contained in this file does not necessarily match the exact graphs provided for analysis. For updated graphs, please re-run the graphing code with the latest version of the clustered claims file.

Last Update: 9/24/25 using 'FimaNfipClaims.csv' from OpenFEMA updated 8/05/25.

### 'County_Risk_Scores.csv'
This contains a csv of all counties and their associated insolvency risk metrics, as well as their calculated aggregate risk index. For latest risk indexes, we recommend re-running the analysis for F7 using updated datasets.

# Contact Me!
If you have general questions about the code or data please feel free to reach out and I am always happy to try to do my best to help out. If you're interested in using similar method or working on a new project, I am always looking to collaborate and am happy to contribute more broadly! Email is always in flux - but try me at adam.nayak@columbia.edu, adam.nayak@alumni.stanford.edu, adamnayak1@gmail.com, or feel free to ping me on LinkedIn.
