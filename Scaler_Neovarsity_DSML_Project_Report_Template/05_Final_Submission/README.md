# Final Submission (WOOLF-Style DSML Report)

This folder contains the final project-report build artifacts for the five selected business cases:

1. Delhivery: Feature Engineering  
2. Yulu: Hypothesis Testing  
3. Walmart: Confidence Interval and CLT  
4. Aerofit: Descriptive Statistics and Probability  
5. Netflix: Data Exploration and Visualisation  

## Files

- `scripts/generate_final_report.py`  
  End-to-end generator for charts + 100-page PDF report.
- `charts/`  
  Auto-generated chart assets used in the report.
- `WOOLF_DSML_5_Case_Project_Report.pdf`  
  Final PDF report (generated).
- `report_build_summary.txt`  
  Build summary including page-count and file-size checks.

## Build Instructions

Run from repository root:

```bash
python3 "Scaler_Neovarsity_DSML_Project_Report_Template/05_Final_Submission/scripts/generate_final_report.py"
```

## Submission Constraints Covered

- Minimum 100 pages: enforced by report-generation plan (20 pages x 5 cases)
- Detailed charts and report sections: generated for every case chapter
- Target output size: validated in `report_build_summary.txt` (must be <= 50 MB)
