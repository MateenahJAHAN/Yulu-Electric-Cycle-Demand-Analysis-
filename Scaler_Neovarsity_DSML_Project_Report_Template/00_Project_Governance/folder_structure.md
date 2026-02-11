# Folder Structure and Reviewer Navigation Guide

This guide helps reviewers quickly locate key report artifacts.

## 1) Governance

- `reviewer_checklist.md`
  - Final quality gates before submission
  - Coverage checks for statistical validity and business interpretation

## 2) Case Inputs and Traceability

- `01_Case_Study_Inputs/case_study_selection.md`
  - Why these five case studies were selected
- `01_Case_Study_Inputs/source_synopsis/*.md`
  - Summary of what was read from each source repository
  - Core business problem, methods, key insights, and recommendations

## 3) Master 100-Page Template

- `02_Master_100_Page_Template/DSML_100_Page_Master_Template.md`
  - Main page-by-page template (Pages 1 to 100)
- `02_Master_100_Page_Template/page_allocation.csv`
  - Structured mapping between page number, case study, and expected content

## 4) Chapter-Wise Writing Templates

- `03_Chapter_Templates/*.md`
  - 5 files, 20 pages each
  - Can be assigned to team members for parallel writing
  - Merged output gives total 100 pages

## 5) Appendices

- `04_Appendices/appendix_template.md`
  - Reference management
  - Model cards and experiment logs
  - Data dictionary and reproducibility evidence

## Suggested Reviewer Flow

1. Read package `README.md`
2. Verify case-study traceability in `01_Case_Study_Inputs/`
3. Audit structure in `page_allocation.csv`
4. Spot-check random pages in master template
5. Apply final checklist before sign-off
