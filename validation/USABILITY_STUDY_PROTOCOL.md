# MALER human usability study protocol

## Status and reporting boundary

This is an execution-ready protocol, not evidence that human testing has taken
place. Recruitment, consent, ethics review, observations and participant scores
must be completed by the study team before the manuscript may claim a formal
usability study.

## Objective

Determine whether biomedical researchers with limited programming experience
can complete MALER's core workflow accurately, understand the distinction
between development and external validation, and respond appropriately to data
quality and privacy warnings.

## Participants

- Target: at least 20 evaluable participants for a summative study.
- Include biological, translational or clinical researchers who analyse numeric
  biomedical data but do not routinely develop machine-learning software.
- Record prior statistics, programming, machine-learning and web-tool experience.
- Exclude members who developed MALER from the primary usability estimate;
  developer observations may be reported separately as formative testing.

## Ethics and consent

Obtain the institution's determination before recruitment. Use written informed
consent, collect no patient data, and assign random participant identifiers.
Screen recordings are optional and require separate consent. Store study logs
outside the MALER upload cache under the institution's retention policy.

## Standardized tasks

1. Identify the correct analysis type for a provided LUAD/LUSC matrix.
2. Upload the example matrix and explain the displayed sample/feature summary.
3. Choose a scaling option and state why scaling matters for SVM/KNN.
4. Run a candidate-feature analysis and identify the development metric.
5. Download the signed `.maler` bundle.
6. Apply the bundle to a reordered external prediction matrix.
7. Diagnose a deliberately duplicated feature identifier.
8. Diagnose a deliberately missing required feature.
9. Locate the privacy warning and state whether direct identifiers may be uploaded.
10. State whether the output is a validated clinical biomarker.

## Primary endpoints

- Complete success on all critical tasks without facilitator intervention.
- Per-task completion rate.
- Median task completion time.
- Critical-error rate: wrong task, result misinterpretation, PHI upload intent,
  or treating exploratory output as a clinical diagnostic.
- System Usability Scale (SUS), scored using the standard 0-100 transformation.

## Secondary endpoints

- Number and type of recoverable errors.
- Requests for clarification.
- Comprehension of confidence intervals, held-out testing and external validation.
- Confidence rating after each task on a five-point scale.

## Facilitation and logging

Use the same browser, example files and task script for every participant. The
facilitator may only use prewritten neutral prompts. Log task start/end, success,
errors, prompts, abandonment and comments. Two reviewers should independently
classify critical errors; disagreements are resolved before analysis.

## Analysis plan

Report counts and percentages with exact binomial 95% confidence intervals,
median and interquartile range for times, and SUS mean/SD plus median/IQR. Do not
perform subgroup hypothesis tests unless sample sizes were planned for them.
List every critical error and corresponding interface remediation. Re-run failed
critical tasks after remediation in a new participant group when feasible.

## Acceptance criteria fixed before recruitment

- At least 85% completion for each critical workflow task.
- No participant proceeds after a PHI/privacy warning with an identified dataset.
- At least 80% correctly distinguish exploratory candidate signatures from
  validated clinical biomarkers.
- Median SUS at least 68.
- No unresolved severity-high accessibility or data-loss defect.

## Required manuscript evidence

Participant flow, characteristics, ethics determination, protocol version,
completion/error results, SUS distribution, interface changes, missing data and
all deviations from this protocol must be reported. A favorable quotation or
developer demonstration is not a substitute for the predefined endpoints.
