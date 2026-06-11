# Organizer-Facing Submission Metadata

This file is intended for workshop organizers only. It contains author-identifying information and form-field answers. It should not be included in the anonymous reviewer supplement unless the organizers explicitly request it.

---

## Title

From Loss to Lookup: Tracing Circuit Formation in a Small Transformer

---

## Paper Type

Long paper, ICML-style workshop submission.

---

## Authors

Nelson Alex

---

## Affiliation

Independent Researcher

---

## Corresponding Email

nelsontharappel@gmail.com

---

## Keywords

Mechanistic Interpretability; Training Dynamics; Transformer Circuits

---

## Additional Keywords

circuit formation; QK/OV analysis; optimizer attribution; symbolic lookup; causal interventions; role-level progress measures; residual stream; circuit emergence

---

## Topic Areas

Understanding model internals; mechanistic discovery methods; training dynamics; causal methods; conceptual/foundational work.

---

## TL;DR

A controlled formation audit tracing a symbolic lookup role from behavior to QK geometry, causal tests, optimizer updates, cross-seed role movement, and contextual write/readout structure.

---

## Abstract

Mechanistic interpretability usually studies finished circuits; this paper studies how a circuit forms during training. I use a controlled symbolic latest-write key-value lookup task, where the model must identify the queried key, retrieve the latest matching support value, and write value identity into the prediction-position residual stream.

I define role-level progress measures for retrieval and write/readout coupling. In the traced reference run, the QK side forms as a low-rank `W_QK` support-value matcher, and first-order attribution using the actual AdamW parameter update tracks QK route growth while the instantaneous raw-gradient / SGD-equivalent direction explains little of the measured movement. Across additional seeds, a similar support-value retrieval role recurs while its implementing head changes.

The write/readout side is causal but different: it is best characterized as a contextual, high-rank prediction-position value-code operation rather than a clean static `W_OV` theorem. The contribution is not a new primitive method, but a controlled formation audit that follows one retrieval role from behavior into route geometry, causal subspaces, optimizer-update attribution, cross-seed address movement, and write/readout structure.

---

## Submission Status

Unpublished; not accepted to an archival venue.

---

## Fast-Track Submission

No. This is not a fast-track submission and has not been accepted to ICML 2026.

---

## Conflicts of Interest

None known.

---

## Reciprocal Reviewer Status

No qualified reciprocal reviewer is available.

I am a solo independent researcher without an institutional coauthor or supervisor. I do not currently have a co-first-author mechanistic interpretability paper and therefore cannot provide a qualified reciprocal reviewer profile. Guidance from the organizers would be appreciated.

---

## LLM Usage Statement

I used large language models as coding, writing, and editing assistants during this work. They helped with draft organization, wording, debugging suggestions, related-work positioning, and preparation of submission-related text.

All experiments were run by the author. All reported numerical results were checked against the corresponding scripts, logs, tables, or analysis artifacts. LLM outputs were treated as suggestions and manually verified before inclusion. LLMs were not used to fabricate experimental results or replace empirical validation.

---

## Code and Data Availability

An anonymous supplement is prepared with training configurations, dataset/probe metadata, compact result tables, figure assets, environment metadata, and reproduction commands for the main claims.

Large raw checkpoint sweeps are not included in the compact supplement. The supplement contains the derived artifacts needed to audit the main tables and figures.

---

## Supplementary Material

The supplementary package contains:

- `README.md`
- `results_ledger.md`
- `artifact_manifest.json`
- training and model configuration metadata
- probe-set metadata
- key result tables
- figure assets
- environment metadata
- reproduction commands for the main analyses

The supplement is anonymized for reviewer use.

---

## OpenReview / Independent Researcher Note

I am an independent researcher without an institutional email address. I was unable to complete normal OpenReview registration because the activation process requested a supervisor, coauthor, or colleague with an existing OpenReview profile and institutional email to vouch for my registration.

The submitted paper PDF and reviewer-facing supplement are anonymized. This metadata file is organizer-facing and contains identity/contact information only for administrative handling.

---

## Camera-Ready Acknowledgement

If accepted, I understand that the camera-ready version must follow the required workshop/ICML formatting instructions.

---

## Email Sharing With Program Chairs

Yes. I authorize sharing the author email with the workshop Program Chairs for submission administration.

---

## Public Release Upon Acceptance

Yes. If accepted, I authorize release of the accepted submission and author information according to the workshop’s publication policy.

---

## License

Default workshop license / CC BY 4.0 if required by the submission system.

---

## Signature

Nelson Alex
