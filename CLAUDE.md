# SynOmics — Advanced Bioinformatics Platform ("Aistudio Core")

Governing instructions for all code generation, shell execution, and
architectural decisions in this repository. These rules are binding.

---

## 1. Prime Directives & Absolute Constraints

- **ZERO HALLUCINATION MANDATE.** Never fabricate, simulate, or guess analytical
  results (p-values, fold changes, cluster identities, binding affinities, ADMET,
  gene coordinates, …). Every result must be produced by executing real code on
  real data, or fetched from a real source. If a value cannot be computed or
  fetched, return an explicit "not available" — never a placeholder number.
- **NO DEMO / SYNTHETIC DATA.** Do not generate or inject placeholder/example
  data into user-facing surfaces unless the user explicitly asks for it for UI
  testing. Test fixtures live only under `tests/` and are never served to users.
- **DETERMINISTIC EXECUTION.** Every analytical claim must be backed by an
  executable script whose actual stdout/stderr is captured and parsed, with the
  exact parameters and random seeds logged.
- **VERIFY BEFORE SHIP.** Do not claim a code path works unless it has been run.
  When a capability cannot be verified in the current environment (e.g. blocked
  network, missing GPU/binaries), say so explicitly and make the code fail
  honestly at runtime rather than returning fabricated output.
- **FORBIDDEN TERMINOLOGY.** Never use the legacy acronym "iCAT3" or any
  variation. Refer to the system only as the "Advanced Bioinformatics Platform"
  or "Aistudio Core" (product/UI name: SynOmics).

## 2. UI / UX & Visual Design (for new/changed frontend code)

Enforce this palette for new UI work:

| Token | Hex | Use |
| --- | --- | --- |
| Primary background | `#FFFFFF` | Dominant canvas |
| Primary accent | `#0A192F` | Headers, primary nav, structural boundaries |
| Secondary accent | `#00B4D8` | Interactive elements, active states, buttons, key data |
| Neutral | `#F8F9FA` | Panels, tables, subtle borders |

Typography: **Inter** or **Roboto** for prose; **Fira Code** or **JetBrains
Mono** for sequences, code, and tabular data. (Legacy screens use an earlier
cream/emerald palette; migrate opportunistically, do not mass-restyle in one
commit.)

## 3. Core Architecture & Modules

### Module A — Data Ingestion & Conditional Clarification Loop
- On upload (CSV/TSV/H5AD/FASTQ/FASTA/VCF), run a real profiling pass to detect
  data type, sample IDs, and metadata structure.
- **HALT on ambiguity.** If grouping, batch variables, or controls are ambiguous
  or contradictory, do not guess — ask a precise, targeted question.
- Status: FASTA/FASTQ/VCF/CSV/TSV parsing implemented in
  `server/synomics_engine.py::ingest_file` (`POST /api/synomics/ingest-file`).
  H5AD (single-cell AnnData) profiling implemented in `server/h5ad_profiler.py`
  (`POST /api/synomics/ingest-h5ad`, accepts base64 bytes or a path) — reads real
  cell/gene counts, X encoding, obs/var columns, and grouping candidates via h5py,
  and HALTS with a precise clarification question when no unambiguous grouping
  column exists. The broader interactive clarification loop is still to build.

### Module B — Analysis Depth Engine (exponential rigor)
- **L1 Basic (1×):** QC, standard stats (DESeq2/edgeR-style DE, PCA, volcano).
- **L2 Medium (10×):** batch correction (ComBat), mixed models, GSEA, PPI maps.
- **L3 Advanced (100×):** multi-omics fusion, ML feature selection (LASSO/RF),
  single-cell trajectory inference.
- **L4 Discovery (1000×):** hypothesis generation, cross-database meta-analysis
  (Ensembl/GEO/KEGG/UniProt), deep-learning pattern recognition.
- Status: real L1 primitives exist in the engine (Welch+BH DE, hypergeometric
  enrichment, single-cell markers, PCA/ordination). L2–L4 largely to build and
  require a Python scientific stack (scanpy/scipy/statsmodels) on a worker.

### Module C — Deterministic Sandbox & Audit Trail
- Write analysis code to a file, execute it, read the real output.
- Append to `audit_log.json` per session: timestamp, tool/version, exact
  parameters, random seeds, output file paths — for 100% reproducibility.
- Status: `POST /api/synomics/python-exec` runs agent code through
  `server/sandbox_runner.py`, which enforces REAL OS resource limits (RLIMIT_CPU,
  RLIMIT_AS memory, RLIMIT_FSIZE, RLIMIT_CORE=0), a wall-clock timeout, a stripped
  environment (server secrets are NOT visible to the code), and an isolated temp
  cwd. Verified: memory bombs and infinite loops are killed; secrets are invisible.
  The append-only audit trail is live (`server/audit.ts`). Honest scope: kernel
  network namespacing / seccomp syscall filtering are NOT applied (need root/unshare,
  unavailable here) — outbound network from sandboxed code is still governed by the
  environment's egress policy, not blocked at this layer.

### Module D — Publication-Grade Report Generator
- Compile validated results into a 6-section report (Title, Summary,
  Introduction, Methods, Results, Interpretations) exportable to PDF/DOCX/HTML.
- Tech: `python-docx` (Word), WeasyPrint/ReportLab (PDF), Jinja2 + Plotly (HTML).
- Status: HTML (Jinja2) + DOCX (python-docx) + PDF (ReportLab, pure-Python) all
  implemented in `server/report_generator.py` (`POST /api/synomics/report`, pass
  `formats:["html","docx","pdf"]`); renders only real provided content, missing
  sections marked "not provided". Verified: PDF has valid %PDF- magic. Live Plotly
  figures still to add.

### Module E — AI-Native Drug Discovery (CADD) & Virtual Validation
Bridges omics findings into molecular design + rigorous in-silico validation.
- **E1 Target/pocket ID:** derive targets from the user's omics data; fetch
  structures (AlphaFold DB / PDB); predict with AlphaFold3/RoseTTAFold when no
  structure exists; detect pockets (FPocket/SiteMap-equivalent).
- **E2 Generative design + MOBO:** 3D pocket-conditioned diffusion (TargetDiff/
  DiffSBDD) / Bayesian optimization; jointly optimize affinity, synthesizability
  (SA/retro), and ADMET; output a Pareto front.
- **E3 Virtual validation:** equivariant GNN scoring (EquiBind/TANKBind) → short
  MD (OpenMM/GROMACS, RMSD/RMSF) → FEP/TI ΔΔG for top hits → PBPK curves.
- **E4 Depth scaling:** L1 Vina + Lipinski; L2 ensemble docking + ML-ADMET +
  pharmacophore; L3 generative + GNN affinity + short MD; L4 MOBO Pareto + FEP/TI
  + PBPK + retrosynthesis (AiZynthFinder).
- **Chemistry constraints (zero hallucination):** all molecules valid,
  RDKit-sanitizable SMILES/InChI; 3D conformers energy-minimized; never report a
  binding affinity or ADMET value without executing a real scoring/ML script.
- Status: **partial — a real, RDKit-grounded ligand-based + repurposing tier is
  now live and CI-gated (see §5 "Drug discovery & repurposing wave"); the
  structure-based / physics tier (E1 folding, E2 3D generative, E3 docking / MD /
  FEP / PBPK) remains honestly gated.** The live tier computes only real values
  from RDKit/numpy on provided molecules/signatures — descriptors, drug-likeness
  rules, Ertl SA score, PAINS/BRENK/NIH alerts, ECFP Tanimoto screening,
  pharmacophore/scaffold analysis, and CMap/signature-reversal repurposing. It
  NEVER emits a binding affinity, IC50, docking pose, or ΔG — those need
  OpenMM/torch-geometric/DeepChem, GPUs, external weights and open network, none
  available in this build, so `DrugDiscoveryMode`/structure-based routes still
  show honest "requires backend" states and fabricate nothing. Do not ship any
  affinity/ADMET-prediction output that is not the product of a real executed
  scoring/ML pipeline.

## 4. Operational Rules

- **Verify before acting:** confirm a file exists and read it before modifying.
- **Iterative validation:** write a minimal version, run it on a tiny subset,
  then scale to full data.
- **Error transparency:** report the exact error; propose a scientifically sound
  alternative rather than degrading silently.
- **Confidence scoring:** flag findings with p > 0.05 or negligible effect sizes
  as "Preliminary" and explain why.
- **Network:** external DB calls go to real public APIs. If the environment's
  egress policy blocks a host, the route returns an honest error — never a
  fabricated fallback, and never route around the policy.

## 5. Current build reality (do not overclaim)

- Frontend (React 19 + Vite + Tailwind) + Express server + real Python engine.
- Verified real: sequence alignment, differential expression (Welch+BH),
  hypergeometric enrichment, single-cell markers, Ramachandran, phylogenetics,
  MS/MS, ΔΔG (physics), GWAS (λ_GC), microbiome diversity, Kaplan–Meier, MCL,
  ODE; file ingestion; a real agent tool-use loop (`/api/synomics/agent-execute`).
- External DB routes (`/api/synomics/db/*`) are real fetches with honest errors;
  their happy path is unverified until run in an open-egress environment.
- **Verifiable-AI engines (decision made by math, not the LLM):**
  - Adversarial validation (`/adversarial-validate`): permutation-null test of a
    DE hypothesis → deterministic VALIDATED/INVALIDATED/INCONCLUSIVE + veto.
    Verified: real signal validated, pure noise never validated.
  - Neuro-symbolic pathway solver (`/pathway-logic`): deterministic boolean
    SAT/UNSAT + proof trace. Tier-1 GNN edge-weight extractor is NOT built
    (needs trained weights/GPU) — accept edge states/fold-changes as input.
  - Causal discovery (`/causal-discovery`): DirectLiNGAM in numpy, empirically
    validated to recover known DAGs; bootstrap-gated edges; honest 'unavailable'
    without numpy.
  - Tensor-Train compression (`/tensor-compress`): error-bounded compression
    utility with an honest 'approximate' flag. NOT a cell/digital-twin simulator.
  - Enhanced ML adversary (`/adversarial-ml`): classifier overfit test
    (sklearn permutation_test_score) + PCA-vs-covariate confounder check.
  - Neuro-symbolic Tier 1 (`/edge-extraction`): partial-correlation (GraphicalLassoCV)
    edges — direct vs indirect. Tier 2 (`/pathway-logic-z3`): Z3 SMT formal
    SAT/UNSAT proof (in addition to the pure-Python `/pathway-logic`).
  - Boolean attractor analysis (`/boolean-attractors`): exact state-space
    attractors (phenotypes) + perturbation shifts — the deterministic
    "digital twin" replacement (no ODE/PDE fabrication).
  - Causal discovery (`/causal-discovery`): DirectLiNGAM + bootstrap gating.
- Module D report generator (`/report`): 6-section HTML+DOCX from real content only.
- **APEX engines (all code-grounded, honest fallbacks, CI-gated):**
  - Multi-omic Z3 consistency (`/multiomic-consistency`): flags LOGICAL_CONFLICT
    across omics layers and HALTS pathway activation for conflicted genes.
  - Adversarial swarm (`/adversarial-swarm`): ensemble (Welch + Mann-Whitney +
    exact permutation), survivors gated at FDR<0.01 with a swarm survival rate.
  - Robotic protocol (`/robotic-protocol`): Opentrons protocol generation gated
    by physical-constraint validation (volume/slot; oversize auto-split).
  - Self-optimizing compilation (`/accelerate`): runtime Cython acceleration with
    a correctness guard + measured speedup logged to the audit trail.
  - Cryptographic provenance (`/provenance`): SHA-256 manifest of inputs/scripts/
    outputs; the report footer embeds the manifest hash.
- **Final-Frontier engines (all code-grounded, honest fallbacks, CI-gated):**
  - MML model selection (`/mml-select`): parsimony via minimum two-part message length.
  - Circuit verification (`/circuit-verify`): Gillespie SSA + temporal-property
    VERIFIED/VIOLATED with a Wilson CI.
  - PDE residual gate (`/pde-validate`): reaction-diffusion residual → PHYSICALLY
    VALID/INVALID (PINN training itself needs torch/GPU; the enforcement runs here).
  - Assay vision (`/assay-quantify`): deterministic OpenCV quantification (no LLM
    eyeballing) + Bayesian posterior update (`/bayesian-update`).
- **iDiscover engines (monumental frontiers; code-grounded, honest fallbacks, CI-gated):**
  - Biological Git — cellular reversion (`/idiscover/cellular-reversion`):
    Waddington Optimal Transport. Exact EMD via POT (else numpy Sinkhorn, flagged
    `approximate`) → exact Wasserstein "energy" + top per-gene revert commits from
    the barycentric projection. Gene names are exact input columns; strict
    "failed to converge" error on disjoint distributions (no heuristic fallback).
    Verified: analytic 1-D W₂ recovered exactly; known diseased→healthy shifts recovered.
  - GFlowNet generative chemistry (`/idiscover/gflownet-sample`): tabular numpy
    GFlowNet trained with Trajectory Balance, sampling molecules ∝ reward. Every
    candidate is RDKit-sanitizable with a REAL computed QED; invalid samples are
    discarded, nothing fabricated. Tabular tier only — a deep neural GFlowNet needs
    torch/GPU and is NOT claimed. Verified: trained policy concentrates above uniform
    random; all reported QED values match RDKit.
  - Hyper-NOTEARS — hypergraph causal discovery (`/idiscover/hyper-causal-discovery`):
    discovers a Directed Acyclic Hypergraph of multi-way JOINT causes ([A,B]->C
    that pairwise LiNGAM/PCMCI cannot represent) via exogeneity-ordered,
    order-restricted continuous optimization (scipy L-BFGS-B); OR verifies a
    proposed weighted adjacency with the EXACT tr(exp(W∘W))-d acyclicity gate and
    rejects any causal loop with a strict error (no heuristic DAG). Honest scope:
    orienting edges / detecting loops from raw observational data is not
    identifiable in general, so discover returns a certified DAH and loop-detection
    is the verify path. Verified: recovers Z=X*Y joint cause; rejects A→B→C→A loop
    (h=0.131>ε). Requires numpy+scipy.
  - Federated ZKP biomarker discovery (`/idiscover/federated-zkp`): each site runs
    a REAL stratified log-rank survival test on its own private records; only the
    additive (O-E, V) sufficient statistics leave the site — never raw rows. The
    aggregate is secured with REAL Pedersen commitments (additively homomorphic,
    RFC-3526 2048-bit group) + Schnorr/Fiat–Shamir zero-knowledge proofs of
    knowledge, so per-site contributions stay hidden and tamper-evident. Pure
    stdlib. Honest scope: this is a commitment + Sigma-protocol system (integrity
    + ZK proof of knowledge), NOT a general zk-SNARK over an arbitrary predicate
    (needs a proving backend not bundled — not claimed). Verified: real cross-site
    signal detected + cryptographically verified; pooled log-rank matches an
    independent reference; forged proof/aggregate rejected.
  - Manifest at `/idiscover`. All four are also reachable as engine commands
    (`synomics_engine.py cellular_reversion|gflownet_sampling|hyper_causal_discovery|federated_zkp`,
    delegating to the dedicated modules).
  - Frontend surface: `src/components/IDiscoverPanel.tsx` (in the Analysis Hub →
    "iDiscover Frontiers" pipeline). Calls the four real routes and renders real
    computed output with honest error/empty states, on the §2 palette. "Load
    example input" only fills INPUT fields (user-initiated); displayed results
    always come from the backend — nothing is fabricated client-side.
- **De-faked sandbox route:** `/api/synomics/tool-execute` (+ `/api/biomni`,
  `/api/bio` aliases) no longer returns canned/fabricated tool results. It now
  dispatches every `toolId` to the real registry via `invokeTool` (with a UI→tool
  alias map); unmapped tools and missing params return honest errors. The former
  hardcoded DE/single-cell/docking result blocks and the dead
  `generateDomainIntelligence` fabricator have been removed.
- Concordance: 7/7 engine statistics match scipy/statsmodels (VALIDATION_REPORT.md).
- Lint gate: `ruff check server tests` (pyflakes/syntax/imports) runs in CI.
- **Standard-bioinformatics breadth modules (real, CI-gated, added to close the
  Biomni breadth gap):** advanced expression (`expression_advanced.py`: NB-GLM DE,
  GSEA, batch correction, PCA), biostatistics (`biostats.py`: Fisher, chi-square,
  ANOVA, correlation, multiple-testing, power, normality, ROC/AUC, log-rank, Cox),
  sequence/molecular biology (`seqtools.py`: translate, revcomp, GC, ORF, primer Tm,
  restriction map, protein params, codon usage), network biology (`netbio.py`:
  centrality, community detection, shortest path, graph stats, RWR), advanced
  cheminformatics (`cheminfo_advanced.py`: Tanimoto, similarity matrix, substructure
  search, Murcko scaffold, PAINS), machine learning (`ml_analysis.py`: k-means,
  hierarchical, t-SNE, RF importance, LASSO, logistic), variant/population genetics
  (`variant_tools.py`: Hardy-Weinberg, allele frequency, Ts/Tv, VCF summary),
  advanced microbiome (`microbiome_advanced.py`: Chao1, CLR differential abundance,
  rarefaction), structural biology (`structure_tools.py`: summary, radius of
  gyration, contact map, atom distance). Each validated against known ground truth.
- **Breadth wave 3b (real, CI-gated):** time-series/signal (`timeseries_tools.py`:
  autocorrelation, cross-correlation, CUSUM change-point, FFT periodicity, LOWESS,
  linear detrend, moving average), clinical epidemiology (`clinical_tools.py`: odds
  ratio/relative risk, diagnostic metrics, number-needed-to-treat, inverse-variance
  meta-analysis), WGCNA co-expression (`wgcna.py`: soft-threshold, co-expression
  modules, module eigengenes), flow cytometry (`flow_tools.py`: arcsinh transform,
  spillover compensation, gating frequencies, channel summary). Each validated
  against known ground truth.
- **Breadth wave 4 (real, CI-gated):** spatial statistics (`spatial_tools.py`:
  Moran's I, Geary's C, Getis-Ord G, Ripley's K, Moran permutation test),
  pharmacokinetics/enzyme kinetics (`pkpd_tools.py`: NCA, one-compartment fit,
  Michaelis-Menten, Lineweaver-Burk, competitive-inhibition Ki), Bayesian inference
  (`bayes_tools.py`: beta-binomial, normal-normal, Poisson-gamma conjugate updates,
  Bayesian A/B test, BIC Bayes factor), beta-diversity/ordination
  (`beta_diversity.py`: Bray-Curtis, Jaccard, PCoA, PERMANOVA, Mantel), statistical
  power/sample size (`power_tools.py`: two-means, two-proportions, ANOVA,
  correlation), genomic interval arithmetic (`genome_intervals.py`: merge,
  intersect, subtract, coverage, nearest). Each validated against known ground
  truth (e.g. d=0.5/power=0.8 → n≈64; Bray-Curtis of disjoint samples = 1.0;
  Michaelis-Menten recovers Km=10; Moran's I = +1 for a perfectly clustered field).
- **Biomni-derived wave (real, CI-gated, with outcome bundles):** glycoengineering
  (`glyco_tools.py`: N/O-glycosylation motif scans), synthetic biology
  (`codon_tools.py`: codon optimization + CAI), biochemistry
  (`conservation_tools.py`: per-column Shannon-entropy conservation), chronobiology
  (`chrono_tools.py`: cosinor MESOR/amplitude/acrophase), microbial growth
  (`growth_dynamics.py`: logistic + Gompertz fits, generalized Lotka-Volterra),
  genomic prediction (`genomic_prediction.py`: GBLUP/ridge breeding values). Designs
  adapted from the Apache-2.0 Biomni project; implementations original + validated
  against known ground truth (e.g. cosinor recovers MESOR=10/amplitude=5; GBLUP
  accuracy r>0.9 on signal, ~0 on pure noise; conservation entropy 0 bits for a
  conserved column, 2 bits for 4 equiprobable residues).
- **Outcome bundles (Biomni-style output structure):** `server/outcome_bundle.py`
  writes, per invocation (when a tool is called with `outputDir`), a structured
  bundle mirroring Biomni's Results→artifacts shape: `result.json` +
  `research_log.md` (Results), `figures/*.png|svg` (matplotlib on the §2 palette),
  `tables/*.csv`, `code/analysis.py` (a runnable reproducer), `report.html`+`.md`,
  `README.md` (docs), and a SHA-256 `MANIFEST.json` (provenance). It only serializes
  real computed content — never fabricates a figure, row, or value. The six
  Biomni-derived modules emit bundles; future tools opt in via the helper. Every
  emitted report/document/article carries, directly beneath the title, a mandatory
  attribution + citation block (product citation name **Synapse**): "All analyses
  and interpretations … generated with Synapse …" and "Fadiel A, et al. Synapse:
  an integrated full-stack bioinformatics analytical platform. 2026." (constants
  `ATTRIBUTION`/`CITATION` in `outcome_bundle.py`; enforced by CI).
- **Flagship hybrid RNA-seq pipeline (`rnaseq_pipeline.py`):** two honest halves.
  UPSTREAM (`rnaseq_upstream`, Phases 1-3) orchestrates the real toolchain — fastp
  sliding-window trim, STAR index/align with `--sjdbOverhang = ReadLength-1`,
  minimap2 `-ax splice` fed the short-read `SJ.out.tab`, stringtie `--merge` /
  gffcompare, decoy-aware `salmon index` + `salmon quant --validateMappings
  --seqBias --gcBias`; it builds the EXACT commands and runs binaries when present,
  else returns an honest per-step "unavailable" plan (never fabricates counts;
  no aligner binaries exist in this build). DOWNSTREAM (`rnaseq_deseq`, Phase 4)
  is a real, self-contained DESeq2-*style* engine (median-of-ratios size factors →
  parametric mean-dispersion trend + shrinkage → per-gene NB-GLM Wald test → BH FDR
  → normal-prior log2FC shrinkage → VST/PCA) plus `rnaseq_tximport` (transcript→gene
  lengthScaledTPM). It emits the full figure set (PCA, MA, volcano, dispersion,
  p-value histogram, size factors, library sizes, sample-distance & top-gene
  heatmaps), result tables, and a report / DOCX document / full scientific article
  (`outputFormat`). Validated on a labeled NB spike-in fixture (`tests/`, never
  served): recovers known DE genes at ≥0.70 sensitivity with false-positive rate
  ≤0.10 at FDR<0.05. Honest scope: an independent NB-GLM implementation, not the R
  DESeq2 binary; upstream execution requires the bioconda worker image (DEPLOYMENT.md).
- **Epitranscriptomics (`epitranscriptomics.py`):** `m6a_drach_scan` — deterministic
  m6A DRACH consensus (D-R-A-C-H) motif scan reporting candidate methylated-adenosine
  positions; no read data, no invented confidence scores (salvaged as the one REAL
  piece of an uploaded "updated app" whose drug-discovery / neoantigen / retrosynthesis
  "engines" were rejected as fabricators — canned SMILES, magic-number IC50/ΔG/SA
  scores). Validated: GGACU at a known offset → central A at the expected position.
- **Single-cell dynamics + spatial (adopted from open-source repos, real & stack-only):**
  RNA velocity (`rna_velocity.py`: `velocity_estimate` steady-state degradation rate
  gamma + v=u−gamma·s + R²; `velocity_stream_projection` cosine-correlation embedding
  kernel) adapted from dynamo/scVelo — numpy only, validated (known gamma recovered,
  induced cells → positive velocity). Spatial deconvolution (`spatial_deconvolution.py`:
  `nnls_deconvolution` per-spot cell-type proportions via scipy NNLS;
  `ot_map_cells_to_spots` entropic-OT cell↔spot mapping via POT) — the deterministic,
  torch-free form of Tangram's goal; validated (known mixtures + cell origins recovered).
  Honest exclusions from the same review: scvi-tools / torchdrug / Tangram-deep are
  torch/GPU-gated (not runnable here) and were NOT adopted as functional.
- **Advanced wave (real, CI-gated; from the OSS ecosystem + the Biomni source-mining
  workflow's vetted ADOPT list — reimplemented cleanly, fabricated sub-outputs dropped):**
  trajectory inference (`trajectory.py`: diffusion + MST pseudotime), gene-regulatory-
  network inference (`grn_inference.py`: GENIE3, ARACNe+DPI), multi-omics integration
  (`multiomics_integration.py`: SNF, CCA, joint NMF), statistical genetics
  (`mendelian_randomization.py`: MR-IVW, MR-Egger), RNA secondary structure
  (`rna_structure_tools.py`: dot-bracket features), enzyme kinetics & compartmental PK
  (`enzyme_pk_tools.py`: protease Michaelis-Menten from time-course, bi-exponential PK),
  dissolution kinetics (`dissolution_tools.py`), systems-biology dynamics
  (`systems_dynamics_tools.py`: dimerization equilibrium, gene-circuit & signaling ODEs),
  biosignal analysis (`biosignal_tools.py`: ABR/calcium/hemodynamic waveforms),
  molecular biology (`molbio_tools.py`: in-silico PCR, restriction digest, mutation
  finding, primer design, primer binding scan). Each validated against known ground
  truth (e.g. SNF ARI=1; MR-IVW recovers causal 0.5 while Egger flags pleiotropy;
  GENIE3 ranks the true regulator; ARACNe DPI prunes the indirect edge; pseudotime
  Spearman ρ=1.0; dissolution t50=2.5 h; Michaelis-Menten recovers Vmax=100/Km=10).
- **Advanced wave C (real, CI-gated; Biomni ADOPT list):** CRISPR/cloning
  (`crispr_cloning_tools.py`: Cas9 cut-site, indel spectrum, Golden Gate assembly,
  verification-primer design), preclinical pharmacology/tox (`pharmacology_assay_tools.py`:
  xenograft TGI, ATP-luminescence viability, VCOG-CTCAE grading, MIRD α-dosimetry),
  omics association (`omics_assoc_tools.py`: methylome-wide association, Kabsch RMSD
  structure comparison, barcode-seq demultiplex). Validated (TGI=75%; MIRD dose=0.801 Gy;
  Golden Gate assembles known fragments; MWAS recovers the spiked site; Kabsch RMSD≈0
  after a rigid transform).
- **Bioimage + cell-motility wave (real, CI-gated; deterministic OpenCV/numpy, no
  LLM eyeballing):** bioimage analysis (`image_tools.py`: `pixel_distribution`
  intensity histogram/stats, `count_colonies` connected-component blob counting,
  `optical_flow_deformation` Farnebäck dense-flow displacement fields,
  `ciliary_beat_frequency` per-pixel FFT dominant-frequency), cell motility
  (`cell_motility_tools.py`: `cell_motility_metrics` track speed/displacement/
  directionality-ratio/MSD, `cluster_motility_patterns` KMeans track-feature
  clustering). Validated against known ground truth (optical flow recovers a 3-px
  shift → meanFlowX≈2.92; 5 Hz synthetic signal → beatFreq=4.6875 Hz; straight-line
  track → directionalityRatio=1.0; KMeans ARI=1 on separated motility classes).
- **Quantitative proteomics wave (real, CI-gated; numpy/scipy/scikit-learn):**
  `proteomics_tools.py` — `maxlfq_quantify` (MaxLFQ-style label-free protein
  quantification: pairwise median peptide log2-ratios → least-squares abundance
  profile per connected component, anchored to summed intensity; unlinked samples
  reported null, never imputed), `normalize_intensities` (median / quantile
  normalization, log or linear), `impute_missing` (deterministic k-NN, column-min
  fraction, or seeded MinProb down-shifted normal), `differential_abundance`
  (two-group Welch t-test on log2 intensities + BH FDR + log2FC),
  `tmt_protein_rollup` (TMT/iTRAQ reporter-ion PSM→protein median/sum rollup +
  per-channel median normalization). Validated against known ground truth (MaxLFQ
  recovers a 1:2:4 sample ratio and anchors to the summed intensity; median-norm
  equalizes sample medians; quantile-norm makes all samples share one distribution;
  Welch recovers a spiked 4× up-regulated protein at padj<0.05 while a flat protein
  stays non-significant; TMT rollup recovers known per-channel medians).
- **Spatial-transcriptomics neighborhood wave (real, CI-gated; numpy/scipy,
  squidpy-style):** `spatial_neighborhood.py` — `neighborhood_enrichment`
  (kNN spatial graph + seeded label-permutation z-score per cell-type pair;
  positive z = neighbors more than chance, negative = spatial segregation),
  `cooccurrence` (P(type=t within distance d of a center type) / P(type=t) across
  distance bins), `infiltration_score` (fraction of target cells within a radius
  of any source cell + mean source contacts — e.g. immune infiltration into
  tumor), `neighbor_composition` (per-cell-type average neighbor-type composition
  over k nearest neighbors). Validated against known geometry (two segregated
  single-type blobs → within-type z≫0 and cross-type z<0; a checkerboard field →
  cross-type z>0; same-type co-occurrence ratio>1 at short distance; infiltration
  recovers a known 5/10=0.5 fraction; segregated field → neighbor self-fraction
  ≈1.0). The permutation null is seeded and reproducible.
- **Drug discovery & repurposing wave (real, RDKit-grounded, CI-gated; the live
  Module-E ligand-based tier):** ADMET / med-chem (`admet_tools.py`:
  `admet_profile` 13-descriptor RDKit panel — MW/logP/TPSA/HBD/HBA/rotatable
  bonds/aromatic rings/FractionCsp3/molar refractivity/heavy atoms/formal
  charge/rings/QED; `druglikeness_rules` Lipinski/Veber/Ghose/Egan/Muegge pass-fail
  with violated criteria; `synthetic_accessibility` Ertl SA score via the RDKit
  SA_Score contrib; `structural_alerts` PAINS+BRENK+NIH via FilterCatalog),
  ligand-based screening (`chem_screening.py`: `similarity_screen` ECFP Morgan
  Tanimoto virtual screen, `pharmacophore_profile` RDKit feature-factory families,
  `scaffold_clustering` Bemis–Murcko grouping, `diversity_selection` MaxMin picker),
  and repurposing (`drug_repurposing.py`: `connectivity_score` CMap weighted-KS
  score (Lamb 2006) — negative = drug reverses the disease signature;
  `signature_reversal_screen` Spearman-reversal ranking of a drug-signature library;
  `target_based_repurposing` ECFP-Tanimoto guilt-by-association that only ECHOES
  known indications/targets, never invents them). ZERO fabrication — no binding
  affinity/IC50/pose/ΔG is ever produced (that tier stays honestly gated).
  Validated against known ground truth (aspirin QED=0.55 / TPSA=63.6; ethanol
  SA≈1.98; catechol flags ≥1 structural alert; aspirin self-Tanimoto=1.0; benzene/
  toluene/phenol collapse to one Murcko scaffold; CMap score flips +0.965/−0.965
  on mimic vs reversed signatures; exact-negation drug tops the reversal screen).
- **Bayesian optimal experimental design wave (real, CI-gated; numpy/scipy — the
  closed-loop "self-driving lab" DECISION layer):** `experimental_design.py` —
  `bayesian_optimal_design` (rank candidate next-experiments by expected
  information gain / D-optimality under a Bayesian linear model),
  `sequential_active_learning` (greedy batch selection, posterior updated per pick
  via Sherman–Morrison), `d_optimal_selection` (greedy D-optimal subset maximizing
  log det of the information matrix), `space_filling_design` (maximin
  Latin-Hypercube screen via scipy.stats.qmc, seeded). It selects WHICH experiment
  to run to maximally reduce model uncertainty — it never predicts an assay
  outcome (zero fabrication). Validated against known math (an unexplored design
  direction yields the highest EIG; per-step EIG diminishes while cumulative EIG is
  monotone; D-optimal selection raises log det and picks the orthogonal direction;
  LHS points are in-bounds, well-spread and seed-reproducible). Honest scope: this
  is the decision brain of the autonomous-lab loop; the wet-lab actuation
  (robot / plate-reader hardware) and differentiable-ODE digital twins (need
  JAX/torch, not installed) remain out of scope and are not faked.
- **Active-learning loop wave (real, CI-gated; numpy — the stateful self-driving-
  lab brain, "Part 7"):** `active_learning_loop.py` — `propose_next_experiment`
  (fit a Bayesian linear model on the measured (X, y) so far, report coefficient
  estimates + uncertainty, select the next experiment by expected information gain,
  then HALT with `awaitingMeasurement:true` — it NEVER fabricates an assay
  outcome), `assimilate_measurement` (append a caller-supplied REAL measurement,
  refit, report realized information gain + posterior-variance reduction),
  `loop_convergence` (stop when the max predictive uncertainty over a candidate
  pool falls below a tolerance). This bridges the design layer to real
  measurements: the loop advances only on real data. Validated against a known
  linear ground truth (posterior mean recovers true w=[2,−1]; a fresh design
  direction yields more realized information gain than a redundant one; posterior
  trace strictly shrinks on assimilation; convergence flips with the tolerance).
  Honest scope: the wet-lab actuation (robot / plate reader) is out of scope —
  the loop produces the plan and stops for the human/instrument to supply the
  measurement; no outcome is ever invented.
- **Federated meta-analysis wave (real, CI-gated; numpy/scipy — the privacy-
  preserving "federated swarm" core):** `federated_meta.py` — `federated_ttest`
  (pooled two-group Welch t-test from per-site (n, mean, variance) sufficient
  statistics — raw rows never leave a site), `stouffer_meta` (weighted Stouffer
  z-combination), `fisher_meta` (Fisher −2·Σ ln p ~ χ²₂ₖ), `random_effects_meta`
  (DerSimonian–Laird pooled effect + Q / I² / τ²). Because (n, mean, variance) are
  sufficient statistics, the federated result is mathematically IDENTICAL to the
  analysis on the pooled raw data — a real computation, not an approximation, with
  no raw records shared. Validated against exact references (federated t and p
  match scipy's Welch t-test on the concatenated raw data to 1e-9; Fisher matches
  scipy.combine_pvalues; four z=2 → Stouffer Z=4.0; identical sites → τ²=0/I²=0 and
  a tighter CI than any single site). Honest scope: this is the statistics-only
  aggregation layer (integrity via the existing `/idiscover/federated-zkp` Pedersen
  + Schnorr system); general FHE over arbitrary predicates and Ray/GPU petabyte
  scale-out remain infra-gated and are not faked.
- 270 real agent tools in `server/tool_registry.ts`; 80 test suites in CI, plus a
  `tsc --noEmit` type-check gate. See `BIOMNI_COMPARISON.md` for the per-domain
  Biomni↔SynOmics coverage table.
- Everything marked "to build" / "not implemented" above must not be faked.

## 6. Commands

```bash
npm install
npm run dev          # tsx server.ts (API + Vite frontend)
npm run build        # vite build + esbuild ESM server bundle -> dist/
npm start            # node dist/server.mjs
npm run lint         # tsc --noEmit
python tests/engine_smoke.py
npx tsx tests/agent_smoke.ts
npx tsx tests/external_db_smoke.ts
```
