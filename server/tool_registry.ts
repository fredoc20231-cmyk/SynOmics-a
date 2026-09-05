import { runEngine, runPythonScript } from './engine_client.ts';
import { ensemblGeneBySymbol, myGeneBySymbol, uniProtByGene, vepByRsId } from './external_db.ts';

/**
 * Typed registry of the REAL analysis tools the agent can call. Each entry maps
 * a stable tool name to an actual engine command. There are no simulated tools
 * here — invoking one runs genuine computation in synomics_engine.py.
 */
export interface ToolParam {
  type: 'string' | 'number' | 'array' | 'object' | 'boolean';
  description: string;
  required?: boolean;
}

export interface ToolSpec {
  name: string;
  category: string;
  description: string;
  /** For engine-backed tools: the synomics_engine.py command to run. */
  engineCommand?: string;
  /** For JS-native tools (e.g. external DB clients): a real async handler. */
  handler?: (input: any) => Promise<any>;
  parameters: Record<string, ToolParam>;
}

export const TOOL_REGISTRY: ToolSpec[] = [
  {
    name: 'ingest_file',
    category: 'Data ingestion',
    description: 'Parse an uploaded FASTA/FASTQ/VCF/CSV/TSV file into structured records with summary stats and routing suggestions.',
    engineCommand: 'ingest_file',
    parameters: {
      filename: { type: 'string', description: 'Original file name (used for format detection).' },
      content: { type: 'string', description: 'Raw file text content.', required: true },
    },
  },
  {
    name: 'align_sequences',
    category: 'Sequence analysis',
    description: 'Pairwise sequence alignment (Needleman–Wunsch or Smith–Waterman) with BLOSUM62 scoring.',
    engineCommand: 'align_sequences',
    parameters: {
      seq1: { type: 'string', description: 'First sequence.', required: true },
      seq2: { type: 'string', description: 'Second sequence.', required: true },
      method: { type: 'string', description: "'needleman_wunsch' (global) or 'smith_waterman' (local)." },
      seq_type: { type: 'string', description: "'protein' or 'nucleotide'." },
    },
  },
  {
    name: 'phylogenetic_tree',
    category: 'Sequence analysis',
    description: 'Build a phylogenetic tree from >=3 sequences (Jukes–Cantor distance + neighbor-joining, Newick output).',
    engineCommand: 'phylogenetic_tree',
    parameters: {
      taxa: { type: 'object', description: 'Map of taxon name -> sequence.', required: true },
      method: { type: 'string', description: "Tree method (default 'neighbor_joining')." },
    },
  },
  {
    name: 'differential_expression',
    category: 'Transcriptomics',
    description: 'Differential expression (log2FC + Welch t-test on log2(count+1) + Benjamini–Hochberg FDR) for a two-group design.',
    engineCommand: 'deseq2',
    parameters: {
      counts: { type: 'object', description: 'Map of sample -> array of counts (one value per gene, gene order shared).', required: true },
      conditions: { type: 'array', description: 'Group label per sample, aligned to counts keys.', required: true },
    },
  },
  {
    name: 'pathway_enrichment',
    category: 'Functional genomics',
    description: 'Hypergeometric gene-set enrichment against provided ontology terms.',
    engineCommand: 'syngo_enrichment',
    parameters: {
      genes: { type: 'array', description: 'Input gene symbols.', required: true },
      terms: { type: 'array', description: 'Ontology terms (id, name, genes[]). Optional; a default set is used if omitted.' },
    },
  },
  {
    name: 'single_cell',
    category: 'Single-cell',
    description: 'Single-cell pipeline: log-CPM, HVG selection and Welch t-test cluster markers.',
    engineCommand: 'scanpy_singlecell',
    parameters: {
      rawMatrix: { type: 'array', description: 'Genes x cells expression matrix.', required: true },
      geneNames: { type: 'array', description: 'Gene names (rows).' },
      cellTypes: { type: 'array', description: 'Cell type / cluster label per cell.' },
    },
  },
  {
    name: 'gwas',
    category: 'Genomics & genetics',
    description: 'GWAS summary-statistics analysis: -log10(P), genomic inflation λ_GC, lead loci.',
    engineCommand: 'gwas',
    parameters: {
      summaryStats: { type: 'array', description: 'Array of {rsid, chr, pos, pvalue}.', required: true },
      trait: { type: 'string', description: 'Trait / phenotype name.' },
      sigThreshold: { type: 'number', description: 'Genome-wide significance threshold (default 5e-8).' },
    },
  },
  {
    name: 'microbiome',
    category: 'Microbiome',
    description: 'Microbiome diversity: Shannon/Simpson/Chao1/Pielou, Bray–Curtis dissimilarity, PCoA ordination.',
    engineCommand: 'microbiome',
    parameters: {
      samples: { type: 'array', description: 'Array of {sampleId, group, abundances{taxon:count}}.', required: true },
      method: { type: 'string', description: "Beta-diversity metric (default 'bray_curtis')." },
    },
  },
  {
    name: 'mass_spec',
    category: 'Proteomics',
    description: 'In-silico tryptic digest and b/y ion MS2 fragmentation of a protein sequence.',
    engineCommand: 'msms_fragment',
    parameters: {
      proteinSequence: { type: 'string', description: 'Protein sequence to digest and fragment.', required: true },
    },
  },
  {
    name: 'ramachandran_contact',
    category: 'Structural biology',
    description: 'Ramachandran phi/psi dihedral analysis and residue contact map from PDB text.',
    engineCommand: 'ramachandran_contact',
    parameters: {
      pdbText: { type: 'string', description: 'PDB file text.' },
      pdb_id: { type: 'string', description: 'PDB accession (alternative to pdbText).' },
    },
  },
  {
    name: 'mutagenesis_ddg',
    category: 'Structural biology',
    description: 'Physics-based in-silico ΔΔG for a point mutation (VdW / electrostatics / solvation / entropy).',
    engineCommand: 'mutagenesis_ddg',
    parameters: {
      gene: { type: 'string', description: 'Gene / protein symbol.' },
      wildtype: { type: 'string', description: 'Wild-type residue (1-letter).' },
      position: { type: 'number', description: 'Residue position.' },
      mutant: { type: 'string', description: 'Mutant residue (1-letter).' },
    },
  },
  {
    name: 'network_topology',
    category: 'Systems biology',
    description: 'Network topology metrics (degree, centrality, components) for an interaction graph.',
    engineCommand: 'network_topology',
    parameters: {
      nodes: { type: 'array', description: 'Node identifiers.', required: true },
      edges: { type: 'array', description: 'Array of [source, target] pairs.', required: true },
    },
  },
  {
    name: 'markov_clustering',
    category: 'Systems biology',
    description: 'Markov clustering (MCL) of an interaction graph into modules.',
    engineCommand: 'markov_clustering',
    parameters: {
      nodes: { type: 'array', description: 'Node identifiers.', required: true },
      edges: { type: 'array', description: 'Array of [source, target] pairs.', required: true },
      inflation: { type: 'number', description: 'MCL inflation parameter (default 2.0).' },
    },
  },
  {
    name: 'kaplan_meier',
    category: 'Clinical / survival',
    description: 'Kaplan–Meier survival estimate with exact chi-square(1) log-rank test.',
    engineCommand: 'kaplan_meier',
    parameters: {
      gene: { type: 'string', description: 'Gene used to stratify.' },
      strata: { type: 'string', description: "Stratification (default 'expression_quantile')." },
    },
  },
  {
    name: 'ode_simulate',
    category: 'Systems biology',
    description: 'Biophysical ODE simulation (RK4) of conductance dynamics.',
    engineCommand: 'ode_simulate',
    parameters: {
      gene: { type: 'string', description: 'Gene / channel.' },
      mode: { type: 'string', description: 'Perturbation mode (e.g. Knockout).' },
      duration_ms: { type: 'number', description: 'Simulation duration in ms.' },
    },
  },
  {
    name: 'pathway_logic',
    category: 'Verifiable AI / neuro-symbolic',
    description: 'Deterministically evaluate pathway activation with a boolean logic solver (AND/OR/NOT over gene up/down states). Returns SATISFIABLE/UNSATISFIABLE + a formal proof trace. No LLM guessing.',
    engineCommand: 'pathway_logic',
    parameters: {
      foldChanges: { type: 'object', description: 'Map of gene -> fold-change (states derived by threshold).' },
      geneStates: { type: 'object', description: "Alternative: explicit map of gene -> 'up'|'down'|'neutral'." },
      threshold: { type: 'number', description: 'Fold-change threshold for up/down (default 1.0).' },
      pathways: { type: 'array', description: 'Pathways: [{id, name, rule}] where rule is a boolean expression.', required: true },
    },
  },
  {
    name: 'circuit_verify',
    category: 'Verifiable AI / synthetic biology',
    description: 'Formally verify a synthetic genetic circuit: Gillespie SSA (exact CTMC) + Monte-Carlo temporal-property check (e.g. P(species reaches N by time T) >= p). Reports VERIFIED/VIOLATED with a Wilson CI. numpy.',
    handler: (i) => runPythonScript('server/circuit_verify.py', i),
    parameters: {
      reactions: { type: 'array', description: '[{reactants:{s:c}, products:{s:c}, rate}] mass-action reactions.', required: true },
      initialState: { type: 'object', description: 'Map species -> initial molecule count.', required: true },
      property: { type: 'object', description: '{species, comparator, threshold, byTime, targetProbability}.' },
      maxTime: { type: 'number', description: 'Simulation horizon.' },
      nRuns: { type: 'number', description: 'Monte-Carlo runs (default 2000).' },
      seed: { type: 'number', description: 'Random seed (default 1337).' },
    },
  },
  {
    name: 'adversarial_swarm',
    category: 'Verifiable AI / validation',
    description: 'Evolutionary falsification swarm: an ensemble of statistical models (Welch t, Mann-Whitney U, exact permutation). Only genes significant under EVERY model at strict FDR<0.01 survive; each is tagged with its swarm survival rate. Requires scipy.',
    handler: (i) => runPythonScript('server/swarm.py', i),
    parameters: {
      counts: { type: 'object', description: 'Map gene -> per-sample counts.', required: true },
      conditions: { type: 'array', description: 'Group label per sample (exactly two groups).', required: true },
      fdr: { type: 'number', description: 'FDR threshold (default 0.01).' },
      nResamples: { type: 'number', description: 'Monte-Carlo resamples if exact permutation is infeasible.' },
      seed: { type: 'number', description: 'Random seed (default 1337).' },
    },
  },
  {
    name: 'adversarial_ml',
    category: 'Verifiable AI / validation',
    description: 'ML adversary: a cross-validated classifier overfit test (sklearn permutation_test_score) plus a PCA-vs-covariate batch-confounder check. VALIDATED/INVALIDATED/INCONCLUSIVE + veto. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/adversary.py', i),
    parameters: {
      counts: { type: 'object', description: 'Map gene -> per-sample counts.', required: true },
      conditions: { type: 'array', description: 'Group label per sample.', required: true },
      covariates: { type: 'object', description: 'Optional map covariateName -> per-sample values (e.g. batch) for confounder testing.' },
      nPermutations: { type: 'number', description: 'Permutations for the overfit test (default 1000).' },
      seed: { type: 'number', description: 'Random seed (default 1337).' },
    },
  },
  {
    name: 'accelerate_kernel',
    category: 'Performance / self-optimizing',
    description: 'Compile a slow numeric kernel to C via Cython at runtime, run it, and report the measured speedup vs pure Python. Correctness is asserted against the reference before any speedup is claimed. Requires Cython + a C compiler.',
    handler: (i) => runPythonScript('server/accelerate.py', i),
    parameters: {
      kernel: { type: 'string', description: "Kernel name (e.g. 'sum_sq_pairwise')." },
      n: { type: 'number', description: 'Problem size (default 2000).' },
      seed: { type: 'number', description: 'Random seed (default 1337).' },
      thresholdSeconds: { type: 'number', description: 'Slowness threshold to flag (default 60).' },
    },
  },
  {
    name: 'boolean_attractors',
    category: 'Verifiable AI / state-space',
    description: 'Exact Boolean-network attractor analysis: enumerate the state space to find fixed-point phenotypes and cyclic attractors with basin sizes, and how they shift under node perturbations (drug knockout/activation). Deterministic state-space simulation, not a fabricated digital twin.',
    engineCommand: 'boolean_attractors',
    parameters: {
      nodes: { type: 'array', description: 'Node names (<=20).' },
      rules: { type: 'object', description: 'Map node -> boolean update rule (AND/OR/NOT over {node} / {const}).', required: true },
      perturbations: { type: 'array', description: 'Optional: [{fix:{node:0|1}}] to recompute attractors under perturbation.' },
    },
  },
  {
    name: 'adversarial_validate',
    category: 'Verifiable AI / validation',
    description: 'Adversarially validate a two-group differential-expression hypothesis via a label-permutation null; returns a deterministic VALIDATED/INVALIDATED/INCONCLUSIVE verdict, confidence, and auto-veto. No LLM in the decision.',
    engineCommand: 'adversarial_validate',
    parameters: {
      counts: { type: 'object', description: 'Map of gene -> per-sample counts.', required: true },
      conditions: { type: 'array', description: 'Group label per sample.', required: true },
      nPermutations: { type: 'number', description: 'Permutations for the null (default 1000).' },
      fdrThreshold: { type: 'number', description: 'FDR cutoff for significance (default 0.05).' },
      seed: { type: 'number', description: 'Random seed for reproducibility (default 1337).' },
    },
  },
  {
    name: 'assay_quantify',
    category: 'Wet-lab automation / vision',
    description: 'Deterministic OpenCV quantification of a physical assay image (Otsu/threshold + contour intensity), no LLM eyeballing. Returns per-region area/centroid/mean-intensity. Requires opencv.',
    handler: (i) => runPythonScript('server/vision_assay.py', { ...i, task: 'quantify_image' }),
    parameters: {
      imageBase64: { type: 'string', description: 'Assay image (PNG/JPEG) as base64.', required: true },
      minArea: { type: 'number', description: 'Minimum contour area to report.' },
      threshold: { type: 'number', description: 'Fixed background threshold; Otsu if omitted.' },
    },
  },
  {
    name: 'bayesian_update',
    category: 'Verifiable AI / inference',
    description: 'Conjugate Bayesian posterior update (Beta-Binomial for proportions, Normal-Normal for continuous) that folds physical assay results into posterior probabilities. Requires numpy/scipy.',
    handler: (i) => runPythonScript('server/vision_assay.py', { ...i, task: 'bayesian_update' }),
    parameters: {
      model: { type: 'string', description: "'beta_binomial' or 'normal'.", required: true },
      prior: { type: 'object', description: 'Prior params ({alpha,beta} or {mean,var}).' },
      data: { type: 'object', description: 'Observed data ({successes,trials} or {values,obsVar}).', required: true },
    },
  },
  {
    name: 'molecule_descriptors',
    category: 'Drug discovery / cheminformatics',
    description: 'Compute REAL molecular descriptors from a SMILES via RDKit (MW, cLogP, TPSA, HBD/HBA, rotatable bonds, QED, Lipinski/Veber). Docking/binding affinity is NOT fabricated (needs a real docking engine). Requires rdkit.',
    handler: (i) => runPythonScript('server/drug_descriptors.py', i),
    parameters: {
      smiles: { type: 'string', description: 'Ligand SMILES string.', required: true },
      name: { type: 'string', description: 'Optional compound name.' },
    },
  },
  {
    name: 'robotic_protocol',
    category: 'Wet-lab automation',
    description: 'Generate an Opentrons (apiLevel 2) liquid-handling protocol from a transfer plan AFTER deterministically verifying physical constraints (volume<=pipette capacity with auto-split, unique deck slots within capacity). Emits no protocol if the plan is physically invalid.',
    handler: (i) => runPythonScript('server/robotics.py', i),
    parameters: {
      pipette: { type: 'object', description: '{model, maxVolume, minVolume}.', required: true },
      labware: { type: 'array', description: '[{name, slot}] deck placements.' },
      transfers: { type: 'array', description: '[{source, dest, volume}] liquid transfers.', required: true },
      deckSlots: { type: 'number', description: 'Available deck slots (default 11 for OT-2).' },
    },
  },
  {
    name: 'provenance_manifest',
    category: 'Reporting / provenance',
    description: 'Build a cryptographic provenance manifest (SHA-256 of inputs, scripts, outputs + a manifest hash) so results are tied to the exact bytes that produced them. Pure stdlib.',
    handler: (i) => runPythonScript('server/provenance.py', i),
    parameters: {
      inputs: { type: 'object', description: 'Map name -> input value.' },
      scripts: { type: 'array', description: 'List of script file paths to hash.' },
      outputs: { type: 'object', description: 'Map name -> output value.' },
      sessionId: { type: 'string', description: 'Optional session id.' },
    },
  },
  {
    name: 'generate_report',
    category: 'Reporting',
    description: 'Render a 6-section publication-grade report (Title/Summary/Introduction/Methods/Results/Interpretations) to HTML + DOCX from REAL provided content. Renders only what is passed; missing sections marked "not provided". Requires jinja2/python-docx.',
    handler: (i) => runPythonScript('server/report_generator.py', i),
    parameters: {
      title: { type: 'string', description: 'Report title.', required: true },
      summary: { type: 'string', description: 'Executive summary.' },
      introduction: { type: 'string', description: 'Biological context / hypothesis.' },
      methods: { type: 'string', description: 'Exact tools, params, tests.' },
      results: { type: 'string', description: 'Objective findings (real, computed).' },
      interpretations: { type: 'string', description: 'Significance, limitations, next steps.' },
      tables: { type: 'array', description: 'Optional [{title, columns, rows}] of real results.' },
      formats: { type: 'array', description: "Subset of ['html','docx'] (default both)." },
    },
  },
  {
    name: 'edge_extraction',
    category: 'Verifiable AI / neuro-symbolic',
    description: 'Tier-1 grounding: partial-correlation (sparse inverse covariance, GraphicalLassoCV) edge extraction that separates direct from indirect associations. Not a neural network. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/neuro_symbolic.py', { ...i, task: 'edge_extraction' }),
    parameters: {
      data: { type: 'array', description: 'Matrix rows=samples, cols=variables (samples > variables).', required: true },
      variables: { type: 'array', description: 'Variable names (optional).' },
      threshold: { type: 'number', description: 'Absolute partial-correlation edge threshold (default 0.1).' },
    },
  },
  {
    name: 'multiomic_consistency',
    category: 'Verifiable AI / neuro-symbolic',
    description: 'Reconcile multi-omic layers with Z3: flag LOGICAL_CONFLICT where layers contradict (e.g. transcript up but protein down) and HALT pathway activation for affected genes. Requires z3-solver.',
    handler: (i) => runPythonScript('server/neuro_symbolic.py', { ...i, task: 'multiomic_consistency' }),
    parameters: {
      layers: { type: 'object', description: 'Map layerName -> {gene: foldChange|state} for >=2 omics layers.', required: true },
      threshold: { type: 'number', description: 'Fold-change threshold for up/down (default 1.0).' },
      pathways: { type: 'array', description: 'Optional pathways [{id,name,rule}] evaluated on consistent genes only.' },
    },
  },
  {
    name: 'pathway_logic_z3',
    category: 'Verifiable AI / neuro-symbolic',
    description: 'Tier-2 formal verification: decide pathway activation with the Z3 SMT solver and emit a satisfying model. UNSAT means not activated and cannot be overridden. Requires z3-solver.',
    handler: (i) => runPythonScript('server/neuro_symbolic.py', { ...i, task: 'z3_pathway' }),
    parameters: {
      foldChanges: { type: 'object', description: 'Map gene -> fold-change (states derived by threshold).' },
      geneStates: { type: 'object', description: "Alternative: map gene -> 'up'|'down'|'neutral'." },
      threshold: { type: 'number', description: 'Fold-change threshold (default 1.0).' },
      pathways: { type: 'array', description: 'Pathways [{id, name, rule}].', required: true },
    },
  },
  {
    name: 'pde_residual',
    category: 'Verifiable AI / physics',
    description: 'Physics-validity gate for spatiotemporal fields: compute the finite-difference reaction-diffusion PDE residual (du/dt - D u_xx - f(u)); reject as PHYSICALLY_INVALID if the max residual exceeds the threshold. numpy.',
    handler: (i) => runPythonScript('server/pde_validate.py', i),
    parameters: {
      u: { type: 'array', description: '2-D field [n_t, n_x].', required: true },
      D: { type: 'number', description: 'Diffusion coefficient.' },
      dx: { type: 'number', description: 'Spatial step.' },
      dt: { type: 'number', description: 'Time step.' },
      reaction: { type: 'object', description: "{type:'none'|'linear'|'logistic', rate, carryingCapacity}." },
      threshold: { type: 'number', description: 'Residual acceptance threshold (default 1e-4).' },
    },
  },
  {
    name: 'mml_select',
    category: 'Verifiable AI / model selection',
    description: 'Minimum Message Length (MML) model selection: choose the model minimizing model complexity + encoded residual, so extra parameters are only kept when they genuinely shorten the data encoding. numpy.',
    handler: (i) => runPythonScript('server/mml.py', i),
    parameters: {
      x: { type: 'array', description: 'Predictor values (polynomial-order mode).' },
      y: { type: 'array', description: 'Response values (polynomial-order mode).' },
      maxDegree: { type: 'number', description: 'Max polynomial degree to consider.' },
      candidates: { type: 'array', description: 'Generic mode: [{name, paramsCount, negLogLik, n}].' },
    },
  },
  {
    name: 'causal_discovery',
    category: 'Verifiable AI / causal inference',
    description: 'Infer a directed causal graph (not just correlation) from linear non-Gaussian data via DirectLiNGAM, with bootstrap-stability edge gating. Requires numpy; returns honest "unavailable" if absent.',
    handler: (i) => runPythonScript('server/causal_discovery.py', i),
    parameters: {
      data: { type: 'array', description: 'Matrix rows=samples, cols=variables.' },
      series: { type: 'object', description: 'Alternative: map variable -> values.' },
      variables: { type: 'array', description: 'Variable names (optional).' },
      nBootstrap: { type: 'number', description: 'Bootstrap resamples for stability (default 200).' },
      stabilityThreshold: { type: 'number', description: 'Keep edges seen in >= this fraction of bootstraps (default 0.9).' },
      seed: { type: 'number', description: 'Random seed (default 1337).' },
    },
  },
  {
    name: 'tensor_compress',
    category: 'Verifiable AI / compression',
    description: 'Tensor-Train (MPS) compression of a high-dimensional array with measured, reported truncation error and an honest "approximate" flag. A compression utility — not a cell simulator. Requires numpy+tensorly.',
    handler: (i) => runPythonScript('server/tensor_compression.py', i),
    parameters: {
      tensor: { type: 'array', description: 'Nested-list tensor (>=2 dims).', required: true },
      rank: { type: 'number', description: 'TT internal rank (optional; adaptively chosen if omitted).' },
      maxRelError: { type: 'number', description: 'Target relative reconstruction error (default 1e-4).' },
    },
  },
  // --- Real external-database grounding (live public APIs; honest errors) ---
  {
    name: 'db_ensembl_gene',
    category: 'External database (grounding)',
    description: 'Look up real gene coordinates, biotype and assembly from the Ensembl REST API by gene symbol.',
    handler: (i) => ensemblGeneBySymbol(i.symbol, i.species || 'homo_sapiens'),
    parameters: {
      symbol: { type: 'string', description: 'Gene symbol (e.g. TP53, BRCA2).', required: true },
      species: { type: 'string', description: "Ensembl species (default 'homo_sapiens')." },
    },
  },
  {
    name: 'db_gene_annotation',
    category: 'External database (grounding)',
    description: 'Fetch real gene annotation (Entrez ID, name, Ensembl gene, summary) from MyGene.info by symbol.',
    handler: (i) => myGeneBySymbol(i.symbol, i.species || 'human'),
    parameters: {
      symbol: { type: 'string', description: 'Gene symbol.', required: true },
      species: { type: 'string', description: "Species (default 'human')." },
    },
  },
  {
    name: 'db_protein_uniprot',
    category: 'External database (grounding)',
    description: 'Fetch the real canonical UniProt protein entry (accession, name, length) for a gene symbol + organism.',
    handler: (i) => uniProtByGene(i.symbol, i.organismId || 9606),
    parameters: {
      symbol: { type: 'string', description: 'Gene symbol.', required: true },
      organismId: { type: 'number', description: 'NCBI taxon id (default 9606 = human).' },
    },
  },
  {
    name: 'db_variant_vep',
    category: 'External database (grounding)',
    description: 'Real variant effect prediction (consequence, SIFT/PolyPhen) from the Ensembl VEP API by dbSNP rsID.',
    handler: (i) => vepByRsId(i.rsid, i.species || 'human'),
    parameters: {
      rsid: { type: 'string', description: 'dbSNP rsID (e.g. rs56116432).', required: true },
      species: { type: 'string', description: "Species (default 'human')." },
    },
  },
  // --- iDiscover: monumental frontier engines (all code-grounded, honest fallbacks) ---
  {
    name: 'cellular_reversion',
    category: 'iDiscover / Optimal Transport',
    description: 'iDiscover "Biological Git": compute the minimum-energy Optimal-Transport plan reverting a diseased single-cell distribution to a healthy reference. Returns the exact Wasserstein distance and the top per-gene perturbations (the "revert commits") from the transport coupling. Exact EMD via POT when available, else numpy Sinkhorn (flagged approximate); strict error if it fails to converge. Requires numpy.',
    handler: (i) => runPythonScript('server/optimal_transport.py', i, 180000),
    parameters: {
      sourceMatrix: { type: 'array', description: 'Diseased cells x genes (rows = cells).', required: true },
      targetMatrix: { type: 'array', description: 'Healthy reference cells x genes (same gene columns).', required: true },
      genes: { type: 'array', description: 'Gene names (length = number of columns).' },
      topK: { type: 'number', description: 'Number of perturbation commits to report (default 5).' },
      reg: { type: 'number', description: 'Sinkhorn entropic regularization (fallback only; default 0.05).' },
    },
  },
  {
    name: 'federated_zkp',
    category: 'iDiscover / privacy-preserving federation',
    description: 'iDiscover federated biomarker discovery: each site runs a REAL stratified log-rank survival test on its own private records; only additive (O-E, V) sufficient statistics leave the site — never raw rows. The aggregate is secured with REAL Pedersen commitments (homomorphic) + Schnorr/Fiat–Shamir zero-knowledge proofs of knowledge, so contributions are hidden and tamper-evident. Pure stdlib. Scope: not a general zk-SNARK over arbitrary predicates (no proving backend bundled) — stated honestly.',
    handler: (i) => runPythonScript('server/federated_zkp.py', i, 120000),
    parameters: {
      sites: { type: 'array', description: 'List of >=2 sites, each { name, durations[], events[0/1], groups[0/1] }.', required: true },
      alpha: { type: 'number', description: 'Significance threshold to report (default 0.01).' },
      scale: { type: 'number', description: 'Fixed-point scale for (O-E) integer commitments (default 1e6).' },
    },
  },
  {
    name: 'hyper_causal_discovery',
    category: 'iDiscover / hypergraph causal discovery',
    description: 'iDiscover Hyper-NOTEARS: discover a Directed Acyclic Hypergraph of multi-way (joint) causes from data (finds e.g. [A,B]->C that pairwise methods miss), OR verify a proposed weighted adjacency for causal loops. Acyclicity is enforced/certified by the exact tr(exp(W∘W))-d gate; a detected loop is rejected with a strict error — no heuristic DAG. Requires numpy + scipy.',
    handler: (i) => runPythonScript('server/hyper_causal.py', i, 180000),
    parameters: {
      data: { type: 'array', description: 'Discover mode: matrix rows=samples, cols=nodes/genes.' },
      series: { type: 'object', description: 'Discover mode alternative: map node -> values.' },
      adjacency: { type: 'array', description: 'Verify mode: square d×d weighted directed adjacency (W[i][j] = edge i->j).' },
      variables: { type: 'array', description: 'Node names (optional).' },
      maxOrder: { type: 'number', description: 'Max hyperedge tail size (default 2 = pairs).' },
      epsilon: { type: 'number', description: 'Acyclicity tolerance (default 1e-5).' },
      edgeThreshold: { type: 'number', description: '|strength| to report a discovered hyperedge (default 0.3).' },
    },
  },
  {
    name: 'gflownet_sample',
    category: 'iDiscover / generative chemistry',
    description: 'iDiscover GFlowNet: sample a diverse set of drug-like molecules proportionally to reward (Trajectory-Balance), not a single optimum. Tabular numpy GFlowNet; every returned molecule is RDKit-valid with a REAL computed QED reward — invalid samples are discarded, nothing fabricated. A deep neural GFlowNet (torch/GPU) is not claimed. Requires numpy + rdkit.',
    handler: (i) => runPythonScript('server/gflownet.py', i, 300000),
    parameters: {
      objective: { type: 'string', description: "Reward property to maximize (currently 'qed')." },
      maxLength: { type: 'number', description: 'Max fragments per molecule (default 4).' },
      beta: { type: 'number', description: 'Reward exponent R^beta to sharpen the target distribution (default 4).' },
      iterations: { type: 'number', description: 'Trajectory-Balance training steps (default 1500).' },
      nSamples: { type: 'number', description: 'Molecules to sample from the trained policy (default 200).' },
      topK: { type: 'number', description: 'Top candidates to return (default 10).' },
      seed: { type: 'number', description: 'Random seed (default 1337).' },
    },
  },
  // --- Module B depth: advanced expression analyses (real, CI-gated) ---
  {
    name: 'nb_differential_expression',
    category: 'Expression / differential analysis',
    description: 'Negative-binomial GLM differential expression (DESeq2-style count model with per-gene dispersion + Wald test + BH FDR) — more rigorous than a t-test. Requires numpy + statsmodels.',
    handler: (i) => runPythonScript('server/expression_advanced.py', { ...i, task: 'nb_de' }),
    parameters: {
      counts: { type: 'object', description: 'gene -> [integer counts per sample].', required: true },
      conditions: { type: 'array', description: 'Per-sample condition labels (exactly two groups).', required: true },
    },
  },
  {
    name: 'gsea',
    category: 'Expression / enrichment',
    description: 'Gene-set enrichment analysis (GSEA prerank) on a ranked gene list vs supplied gene sets → ES/NES/p/FDR per set. Requires gseapy.',
    handler: (i) => runPythonScript('server/expression_advanced.py', { ...i, task: 'gsea' }, 120000),
    parameters: {
      rnk: { type: 'object', description: 'gene -> ranking score.', required: true },
      geneSets: { type: 'object', description: 'set name -> [genes].', required: true },
      permutations: { type: 'number', description: 'Permutations (default 200).' },
    },
  },
  {
    name: 'batch_correct',
    category: 'Expression / preprocessing',
    description: 'Linear batch-effect removal (limma removeBatchEffect-style OLS): subtracts batch-indicator contributions while retaining biology. Requires numpy.',
    handler: (i) => runPythonScript('server/expression_advanced.py', { ...i, task: 'batch_correct' }),
    parameters: {
      matrix: { type: 'array', description: 'samples x features matrix.', required: true },
      batch: { type: 'array', description: 'per-sample batch labels.', required: true },
    },
  },
  {
    name: 'pca',
    category: 'Expression / dimensionality reduction',
    description: 'Principal component analysis with explained-variance ratios and sample scores. Requires numpy + scikit-learn.',
    handler: (i) => runPythonScript('server/expression_advanced.py', { ...i, task: 'pca' }),
    parameters: {
      matrix: { type: 'array', description: 'samples x features matrix.', required: true },
      nComponents: { type: 'number', description: 'Components to keep (default min(n,p,10)).' },
    },
  },
  // --- Core biostatistics (real scipy/statsmodels/sklearn tests) ---
  {
    name: 'fisher_exact', category: 'Biostatistics',
    description: "Fisher's exact test on a 2x2 contingency table (odds ratio + p). Requires scipy.",
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'fisher_exact' }),
    parameters: { table: { type: 'array', description: '2x2 contingency table.', required: true } },
  },
  {
    name: 'chi_square', category: 'Biostatistics',
    description: 'Chi-square test of independence on a contingency table. Requires scipy.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'chi_square' }),
    parameters: { table: { type: 'array', description: 'Contingency table (rows x cols).', required: true } },
  },
  {
    name: 'anova', category: 'Biostatistics',
    description: 'One-way ANOVA across >=2 numeric groups (F, p). Requires scipy.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'anova' }),
    parameters: { groups: { type: 'array', description: 'List of numeric arrays.', required: true } },
  },
  {
    name: 'correlation', category: 'Biostatistics',
    description: 'Pearson/Spearman/Kendall correlation for x,y, or a Pearson correlation matrix. Requires scipy.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'correlation' }),
    parameters: {
      x: { type: 'array', description: 'First vector (with y).' }, y: { type: 'array', description: 'Second vector.' },
      matrix: { type: 'array', description: 'Alternative: samples x features for a correlation matrix.' },
      method: { type: 'string', description: 'pearson|spearman|kendall (default pearson).' },
    },
  },
  {
    name: 'multiple_testing', category: 'Biostatistics',
    description: 'Multiple-testing correction (BH/Bonferroni/Holm/...) over p-values. Requires statsmodels.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'multiple_testing' }),
    parameters: { pvalues: { type: 'array', description: 'Raw p-values.', required: true }, method: { type: 'string', description: 'fdr_bh (default), bonferroni, holm, ...' } },
  },
  {
    name: 'power_ttest', category: 'Biostatistics',
    description: 'Two-sample t-test power analysis: give any two of effectSize/nobs/power, solves the third. Requires statsmodels.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'power_ttest' }),
    parameters: { effectSize: { type: 'number', description: "Cohen's d." }, nobs: { type: 'number', description: 'Per-group sample size.' }, power: { type: 'number', description: 'Target power.' } },
  },
  {
    name: 'normality_test', category: 'Biostatistics',
    description: 'Shapiro-Wilk + Kolmogorov-Smirnov normality tests. Requires scipy.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'normality' }),
    parameters: { x: { type: 'array', description: 'Numeric sample (>=3).', required: true } },
  },
  {
    name: 'roc_auc', category: 'Biostatistics',
    description: 'ROC curve + AUC for binary labels vs scores. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'roc_auc' }),
    parameters: { yTrue: { type: 'array', description: 'Binary labels 0/1.', required: true }, yScore: { type: 'array', description: 'Predicted scores.', required: true } },
  },
  {
    name: 'logrank_test', category: 'Biostatistics / survival',
    description: 'Two-group log-rank survival test (O-E, variance, chi-square, p, z). Requires scipy.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'logrank' }),
    parameters: { durations: { type: 'array', required: true, description: 'Follow-up times.' }, events: { type: 'array', required: true, description: 'Event 0/1.' }, groups: { type: 'array', required: true, description: 'Group 0/1.' } },
  },
  {
    name: 'cox_regression', category: 'Biostatistics / survival',
    description: 'Cox proportional-hazards regression (partial likelihood; per-covariate log HR, hazard ratio, p). Requires statsmodels.',
    handler: (i) => runPythonScript('server/biostats.py', { ...i, task: 'cox' }),
    parameters: {
      durations: { type: 'array', required: true, description: 'Follow-up times.' },
      events: { type: 'array', required: true, description: 'Event 0/1.' },
      covariates: { type: 'array', required: true, description: 'samples x k covariate matrix.' },
      covariateNames: { type: 'array', description: 'Optional covariate names.' },
    },
  },
  // --- Sequence & molecular biology (biopython) ---
  {
    name: 'translate_dna', category: 'Sequence / molecular biology',
    description: 'Translate a DNA/RNA sequence to protein (NCBI codon tables). Requires biopython.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'translate' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Nucleotide sequence.' }, codonTable: { type: 'number', description: 'NCBI table id (default 1).' }, toStop: { type: 'boolean', description: 'Stop at first stop codon.' } },
  },
  {
    name: 'reverse_complement', category: 'Sequence / molecular biology',
    description: 'Reverse complement (and complement) of a DNA sequence. Requires biopython.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'revcomp' }),
    parameters: { sequence: { type: 'string', required: true, description: 'DNA sequence.' } },
  },
  {
    name: 'gc_content', category: 'Sequence / molecular biology',
    description: 'GC content (%) and base composition of a sequence.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'gc_content' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Nucleotide sequence.' } },
  },
  {
    name: 'orf_find', category: 'Sequence / molecular biology',
    description: 'Find open reading frames (ATG→stop) on both strands / 3 frames, with translated protein. Requires biopython.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'orf_find' }),
    parameters: { sequence: { type: 'string', required: true, description: 'DNA sequence.' }, minAminoAcids: { type: 'number', description: 'Minimum ORF length in aa (default 20).' } },
  },
  {
    name: 'primer_tm', category: 'Sequence / molecular biology',
    description: 'Primer melting temperature (nearest-neighbor or Wallace) + GC%. Requires biopython.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'primer_tm' }),
    parameters: { primer: { type: 'string', required: true, description: 'Primer sequence.' }, method: { type: 'string', description: 'nn (default) or wallace.' } },
  },
  {
    name: 'restriction_map', category: 'Sequence / molecular biology',
    description: 'Restriction-enzyme cut sites over commercial enzymes (or a supplied subset). Requires biopython.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'restriction_map' }),
    parameters: { sequence: { type: 'string', required: true, description: 'DNA sequence.' }, enzymes: { type: 'array', description: 'Optional enzyme-name filter.' } },
  },
  {
    name: 'protein_params', category: 'Sequence / protein',
    description: 'Protein parameters (MW, pI, GRAVY hydrophobicity, aromaticity, instability index, secondary-structure fractions). Requires biopython.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'protein_params' }),
    parameters: { protein: { type: 'string', required: true, description: 'Amino-acid sequence.' } },
  },
  {
    name: 'codon_usage', category: 'Sequence / molecular biology',
    description: 'Codon counts and frequencies for a coding sequence.',
    handler: (i) => runPythonScript('server/seqtools.py', { ...i, task: 'codon_usage' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Coding DNA sequence.' } },
  },
  // --- Network biology (networkx) ---
  {
    name: 'network_centrality', category: 'Network biology',
    description: 'Degree/betweenness/closeness/eigenvector/PageRank centrality per node + top hub. Requires networkx.',
    handler: (i) => runPythonScript('server/netbio.py', { ...i, task: 'centrality' }),
    parameters: { edges: { type: 'array', required: true, description: '[[u,v] or [u,v,weight], ...].' }, directed: { type: 'boolean', description: 'Directed graph (default false).' } },
  },
  {
    name: 'community_detection', category: 'Network biology',
    description: 'Greedy-modularity community detection with modularity score. Requires networkx.',
    handler: (i) => runPythonScript('server/netbio.py', { ...i, task: 'community' }),
    parameters: { edges: { type: 'array', required: true, description: '[[u,v] or [u,v,weight], ...].' } },
  },
  {
    name: 'shortest_path', category: 'Network biology',
    description: 'Shortest (optionally weighted) path between two nodes. Requires networkx.',
    handler: (i) => runPythonScript('server/netbio.py', { ...i, task: 'shortest_path' }),
    parameters: { edges: { type: 'array', required: true, description: 'Edge list.' }, source: { type: 'string', required: true, description: 'Source node.' }, target: { type: 'string', required: true, description: 'Target node.' }, weighted: { type: 'boolean', description: 'Use edge weights (default true).' } },
  },
  {
    name: 'graph_stats', category: 'Network biology',
    description: 'Graph statistics: nodes/edges/density/avg clustering/components/diameter. Requires networkx.',
    handler: (i) => runPythonScript('server/netbio.py', { ...i, task: 'graph_stats' }),
    parameters: { edges: { type: 'array', required: true, description: 'Edge list.' }, directed: { type: 'boolean', description: 'Directed graph.' } },
  },
  {
    name: 'random_walk_restart', category: 'Network biology',
    description: 'Random walk with restart (personalized PageRank) from seed nodes — network propagation / gene prioritization. Requires networkx.',
    handler: (i) => runPythonScript('server/netbio.py', { ...i, task: 'rwr' }),
    parameters: { edges: { type: 'array', required: true, description: 'Edge list.' }, seeds: { type: 'array', required: true, description: 'Seed node ids.' }, restart: { type: 'number', description: 'Restart probability (default 0.15).' } },
  },
  // --- Advanced cheminformatics (RDKit) ---
  {
    name: 'tanimoto_similarity', category: 'Cheminformatics',
    description: 'Morgan-fingerprint Tanimoto similarity between two molecules (SMILES). Requires rdkit.',
    handler: (i) => runPythonScript('server/cheminfo_advanced.py', { ...i, task: 'tanimoto' }),
    parameters: { smiles1: { type: 'string', required: true, description: 'First SMILES.' }, smiles2: { type: 'string', required: true, description: 'Second SMILES.' } },
  },
  {
    name: 'similarity_matrix', category: 'Cheminformatics',
    description: 'Pairwise Tanimoto similarity matrix over a list of SMILES. Requires rdkit.',
    handler: (i) => runPythonScript('server/cheminfo_advanced.py', { ...i, task: 'similarity_matrix' }),
    parameters: { smiles: { type: 'array', required: true, description: 'List of SMILES (>=2).' } },
  },
  {
    name: 'substructure_search', category: 'Cheminformatics',
    description: 'Screen molecules for a SMARTS/SMILES substructure query. Requires rdkit.',
    handler: (i) => runPythonScript('server/cheminfo_advanced.py', { ...i, task: 'substructure_search' }),
    parameters: { query: { type: 'string', required: true, description: 'SMARTS or SMILES pattern.' }, smiles: { type: 'array', required: true, description: 'Molecules to screen.' } },
  },
  {
    name: 'murcko_scaffold', category: 'Cheminformatics',
    description: 'Bemis-Murcko scaffold (and generic scaffold) of a molecule. Requires rdkit.',
    handler: (i) => runPythonScript('server/cheminfo_advanced.py', { ...i, task: 'murcko_scaffold' }),
    parameters: { smiles: { type: 'string', required: true, description: 'Molecule SMILES.' } },
  },
  {
    name: 'pains_filter', category: 'Cheminformatics',
    description: 'Flag PAINS (pan-assay interference) substructures in molecules — screening triage. Requires rdkit.',
    handler: (i) => runPythonScript('server/cheminfo_advanced.py', { ...i, task: 'pains_filter' }),
    parameters: { smiles: { type: 'array', required: true, description: 'SMILES (string or list).' } },
  },
  // --- Machine learning (scikit-learn) ---
  {
    name: 'kmeans_cluster', category: 'Machine learning',
    description: 'K-means clustering with silhouette score + cluster centers. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/ml_analysis.py', { ...i, task: 'kmeans' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, k: { type: 'number', description: 'Clusters (default 3).' } },
  },
  {
    name: 'hierarchical_cluster', category: 'Machine learning',
    description: 'Agglomerative (hierarchical) clustering + silhouette. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/ml_analysis.py', { ...i, task: 'hierarchical' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, k: { type: 'number', description: 'Clusters (default 3).' }, linkage: { type: 'string', description: 'ward|complete|average|single.' } },
  },
  {
    name: 'tsne_embed', category: 'Machine learning',
    description: 't-SNE 2-D embedding for visualization. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/ml_analysis.py', { ...i, task: 'tsne' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, perplexity: { type: 'number', description: 'Perplexity (auto by default).' } },
  },
  {
    name: 'rf_feature_importance', category: 'Machine learning',
    description: 'Random-Forest feature importances (classification or regression). Requires scikit-learn.',
    handler: (i) => runPythonScript('server/ml_analysis.py', { ...i, task: 'rf_importance' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Target.' }, featureNames: { type: 'array', description: 'Feature names.' }, classification: { type: 'boolean', description: 'Classification (default true).' } },
  },
  {
    name: 'lasso_feature_select', category: 'Machine learning',
    description: 'LASSO (L1) feature selection via cross-validated LassoCV — sparse predictors. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/ml_analysis.py', { ...i, task: 'lasso_select' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Continuous target.' }, featureNames: { type: 'array', description: 'Feature names.' } },
  },
  {
    name: 'logistic_classifier', category: 'Machine learning',
    description: 'Cross-validated logistic-regression classifier (accuracy + AUC). Requires scikit-learn.',
    handler: (i) => runPythonScript('server/ml_analysis.py', { ...i, task: 'logistic' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Binary target 0/1.' } },
  },
  // --- Variant / population genetics ---
  {
    name: 'hardy_weinberg', category: 'Population genetics',
    description: 'Hardy-Weinberg equilibrium test from genotype counts (AA, Aa, aa): chi-square, p, allele freqs. Requires scipy.',
    handler: (i) => runPythonScript('server/variant_tools.py', { ...i, task: 'hardy_weinberg' }),
    parameters: { AA: { type: 'number', required: true, description: 'Homozygous-major count.' }, Aa: { type: 'number', required: true, description: 'Heterozygous count.' }, aa: { type: 'number', required: true, description: 'Homozygous-minor count.' } },
  },
  {
    name: 'allele_frequency', category: 'Population genetics',
    description: 'Allele + genotype frequencies from genotype counts.',
    handler: (i) => runPythonScript('server/variant_tools.py', { ...i, task: 'allele_frequency' }),
    parameters: { AA: { type: 'number', required: true, description: 'AA count.' }, Aa: { type: 'number', required: true, description: 'Aa count.' }, aa: { type: 'number', required: true, description: 'aa count.' } },
  },
  {
    name: 'ts_tv_ratio', category: 'Genomics / variants',
    description: 'Transition/transversion ratio from a list of SNVs ({ref, alt}).',
    handler: (i) => runPythonScript('server/variant_tools.py', { ...i, task: 'ts_tv' }),
    parameters: { variants: { type: 'array', required: true, description: '[{ref, alt}, ...].' } },
  },
  {
    name: 'vcf_summary', category: 'Genomics / variants',
    description: 'Summarize variants: SNV/indel counts, Ts/Tv, per-chromosome counts.',
    handler: (i) => runPythonScript('server/variant_tools.py', { ...i, task: 'vcf_summary' }),
    parameters: { variants: { type: 'array', required: true, description: '[{chrom, ref, alt}, ...].' } },
  },
  // --- Advanced microbiome ---
  {
    name: 'chao1_richness', category: 'Microbiome',
    description: 'Chao1 bias-corrected richness estimator per sample. Requires numpy.',
    handler: (i) => runPythonScript('server/microbiome_advanced.py', { ...i, task: 'chao1' }),
    parameters: { counts: { type: 'object', required: true, description: 'sample -> {taxon: count}.' } },
  },
  {
    name: 'differential_abundance', category: 'Microbiome',
    description: 'Compositional differential abundance: CLR transform + Welch t-test per taxon + BH FDR (ALDEx2-style). Requires numpy/scipy/statsmodels.',
    handler: (i) => runPythonScript('server/microbiome_advanced.py', { ...i, task: 'differential_abundance' }),
    parameters: { counts: { type: 'object', required: true, description: 'sample -> {taxon: count}.' }, groups: { type: 'object', required: true, description: 'sample -> group label (two groups).' } },
  },
  {
    name: 'rarefaction_curve', category: 'Microbiome',
    description: 'Rarefaction curve (Hurlbert analytic expected richness vs depth) per sample. Requires numpy.',
    handler: (i) => runPythonScript('server/microbiome_advanced.py', { ...i, task: 'rarefaction' }),
    parameters: { counts: { type: 'object', required: true, description: 'sample -> {taxon: count}.' }, steps: { type: 'number', description: 'Depth steps (default 10).' } },
  },
  // --- Protein structure (biopython PDB) ---
  {
    name: 'structure_summary', category: 'Structural biology',
    description: 'PDB structure summary: chains, residue/atom counts. Requires biopython.',
    handler: (i) => runPythonScript('server/structure_tools.py', { ...i, task: 'structure_summary' }),
    parameters: { pdb: { type: 'string', required: true, description: 'PDB-format text.' } },
  },
  {
    name: 'radius_of_gyration', category: 'Structural biology',
    description: 'Radius of gyration (CA) — protein compactness — + center of mass. Requires biopython.',
    handler: (i) => runPythonScript('server/structure_tools.py', { ...i, task: 'radius_of_gyration' }),
    parameters: { pdb: { type: 'string', required: true, description: 'PDB-format text.' } },
  },
  {
    name: 'contact_map', category: 'Structural biology',
    description: 'CA-CA residue contact map under a distance threshold (+ contact order). Requires biopython.',
    handler: (i) => runPythonScript('server/structure_tools.py', { ...i, task: 'contact_map' }),
    parameters: { pdb: { type: 'string', required: true, description: 'PDB text.' }, threshold: { type: 'number', description: 'Contact distance Å (default 8).' }, minSeqSep: { type: 'number', description: 'Min sequence separation (default 3).' } },
  },
  {
    name: 'atom_distance', category: 'Structural biology',
    description: 'Distance between two atoms (chain/resid/atom) in a structure. Requires biopython.',
    handler: (i) => runPythonScript('server/structure_tools.py', { ...i, task: 'distance' }),
    parameters: { pdb: { type: 'string', required: true, description: 'PDB text.' }, atomA: { type: 'object', required: true, description: '{chain, resid, atom?}.' }, atomB: { type: 'object', required: true, description: '{chain, resid, atom?}.' } },
  },
  // --- Pharmacology / dose-response ---
  {
    name: 'dose_response_ic50', category: 'Pharmacology',
    description: '4-parameter logistic (Hill) dose-response fit → IC50/EC50, Hill slope, top/bottom, R^2. Requires scipy.',
    handler: (i) => runPythonScript('server/doseresponse.py', { ...i, task: 'ic50' }),
    parameters: { doses: { type: 'array', required: true, description: 'Positive dose concentrations.' }, responses: { type: 'array', required: true, description: 'Measured responses.' } },
  },
  {
    name: 'curve_auc', category: 'Pharmacology',
    description: 'Trapezoidal area under a response curve. Requires scipy/numpy.',
    handler: (i) => runPythonScript('server/doseresponse.py', { ...i, task: 'auc' }),
    parameters: { x: { type: 'array', required: true, description: 'x values.' }, y: { type: 'array', required: true, description: 'y values.' } },
  },
  // --- Regression models (statsmodels) ---
  {
    name: 'ols_regression', category: 'Regression',
    description: 'Ordinary least squares linear regression (coefficients, p-values, R^2). Requires statsmodels.',
    handler: (i) => runPythonScript('server/regression_tools.py', { ...i, task: 'ols' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Continuous target.' }, featureNames: { type: 'array', description: 'Feature names.' } },
  },
  {
    name: 'logistic_glm', category: 'Regression',
    description: 'Logistic regression (GLM binomial): coefficients, odds ratios, AIC. Requires statsmodels.',
    handler: (i) => runPythonScript('server/regression_tools.py', { ...i, task: 'logistic_glm' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Binary 0/1 target.' }, featureNames: { type: 'array', description: 'Feature names.' } },
  },
  {
    name: 'poisson_glm', category: 'Regression',
    description: 'Poisson regression (GLM) for count outcomes: coefficients, rate ratios. Requires statsmodels.',
    handler: (i) => runPythonScript('server/regression_tools.py', { ...i, task: 'poisson_glm' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Count target.' }, featureNames: { type: 'array', description: 'Feature names.' } },
  },
  {
    name: 'mixed_effects_model', category: 'Regression',
    description: 'Linear mixed-effects model (random intercept per group). Requires statsmodels+pandas.',
    handler: (i) => runPythonScript('server/regression_tools.py', { ...i, task: 'mixedlm' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Continuous target.' }, groups: { type: 'array', required: true, description: 'Grouping labels.' }, featureNames: { type: 'array', description: 'Feature names.' } },
  },
  {
    name: 'robust_regression', category: 'Regression',
    description: 'Robust regression (Huber M-estimator) resistant to outliers. Requires statsmodels.',
    handler: (i) => runPythonScript('server/regression_tools.py', { ...i, task: 'robust_regression' }),
    parameters: { X: { type: 'array', required: true, description: 'samples x features.' }, y: { type: 'array', required: true, description: 'Continuous target.' }, featureNames: { type: 'array', description: 'Feature names.' } },
  },
  // --- Dimensionality reduction (scikit-learn) ---
  {
    name: 'mds_embed', category: 'Dimensionality reduction',
    description: 'Multidimensional scaling (MDS) embedding. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/dimreduction_tools.py', { ...i, task: 'mds' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples x features.' }, nComponents: { type: 'number', description: 'Components (default 2).' } },
  },
  {
    name: 'ica_decompose', category: 'Dimensionality reduction',
    description: 'Independent Component Analysis (FastICA) — separate independent sources. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/dimreduction_tools.py', { ...i, task: 'ica' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples x features.' }, nComponents: { type: 'number', description: 'Components.' } },
  },
  {
    name: 'nmf_decompose', category: 'Dimensionality reduction',
    description: 'Non-negative matrix factorization (NMF) for non-negative data (metagenes/signatures). Requires scikit-learn.',
    handler: (i) => runPythonScript('server/dimreduction_tools.py', { ...i, task: 'nmf' }),
    parameters: { matrix: { type: 'array', required: true, description: 'Non-negative samples x features.' }, nComponents: { type: 'number', description: 'Components.' } },
  },
  {
    name: 'factor_analysis', category: 'Dimensionality reduction',
    description: 'Factor analysis latent-factor extraction. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/dimreduction_tools.py', { ...i, task: 'factor_analysis' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples x features.' }, nComponents: { type: 'number', description: 'Factors.' } },
  },
  {
    name: 'kernel_pca', category: 'Dimensionality reduction',
    description: 'Kernel PCA (nonlinear) embedding. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/dimreduction_tools.py', { ...i, task: 'kernel_pca' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples x features.' }, kernel: { type: 'string', description: 'rbf|poly|sigmoid|cosine (default rbf).' }, nComponents: { type: 'number', description: 'Components.' } },
  },
  // --- Population genetics ---
  {
    name: 'nucleotide_diversity', category: 'Population genetics',
    description: 'Nucleotide diversity (pi) from a 0/1 haplotype matrix. Requires numpy.',
    handler: (i) => runPythonScript('server/population_genetics.py', { ...i, task: 'nucleotide_diversity' }),
    parameters: { haplotypes: { type: 'array', required: true, description: '0/1 matrix (samples x sites).' } },
  },
  {
    name: 'tajimas_d', category: 'Population genetics',
    description: "Tajima's D neutrality test statistic from a haplotype matrix. Requires numpy.",
    handler: (i) => runPythonScript('server/population_genetics.py', { ...i, task: 'tajimas_d' }),
    parameters: { haplotypes: { type: 'array', required: true, description: '0/1 matrix (>=4 samples).' } },
  },
  {
    name: 'fst', category: 'Population genetics',
    description: 'Fixation index (Fst) between two populations from 0/1 matrices. Requires numpy.',
    handler: (i) => runPythonScript('server/population_genetics.py', { ...i, task: 'fst' }),
    parameters: { pop1: { type: 'array', required: true, description: '0/1 matrix pop1.' }, pop2: { type: 'array', required: true, description: '0/1 matrix pop2 (same sites).' } },
  },
  {
    name: 'linkage_disequilibrium', category: 'Population genetics',
    description: 'LD between two loci (D and r^2). Requires numpy.',
    handler: (i) => runPythonScript('server/population_genetics.py', { ...i, task: 'ld_r2' }),
    parameters: { locusA: { type: 'array', required: true, description: '0/1 per individual.' }, locusB: { type: 'array', required: true, description: '0/1 per individual.' } },
  },
  {
    name: 'maf_spectrum', category: 'Population genetics',
    description: 'Minor-allele-frequency spectrum (histogram) from a haplotype matrix. Requires numpy.',
    handler: (i) => runPythonScript('server/population_genetics.py', { ...i, task: 'maf_spectrum' }),
    parameters: { haplotypes: { type: 'array', required: true, description: '0/1 matrix (samples x sites).' } },
  },
  // --- Enrichment (over-representation) ---
  {
    name: 'over_representation', category: 'Enrichment',
    description: 'Hypergeometric over-representation analysis of a gene list vs gene sets (fold enrichment + BH FDR). Requires scipy/statsmodels.',
    handler: (i) => runPythonScript('server/enrichment_tools.py', { ...i, task: 'ora' }),
    parameters: { query: { type: 'array', required: true, description: 'Query gene list.' }, geneSets: { type: 'object', required: true, description: 'name -> [genes].' }, universe: { type: 'array', description: 'Background gene universe (optional).' } },
  },
  {
    name: 'geneset_overlap', category: 'Enrichment',
    description: 'Overlap between two gene sets (Jaccard + overlap coefficient + shared genes).',
    handler: (i) => runPythonScript('server/enrichment_tools.py', { ...i, task: 'geneset_overlap' }),
    parameters: { setA: { type: 'array', required: true, description: 'Gene set A.' }, setB: { type: 'array', required: true, description: 'Gene set B.' } },
  },
  // --- Quality control ---
  {
    name: 'fastq_quality', category: 'Quality control',
    description: 'FASTQ per-base Phred quality profile, mean/min/max Q, %Q30, GC%. Requires numpy.',
    handler: (i) => runPythonScript('server/qc_tools.py', { ...i, task: 'fastq_quality' }),
    parameters: { fastq: { type: 'string', required: true, description: 'FASTQ text (4-line records).' } },
  },
  {
    name: 'count_matrix_qc', category: 'Quality control',
    description: 'scRNA/bulk count-matrix QC: library sizes, genes detected, mito %. Requires numpy.',
    handler: (i) => runPythonScript('server/qc_tools.py', { ...i, task: 'count_matrix_qc' }),
    parameters: { counts: { type: 'array', required: true, description: 'cells x genes matrix.' }, genes: { type: 'array', description: 'Gene names (for mito %).' } },
  },
  {
    name: 'outlier_mad', category: 'Quality control',
    description: 'Median-absolute-deviation (modified z-score) outlier detection. Requires numpy.',
    handler: (i) => runPythonScript('server/qc_tools.py', { ...i, task: 'outlier_mad' }),
    parameters: { x: { type: 'array', required: true, description: 'Numeric values.' }, threshold: { type: 'number', description: 'Modified-z threshold (default 3.5).' } },
  },
  // --- Proteomics utilities ---
  {
    name: 'peptide_mass', category: 'Proteomics',
    description: 'Peptide monoisotopic mass + singly/doubly-charged m/z.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'peptide_mass' }),
    parameters: { peptide: { type: 'string', required: true, description: 'Amino-acid sequence.' } },
  },
  {
    name: 'tryptic_digest', category: 'Proteomics',
    description: 'In-silico trypsin digestion (cut after K/R not before P) with peptide masses.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'tryptic_digest' }),
    parameters: { protein: { type: 'string', required: true, description: 'Protein sequence.' }, missedCleavages: { type: 'number', description: 'Allowed missed cleavages (default 0).' } },
  },
  {
    name: 'fragment_ions', category: 'Proteomics',
    description: 'b/y fragment-ion ladder (singly charged) for a peptide — MS/MS annotation.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'fragment_ions' }),
    parameters: { peptide: { type: 'string', required: true, description: 'Peptide sequence (>=2 aa).' } },
  },
  // --- Phylogenetics (biopython) ---
  {
    name: 'phylo_distance_matrix', category: 'Phylogenetics',
    description: 'Identity distance matrix from aligned sequences. Requires biopython.',
    handler: (i) => runPythonScript('server/phylo_tools.py', { ...i, task: 'distance_matrix' }),
    parameters: { sequences: { type: 'object', required: true, description: '{name: aligned_seq} (>=3, equal length).' } },
  },
  {
    name: 'nj_tree', category: 'Phylogenetics',
    description: 'Neighbor-joining phylogenetic tree (Newick). Requires biopython.',
    handler: (i) => runPythonScript('server/phylo_tools.py', { ...i, task: 'nj_tree' }),
    parameters: { sequences: { type: 'object', required: true, description: '{name: aligned_seq}.' } },
  },
  {
    name: 'upgma_tree', category: 'Phylogenetics',
    description: 'UPGMA phylogenetic tree (Newick). Requires biopython.',
    handler: (i) => runPythonScript('server/phylo_tools.py', { ...i, task: 'upgma_tree' }),
    parameters: { sequences: { type: 'object', required: true, description: '{name: aligned_seq}.' } },
  },
  {
    name: 'patristic_distances', category: 'Phylogenetics',
    description: 'Patristic (tree path) distances between all taxa from the NJ tree. Requires biopython.',
    handler: (i) => runPythonScript('server/phylo_tools.py', { ...i, task: 'patristic' }),
    parameters: { sequences: { type: 'object', required: true, description: '{name: aligned_seq}.' } },
  },
  // --- Pairwise alignment (biopython) ---
  {
    name: 'global_align', category: 'Alignment',
    description: 'Needleman-Wunsch global alignment (score + % identity). Requires biopython.',
    handler: (i) => runPythonScript('server/align_tools.py', { ...i, task: 'global_align' }),
    parameters: { seq1: { type: 'string', required: true, description: 'First sequence.' }, seq2: { type: 'string', required: true, description: 'Second sequence.' } },
  },
  {
    name: 'local_align', category: 'Alignment',
    description: 'Smith-Waterman local alignment (score + % identity). Requires biopython.',
    handler: (i) => runPythonScript('server/align_tools.py', { ...i, task: 'local_align' }),
    parameters: { seq1: { type: 'string', required: true, description: 'First sequence.' }, seq2: { type: 'string', required: true, description: 'Second sequence.' } },
  },
  {
    name: 'percent_identity', category: 'Alignment',
    description: 'Percent identity from a global alignment. Requires biopython.',
    handler: (i) => runPythonScript('server/align_tools.py', { ...i, task: 'percent_identity' }),
    parameters: { seq1: { type: 'string', required: true, description: 'First sequence.' }, seq2: { type: 'string', required: true, description: 'Second sequence.' } },
  },
  {
    name: 'kmer_distance', category: 'Alignment',
    description: 'Alignment-free k-mer Jaccard distance between two sequences.',
    handler: (i) => runPythonScript('server/align_tools.py', { ...i, task: 'kmer_distance' }),
    parameters: { seq1: { type: 'string', required: true, description: 'First sequence.' }, seq2: { type: 'string', required: true, description: 'Second sequence.' }, k: { type: 'number', description: 'k-mer size (default 3).' } },
  },
  // --- Epigenomics ---
  {
    name: 'interval_jaccard', category: 'Epigenomics',
    description: 'Base-pair Jaccard overlap between two genomic interval (peak) sets.',
    handler: (i) => runPythonScript('server/epigenomics.py', { ...i, task: 'interval_jaccard' }),
    parameters: { setA: { type: 'array', required: true, description: '[[start,end],...].' }, setB: { type: 'array', required: true, description: '[[start,end],...].' } },
  },
  {
    name: 'pwm_scan', category: 'Epigenomics',
    description: 'Scan a sequence for PWM (motif) log-odds hits above threshold. Requires numpy.',
    handler: (i) => runPythonScript('server/epigenomics.py', { ...i, task: 'pwm_scan' }),
    parameters: { pwm: { type: 'object', required: true, description: '{A,C,G,T: [per-position probs]}.' }, sequence: { type: 'string', required: true, description: 'Sequence to scan.' }, threshold: { type: 'number', description: 'Log-odds threshold (default 0).' } },
  },
  {
    name: 'methylation_mvalues', category: 'Epigenomics',
    description: 'Transform methylation beta values to M-values (logit2).',
    handler: (i) => runPythonScript('server/epigenomics.py', { ...i, task: 'methylation_mvalues' }),
    parameters: { betas: { type: 'array', required: true, description: 'Beta values (0..1).' } },
  },
  {
    name: 'dmr_test', category: 'Epigenomics',
    description: 'Differentially methylated sites: per-site Welch t-test + BH FDR. Requires numpy/scipy/statsmodels.',
    handler: (i) => runPythonScript('server/epigenomics.py', { ...i, task: 'dmr_test' }),
    parameters: { group1: { type: 'array', required: true, description: 'samples x sites beta matrix.' }, group2: { type: 'array', required: true, description: 'samples x sites beta matrix.' } },
  },
  // --- Immunoinformatics ---
  {
    name: 'repertoire_diversity', category: 'Immunoinformatics',
    description: 'Immune-repertoire diversity: Shannon, Simpson, clonality, richness.',
    handler: (i) => runPythonScript('server/immunoinformatics.py', { ...i, task: 'repertoire_diversity' }),
    parameters: { clones: { type: 'object', required: true, description: '{clonotype: count}.' } },
  },
  {
    name: 'vj_usage', category: 'Immunoinformatics',
    description: 'V/J gene usage frequencies and top V-J pairings.',
    handler: (i) => runPythonScript('server/immunoinformatics.py', { ...i, task: 'vj_usage' }),
    parameters: { rearrangements: { type: 'array', required: true, description: '[{v, j}, ...].' } },
  },
  {
    name: 'cdr3_spectratype', category: 'Immunoinformatics',
    description: 'CDR3 length distribution (spectratype).',
    handler: (i) => runPythonScript('server/immunoinformatics.py', { ...i, task: 'cdr3_spectratype' }),
    parameters: { cdr3: { type: 'array', required: true, description: 'List of CDR3 sequences.' } },
  },
  {
    name: 'repertoire_overlap', category: 'Immunoinformatics',
    description: 'Overlap between two repertoires: Jaccard + Morisita-Horn index.',
    handler: (i) => runPythonScript('server/immunoinformatics.py', { ...i, task: 'repertoire_overlap' }),
    parameters: { repertoireA: { type: 'object', required: true, description: '{clonotype: count}.' }, repertoireB: { type: 'object', required: true, description: '{clonotype: count}.' } },
  },
  // --- Time series / signal analysis (numpy/scipy/statsmodels) ---
  {
    name: 'autocorrelation', category: 'Time series / signal',
    description: 'Autocorrelation function (ACF) of a series up to maxLag. Requires numpy.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'autocorrelation' }),
    parameters: { x: { type: 'array', required: true, description: 'Numeric series (>=3 values).' }, maxLag: { type: 'number', description: 'Maximum lag (default min(n-1,20)).' } },
  },
  {
    name: 'cross_correlation', category: 'Time series / signal',
    description: 'Normalized cross-correlation between two equal-length series; reports best lag/correlation. Requires numpy.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'cross_correlation' }),
    parameters: { x: { type: 'array', required: true, description: 'First series.' }, y: { type: 'array', required: true, description: 'Second series (same length).' }, maxLag: { type: 'number', description: 'Maximum +/- lag.' } },
  },
  {
    name: 'changepoint_cusum', category: 'Time series / signal',
    description: 'CUSUM change-point detection with bootstrap significance. Requires numpy.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'changepoint_cusum' }),
    parameters: { x: { type: 'array', required: true, description: 'Numeric series.' }, nBootstrap: { type: 'number', description: 'Bootstrap iterations (default 500).' }, seed: { type: 'number', description: 'RNG seed.' } },
  },
  {
    name: 'periodicity_fft', category: 'Time series / signal',
    description: 'Dominant frequency/period via FFT power spectrum. Requires numpy.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'periodicity_fft' }),
    parameters: { x: { type: 'array', required: true, description: 'Numeric series.' }, dt: { type: 'number', description: 'Sample spacing (default 1.0).' } },
  },
  {
    name: 'lowess_trend', category: 'Time series / signal',
    description: 'LOWESS locally-weighted smoothing trend. Requires statsmodels.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'lowess_trend' }),
    parameters: { y: { type: 'array', required: true, description: 'Response values.' }, x: { type: 'array', description: 'Optional x (defaults to index).' }, frac: { type: 'number', description: 'Smoothing span fraction (default 0.3).' } },
  },
  {
    name: 'linear_detrend', category: 'Time series / signal',
    description: 'Remove a least-squares linear trend; returns slope, intercept, residuals. Requires numpy.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'linear_detrend' }),
    parameters: { y: { type: 'array', required: true, description: 'Response values.' }, x: { type: 'array', description: 'Optional x (defaults to index).' } },
  },
  {
    name: 'moving_average', category: 'Time series / signal',
    description: 'Simple moving average (valid convolution) over a window. Requires numpy.',
    handler: (i) => runPythonScript('server/timeseries_tools.py', { ...i, task: 'moving_average' }),
    parameters: { y: { type: 'array', required: true, description: 'Numeric series.' }, window: { type: 'number', description: 'Window size (default 3).' } },
  },
  // --- Clinical epidemiology / diagnostics (numpy/scipy) ---
  {
    name: 'odds_ratio_rr', category: 'Clinical epidemiology',
    description: 'Odds ratio, relative risk, ARR, NNT with 95% CIs from a 2x2 table (Haldane-corrected for zeros).',
    handler: (i) => runPythonScript('server/clinical_tools.py', { ...i, task: 'odds_ratio_rr' }),
    parameters: { table: { type: 'array', required: true, description: '2x2 [[exposed_event, exposed_noevent],[unexposed_event, unexposed_noevent]].' } },
  },
  {
    name: 'diagnostic_metrics', category: 'Clinical epidemiology',
    description: 'Sensitivity/specificity/PPV/NPV/accuracy/F1/Youden J from tp,fp,fn,tn counts.',
    handler: (i) => runPythonScript('server/clinical_tools.py', { ...i, task: 'diagnostic_metrics' }),
    parameters: { tp: { type: 'number', required: true, description: 'True positives.' }, fp: { type: 'number', required: true, description: 'False positives.' }, fn: { type: 'number', required: true, description: 'False negatives.' }, tn: { type: 'number', required: true, description: 'True negatives.' } },
  },
  {
    name: 'number_needed_to_treat', category: 'Clinical epidemiology',
    description: 'Number needed to treat from control vs treated event rates (ARR -> NNT).',
    handler: (i) => runPythonScript('server/clinical_tools.py', { ...i, task: 'number_needed_to_treat' }),
    parameters: { controlEventRate: { type: 'number', required: true, description: 'Control event rate 0..1.' }, treatedEventRate: { type: 'number', required: true, description: 'Treated event rate 0..1.' } },
  },
  {
    name: 'meta_analysis', category: 'Clinical epidemiology',
    description: 'Inverse-variance fixed-effect meta-analysis: pooled effect, 95% CI, Cochran Q, I^2. Requires numpy.',
    handler: (i) => runPythonScript('server/clinical_tools.py', { ...i, task: 'meta_analysis' }),
    parameters: { studies: { type: 'array', required: true, description: '[{effect, se}, ...] (>=2 studies).' } },
  },
  // --- WGCNA co-expression network analysis (numpy/scipy/sklearn) ---
  {
    name: 'wgcna_soft_threshold', category: 'Systems biology / WGCNA',
    description: 'Pick a soft-threshold power by scale-free topology fit (R^2 vs connectivity). Requires numpy.',
    handler: (i) => runPythonScript('server/wgcna.py', { ...i, task: 'soft_threshold' }),
    parameters: { expression: { type: 'array', required: true, description: 'samples x genes matrix.' }, powers: { type: 'array', description: 'Candidate powers (default [1,2,4,6,8,10,12]).' } },
  },
  {
    name: 'wgcna_coexpression_modules', category: 'Systems biology / WGCNA',
    description: 'Detect co-expression modules via |corr|^beta adjacency + hierarchical clustering. Requires scipy.',
    handler: (i) => runPythonScript('server/wgcna.py', { ...i, task: 'coexpression_modules' }),
    parameters: { expression: { type: 'array', required: true, description: 'samples x genes matrix.' }, power: { type: 'number', description: 'Soft-threshold power (default 6).' }, nModules: { type: 'number', description: 'Target module count (default 3).' }, geneNames: { type: 'array', description: 'Gene names aligned to columns.' } },
  },
  {
    name: 'wgcna_module_eigengenes', category: 'Systems biology / WGCNA',
    description: 'Module eigengenes (PC1 per module) with variance explained. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/wgcna.py', { ...i, task: 'module_eigengenes' }),
    parameters: { expression: { type: 'array', required: true, description: 'samples x genes matrix.' }, moduleAssignments: { type: 'object', required: true, description: '{module: [geneNames]}.' }, geneNames: { type: 'array', description: 'Gene names aligned to columns.' } },
  },
  // --- Flow cytometry (numpy) ---
  {
    name: 'flow_arcsinh_transform', category: 'Flow cytometry',
    description: 'Arcsinh (biexponential-style) transform of event x channel intensities. Requires numpy.',
    handler: (i) => runPythonScript('server/flow_tools.py', { ...i, task: 'arcsinh_transform' }),
    parameters: { events: { type: 'array', required: true, description: 'events x channels matrix.' }, cofactor: { type: 'number', description: 'Arcsinh cofactor (default 150).' } },
  },
  {
    name: 'flow_compensation', category: 'Flow cytometry',
    description: 'Fluorescence compensation via spillover-matrix inverse. Requires numpy.',
    handler: (i) => runPythonScript('server/flow_tools.py', { ...i, task: 'compensation' }),
    parameters: { events: { type: 'array', required: true, description: 'events x channels matrix.' }, spillover: { type: 'array', required: true, description: 'channels x channels spillover matrix.' } },
  },
  {
    name: 'flow_gating_frequencies', category: 'Flow cytometry',
    description: 'AND-combined threshold gating -> population frequency (%). Requires numpy.',
    handler: (i) => runPythonScript('server/flow_tools.py', { ...i, task: 'gating_frequencies' }),
    parameters: { events: { type: 'array', required: true, description: 'events x channels matrix.' }, gates: { type: 'array', required: true, description: '[{channel, min?, max?}, ...] AND-combined.' }, channels: { type: 'array', description: 'Channel names aligned to columns.' } },
  },
  {
    name: 'flow_channel_summary', category: 'Flow cytometry',
    description: 'Per-channel median/mean/CV/p95 summary of flow events. Requires numpy.',
    handler: (i) => runPythonScript('server/flow_tools.py', { ...i, task: 'channel_summary' }),
    parameters: { events: { type: 'array', required: true, description: 'events x channels matrix.' }, channels: { type: 'array', description: 'Channel names aligned to columns.' } },
  },
  // --- Spatial statistics (numpy/scipy) ---
  {
    name: 'morans_i', category: 'Spatial statistics',
    description: "Moran's I global spatial autocorrelation (+ expected I, z-score, normal p-value). Requires numpy.",
    handler: (i) => runPythonScript('server/spatial_tools.py', { ...i, task: 'morans_i' }),
    parameters: { values: { type: 'array', required: true, description: 'Numeric observation per location.' }, weights: { type: 'array', required: true, description: 'n x n spatial weight matrix.' } },
  },
  {
    name: 'gearys_c', category: 'Spatial statistics',
    description: "Geary's C spatial autocorrelation (~0 strong positive, 1 none). Requires numpy.",
    handler: (i) => runPythonScript('server/spatial_tools.py', { ...i, task: 'gearys_c' }),
    parameters: { values: { type: 'array', required: true, description: 'Numeric observation per location.' }, weights: { type: 'array', required: true, description: 'n x n spatial weight matrix.' } },
  },
  {
    name: 'getis_ord_general_g', category: 'Spatial statistics',
    description: 'Getis-Ord General G hot/cold-spot statistic on non-negative values. Requires numpy.',
    handler: (i) => runPythonScript('server/spatial_tools.py', { ...i, task: 'getis_ord_general_g' }),
    parameters: { values: { type: 'array', required: true, description: 'Non-negative value per location.' }, weights: { type: 'array', required: true, description: 'n x n spatial weight matrix.' } },
  },
  {
    name: 'ripleys_k', category: 'Spatial statistics',
    description: "Ripley's K / Besag L point-pattern clustering across radii. Requires numpy.",
    handler: (i) => runPythonScript('server/spatial_tools.py', { ...i, task: 'ripleys_k' }),
    parameters: { points: { type: 'array', required: true, description: 'List of [x,y] coordinates.' }, radii: { type: 'array', required: true, description: 'Radii to evaluate.' }, area: { type: 'number', required: true, description: 'Study-region area.' } },
  },
  {
    name: 'moran_permutation_test', category: 'Spatial statistics',
    description: "Permutation p-value for Moran's I (no normality assumption). Requires numpy.",
    handler: (i) => runPythonScript('server/spatial_tools.py', { ...i, task: 'moran_permutation_test' }),
    parameters: { values: { type: 'array', required: true, description: 'Numeric observation per location.' }, weights: { type: 'array', required: true, description: 'n x n spatial weight matrix.' }, nPermutations: { type: 'number', description: 'Permutations (default 999).' }, seed: { type: 'number', description: 'RNG seed.' } },
  },
  // --- Pharmacokinetics / enzyme kinetics (scipy) ---
  {
    name: 'nca', category: 'Pharmacokinetics',
    description: 'Non-compartmental PK analysis: Cmax, Tmax, AUC(last/inf), kel, half-life, clearance. Requires scipy.',
    handler: (i) => runPythonScript('server/pkpd_tools.py', { ...i, task: 'nca' }),
    parameters: { time: { type: 'array', required: true, description: 'Sampling times.' }, conc: { type: 'array', required: true, description: 'Concentrations.' }, dose: { type: 'number', description: 'Dose (enables clearance).' } },
  },
  {
    name: 'one_compartment_fit', category: 'Pharmacokinetics',
    description: 'One-compartment IV bolus fit C(t)=(dose/Vd)e^(-kt) -> k, Vd, half-life, R^2. Requires scipy.',
    handler: (i) => runPythonScript('server/pkpd_tools.py', { ...i, task: 'one_compartment_fit' }),
    parameters: { time: { type: 'array', required: true, description: 'Sampling times.' }, conc: { type: 'array', required: true, description: 'Concentrations.' }, dose: { type: 'number', required: true, description: 'Administered dose.' } },
  },
  {
    name: 'michaelis_menten', category: 'Pharmacokinetics',
    description: 'Michaelis-Menten fit v=Vmax*S/(Km+S) -> Vmax, Km, R^2. Requires scipy.',
    handler: (i) => runPythonScript('server/pkpd_tools.py', { ...i, task: 'michaelis_menten' }),
    parameters: { substrate: { type: 'array', required: true, description: 'Substrate concentrations.' }, velocity: { type: 'array', required: true, description: 'Reaction velocities.' } },
  },
  {
    name: 'lineweaver_burk', category: 'Pharmacokinetics',
    description: 'Lineweaver-Burk linearization (1/v vs 1/S) -> Vmax, Km, slope, intercept. Requires numpy.',
    handler: (i) => runPythonScript('server/pkpd_tools.py', { ...i, task: 'lineweaver_burk' }),
    parameters: { substrate: { type: 'array', required: true, description: 'Substrate concentrations.' }, velocity: { type: 'array', required: true, description: 'Reaction velocities.' } },
  },
  {
    name: 'competitive_inhibition_ki', category: 'Pharmacokinetics',
    description: 'Estimate inhibition constant Ki from apparent Km vs inhibitor concentration. Requires numpy.',
    handler: (i) => runPythonScript('server/pkpd_tools.py', { ...i, task: 'competitive_inhibition_ki' }),
    parameters: { inhibitor: { type: 'array', required: true, description: 'Inhibitor concentrations [I].' }, km_apparent: { type: 'array', required: true, description: 'Apparent Km at each [I].' } },
  },
  // --- Bayesian inference (scipy) ---
  {
    name: 'beta_binomial_update', category: 'Bayesian inference',
    description: 'Beta-Binomial conjugate update -> posterior params, mean, 95% credible interval. Requires scipy.',
    handler: (i) => runPythonScript('server/bayes_tools.py', { ...i, task: 'beta_binomial_update' }),
    parameters: { successes: { type: 'number', required: true, description: 'Observed successes.' }, trials: { type: 'number', required: true, description: 'Total trials.' }, priorAlpha: { type: 'number', description: 'Prior alpha (default 1).' }, priorBeta: { type: 'number', description: 'Prior beta (default 1).' } },
  },
  {
    name: 'normal_normal_update', category: 'Bayesian inference',
    description: 'Normal-Normal conjugate update (known variance) -> posterior mean/var + credible interval. Requires scipy.',
    handler: (i) => runPythonScript('server/bayes_tools.py', { ...i, task: 'normal_normal_update' }),
    parameters: { priorMean: { type: 'number', required: true, description: 'Prior mean.' }, priorVar: { type: 'number', required: true, description: 'Prior variance.' }, dataMean: { type: 'number', description: 'Observed mean (or provide data).' }, data: { type: 'array', description: 'Raw observations (alternative to dataMean).' }, sigma2: { type: 'number', description: 'Known per-obs variance.' }, n: { type: 'number', description: 'Sample size (default 1).' } },
  },
  {
    name: 'poisson_gamma_update', category: 'Bayesian inference',
    description: 'Poisson-Gamma conjugate update -> posterior shape/rate, mean, credible interval. Requires scipy.',
    handler: (i) => runPythonScript('server/bayes_tools.py', { ...i, task: 'poisson_gamma_update' }),
    parameters: { priorShape: { type: 'number', required: true, description: 'Prior gamma shape.' }, priorRate: { type: 'number', required: true, description: 'Prior gamma rate.' }, counts: { type: 'array', description: 'Poisson counts (or sumCounts+nObs).' }, sumCounts: { type: 'number', description: 'Sum of counts.' }, nObs: { type: 'number', description: 'Number of observations.' } },
  },
  {
    name: 'bayesian_ab_test', category: 'Bayesian inference',
    description: 'Bayesian A/B test on two Beta posteriors -> P(B>A), means, expected uplift (seeded Monte Carlo). Requires numpy.',
    handler: (i) => runPythonScript('server/bayes_tools.py', { ...i, task: 'bayesian_ab_test' }),
    parameters: { successesA: { type: 'number', required: true, description: 'Successes in A.' }, trialsA: { type: 'number', required: true, description: 'Trials in A.' }, successesB: { type: 'number', required: true, description: 'Successes in B.' }, trialsB: { type: 'number', required: true, description: 'Trials in B.' }, nSamples: { type: 'number', description: 'MC samples (default 100000).' }, seed: { type: 'number', description: 'RNG seed (default 0).' } },
  },
  {
    name: 'bayes_factor_bic', category: 'Bayesian inference',
    description: 'BIC-approximation Bayes factor BF10 for two nested models + Jeffreys evidence label.',
    handler: (i) => runPythonScript('server/bayes_tools.py', { ...i, task: 'bayes_factor_bic' }),
    parameters: { bic0: { type: 'number', required: true, description: 'BIC of H0 model.' }, bic1: { type: 'number', required: true, description: 'BIC of H1 model.' } },
  },
  // --- Beta diversity / ordination (numpy/scipy) ---
  {
    name: 'bray_curtis', category: 'Beta diversity',
    description: 'Pairwise Bray-Curtis dissimilarity matrix across samples. Requires numpy.',
    handler: (i) => runPythonScript('server/beta_diversity.py', { ...i, task: 'bray_curtis' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples x features count matrix.' }, sampleIds: { type: 'array', description: 'Sample labels.' } },
  },
  {
    name: 'jaccard_distance', category: 'Beta diversity',
    description: 'Pairwise Jaccard dissimilarity on presence/absence (binarized at >0). Requires numpy.',
    handler: (i) => runPythonScript('server/beta_diversity.py', { ...i, task: 'jaccard_distance' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples x features matrix.' }, sampleIds: { type: 'array', description: 'Sample labels.' } },
  },
  {
    name: 'pcoa', category: 'Beta diversity',
    description: 'Principal Coordinates Analysis (classical MDS) on a distance matrix -> coords, eigenvalues, variance explained. Requires numpy.',
    handler: (i) => runPythonScript('server/beta_diversity.py', { ...i, task: 'pcoa' }),
    parameters: { distanceMatrix: { type: 'array', required: true, description: 'Square symmetric distance matrix.' }, nComponents: { type: 'number', description: 'Axes to return (default 2).' }, sampleIds: { type: 'array', description: 'Sample labels.' } },
  },
  {
    name: 'permanova', category: 'Beta diversity',
    description: 'PERMANOVA (Anderson 2001) pseudo-F + permutation p-value from a distance matrix + groups. Requires numpy.',
    handler: (i) => runPythonScript('server/beta_diversity.py', { ...i, task: 'permanova' }),
    parameters: { distanceMatrix: { type: 'array', required: true, description: 'Square symmetric distance matrix.' }, groups: { type: 'array', required: true, description: 'Group label per sample.' }, nPermutations: { type: 'number', description: 'Permutations (default 999).' }, seed: { type: 'number', description: 'RNG seed.' } },
  },
  {
    name: 'mantel_test', category: 'Beta diversity',
    description: 'Mantel correlation between two distance matrices + permutation p-value. Requires numpy.',
    handler: (i) => runPythonScript('server/beta_diversity.py', { ...i, task: 'mantel_test' }),
    parameters: { matrixA: { type: 'array', required: true, description: 'First distance matrix.' }, matrixB: { type: 'array', required: true, description: 'Second distance matrix (same shape).' }, nPermutations: { type: 'number', description: 'Permutations (default 999).' }, seed: { type: 'number', description: 'RNG seed.' } },
  },
  // --- Statistical power / sample size (statsmodels) ---
  {
    name: 'sample_size_two_means', category: 'Power analysis',
    description: 'Required n per group for a two-sample t-test given Cohen d, alpha, power. Requires statsmodels.',
    handler: (i) => runPythonScript('server/power_tools.py', { ...i, task: 'sample_size_two_means' }),
    parameters: { effectSize: { type: 'number', required: true, description: "Cohen's d." }, alpha: { type: 'number', description: 'Significance (default 0.05).' }, power: { type: 'number', description: 'Target power (default 0.8).' }, ratio: { type: 'number', description: 'n2/n1 (default 1).' }, alternative: { type: 'string', description: "'two-sided'|'larger'|'smaller'." } },
  },
  {
    name: 'power_two_means', category: 'Power analysis',
    description: 'Achieved power of a two-sample t-test given d, n per group, alpha. Requires statsmodels.',
    handler: (i) => runPythonScript('server/power_tools.py', { ...i, task: 'power_two_means' }),
    parameters: { effectSize: { type: 'number', required: true, description: "Cohen's d." }, nPerGroup: { type: 'number', required: true, description: 'n in group 1.' }, alpha: { type: 'number', description: 'Significance (default 0.05).' }, ratio: { type: 'number', description: 'n2/n1 (default 1).' }, alternative: { type: 'string', description: "'two-sided'|'larger'|'smaller'." } },
  },
  {
    name: 'sample_size_two_proportions', category: 'Power analysis',
    description: 'Required n per group to detect p1 vs p2 (proportion effect size h). Requires statsmodels.',
    handler: (i) => runPythonScript('server/power_tools.py', { ...i, task: 'sample_size_two_proportions' }),
    parameters: { p1: { type: 'number', required: true, description: 'Proportion 1.' }, p2: { type: 'number', required: true, description: 'Proportion 2.' }, alpha: { type: 'number', description: 'Significance (default 0.05).' }, power: { type: 'number', description: 'Target power (default 0.8).' }, ratio: { type: 'number', description: 'n2/n1 (default 1).' }, alternative: { type: 'string', description: "'two-sided'|'larger'|'smaller'." } },
  },
  {
    name: 'power_anova', category: 'Power analysis',
    description: 'Power for a one-way ANOVA given k groups, Cohen f, n per group, alpha. Requires statsmodels.',
    handler: (i) => runPythonScript('server/power_tools.py', { ...i, task: 'power_anova' }),
    parameters: { groups: { type: 'number', required: true, description: 'Number of groups k.' }, effectSize: { type: 'number', required: true, description: "Cohen's f." }, nPerGroup: { type: 'number', required: true, description: 'n per group.' }, alpha: { type: 'number', description: 'Significance (default 0.05).' } },
  },
  {
    name: 'sample_size_correlation', category: 'Power analysis',
    description: 'Required n to detect a Pearson correlation r (Fisher z formula). Requires scipy.',
    handler: (i) => runPythonScript('server/power_tools.py', { ...i, task: 'sample_size_correlation' }),
    parameters: { r: { type: 'number', required: true, description: 'Target correlation.' }, alpha: { type: 'number', description: 'Significance (default 0.05).' }, power: { type: 'number', description: 'Target power (default 0.8).' }, alternative: { type: 'string', description: "'two-sided'|'one-sided'." } },
  },
  // --- Genomic interval arithmetic (BEDTools-style, stdlib) ---
  {
    name: 'interval_merge', category: 'Genomic intervals',
    description: 'Merge overlapping/adjacent genomic intervals (half-open [start,end)).',
    handler: (i) => runPythonScript('server/genome_intervals.py', { ...i, task: 'interval_merge' }),
    parameters: { intervals: { type: 'array', required: true, description: 'List of [start,end] pairs.' }, minGap: { type: 'number', description: 'Merge neighbors within this gap (default 0).' } },
  },
  {
    name: 'interval_intersect', category: 'Genomic intervals',
    description: 'Intersections between two interval sets + total overlap bp.',
    handler: (i) => runPythonScript('server/genome_intervals.py', { ...i, task: 'interval_intersect' }),
    parameters: { a: { type: 'array', required: true, description: 'First interval set.' }, b: { type: 'array', required: true, description: 'Second interval set.' } },
  },
  {
    name: 'interval_subtract', category: 'Genomic intervals',
    description: 'Subtract interval set B from A (portions of A not covered by B).',
    handler: (i) => runPythonScript('server/genome_intervals.py', { ...i, task: 'interval_subtract' }),
    parameters: { a: { type: 'array', required: true, description: 'Source intervals.' }, b: { type: 'array', required: true, description: 'Intervals to subtract.' } },
  },
  {
    name: 'interval_coverage', category: 'Genomic intervals',
    description: 'Union coverage of intervals over a region -> covered bp + fraction (overlap-aware).',
    handler: (i) => runPythonScript('server/genome_intervals.py', { ...i, task: 'interval_coverage' }),
    parameters: { intervals: { type: 'array', required: true, description: 'Intervals to measure.' }, regionStart: { type: 'number', description: 'Region start (with regionEnd).' }, regionEnd: { type: 'number', description: 'Region end.' }, regionLength: { type: 'number', description: 'Region [0,length) alternative.' } },
  },
  {
    name: 'interval_nearest', category: 'Genomic intervals',
    description: 'Nearest feature interval + distance for each query interval (0 if overlapping).',
    handler: (i) => runPythonScript('server/genome_intervals.py', { ...i, task: 'interval_nearest' }),
    parameters: { query: { type: 'array', required: true, description: 'Query intervals.' }, features: { type: 'array', required: true, description: 'Candidate feature intervals.' } },
  },
  // --- Glycoengineering (Biomni-derived; each emits a Biomni-style outcome bundle when outputDir set) ---
  {
    name: 'n_glycosylation_motifs', category: 'Glycoengineering',
    description: 'Scan a protein sequence for N-linked glycosylation sequons (N-X-[S/T], X≠Pro).',
    handler: (i) => runPythonScript('server/glyco_tools.py', { ...i, task: 'n_glycosylation_motifs' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Protein sequence.' }, allowOverlap: { type: 'boolean', description: 'Count overlapping sequons (default false).' }, outputDir: { type: 'string', description: 'If set, write figures/tables/code/report bundle.' } },
  },
  {
    name: 'o_glycosylation_hotspots', category: 'Glycoengineering',
    description: 'Sliding-window Ser/Thr-rich O-glycosylation hotspot scan.',
    handler: (i) => runPythonScript('server/glyco_tools.py', { ...i, task: 'o_glycosylation_hotspots' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Protein sequence.' }, window: { type: 'number', description: 'Window size (default 10).' }, threshold: { type: 'number', description: 'Min S/T fraction to flag (default 0.5).' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Synthetic biology: codon optimization ---
  {
    name: 'codon_optimize', category: 'Synthetic biology',
    description: 'Codon-optimize a coding DNA sequence for a host and report CAI before/after.',
    handler: (i) => runPythonScript('server/codon_tools.py', { ...i, task: 'codon_optimize' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Coding DNA (length multiple of 3).' }, hostCodonUsage: { type: 'object', required: true, description: '{codon: relativeFrequency} host usage table.' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Biochemistry: sequence conservation ---
  {
    name: 'protein_conservation', category: 'Biochemistry',
    description: 'Per-column Shannon-entropy conservation across a pre-aligned protein MSA.',
    handler: (i) => runPythonScript('server/conservation_tools.py', { ...i, task: 'protein_conservation' }),
    parameters: { sequences: { type: 'array', required: true, description: 'Equal-length aligned protein sequences.' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Chronobiology: cosinor ---
  {
    name: 'cosinor_analysis', category: 'Chronobiology',
    description: 'Single-component cosinor fit → MESOR, amplitude, acrophase, R² for circadian data.',
    handler: (i) => runPythonScript('server/chrono_tools.py', { ...i, task: 'cosinor_analysis' }),
    parameters: { time: { type: 'array', required: true, description: 'Sample times.' }, values: { type: 'array', required: true, description: 'Measured values (same length).' }, period: { type: 'number', description: 'Cycle length (default 24).' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Microbial growth dynamics ---
  {
    name: 'logistic_growth_fit', category: 'Growth dynamics',
    description: 'Fit a logistic growth curve → carrying capacity K, rate r, N0, R². Requires scipy.',
    handler: (i) => runPythonScript('server/growth_dynamics.py', { ...i, task: 'logistic_growth_fit' }),
    parameters: { time: { type: 'array', required: true, description: 'Observation times.' }, population: { type: 'array', required: true, description: 'Population at each time.' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  {
    name: 'gompertz_growth_fit', category: 'Growth dynamics',
    description: 'Fit a Zwietering-Gompertz growth curve → asymptote A, max rate mu, lag, R². Requires scipy.',
    handler: (i) => runPythonScript('server/growth_dynamics.py', { ...i, task: 'gompertz_growth_fit' }),
    parameters: { time: { type: 'array', required: true, description: 'Observation times.' }, population: { type: 'array', required: true, description: 'Population at each time.' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  {
    name: 'lotka_volterra_simulate', category: 'Growth dynamics',
    description: 'Integrate generalized Lotka-Volterra dynamics for n interacting species. Requires scipy.',
    handler: (i) => runPythonScript('server/growth_dynamics.py', { ...i, task: 'lotka_volterra_simulate' }),
    parameters: { initialAbundances: { type: 'array', required: true, description: 'Initial abundance per species.' }, growthRates: { type: 'array', required: true, description: 'Intrinsic growth rate per species.' }, interactionMatrix: { type: 'array', required: true, description: 'n×n interaction coefficients.' }, timePoints: { type: 'array', required: true, description: 'Increasing evaluation times.' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Genomic prediction ---
  {
    name: 'gblup', category: 'Genomic prediction',
    description: 'Genomic prediction (GBLUP / ridge) of breeding values with CV/hold-out accuracy. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/genomic_prediction.py', { ...i, task: 'gblup' }),
    parameters: { genotypes: { type: 'array', required: true, description: 'individuals × markers dosage matrix.' }, phenotypes: { type: 'array', required: true, description: 'Phenotype per individual.' }, lambdaReg: { type: 'number', description: 'Ridge penalty (default = #markers).' }, testIndices: { type: 'array', description: 'Hold-out indices (else 5-fold CV).' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Flagship hybrid RNA-seq pipeline ---
  {
    name: 'rnaseq_upstream', category: 'RNA-seq pipeline',
    description: 'Hybrid short+long-read RNA-seq upstream orchestrator (Phases 1-3: fastp trim, STAR index/align with --sjdbOverhang=RL-1, minimap2 -ax splice fed the short-read SJ map, stringtie --merge/gffcompare, decoy-aware salmon index+quant). Builds the exact real commands; runs binaries when present, else returns an honest per-step "unavailable" plan (never fabricates counts).',
    handler: (i) => runPythonScript('server/rnaseq_pipeline.py', { ...i, task: 'rnaseq_upstream' }),
    parameters: { readLength: { type: 'number', description: 'Short-read length (sets STAR --sjdbOverhang = RL-1).' }, longPlatform: { type: 'string', description: "'nanopore' (Q10) or 'pacbio_hifi' (Q20)." }, referenceFasta: { type: 'string', description: 'Reference genome FASTA path.' }, annotationGtf: { type: 'string', description: 'Reference GTF path.' }, read1: { type: 'string', description: 'Short-read R1 FASTQ.' }, read2: { type: 'string', description: 'Short-read R2 FASTQ.' }, longReads: { type: 'string', description: 'Long-read FASTQ.' }, samples: { type: 'array', description: 'Sample IDs for stringtie merge.' }, execute: { type: 'boolean', description: 'Run available binaries (default false = plan only).' } },
  },
  {
    name: 'rnaseq_tximport', category: 'RNA-seq pipeline',
    description: 'Phase 4a: summarize transcript-level salmon quantifications to gene-level counts (lengthScaledTPM-style) via a tx2gene map.',
    handler: (i) => runPythonScript('server/rnaseq_pipeline.py', { ...i, task: 'rnaseq_tximport' }),
    parameters: { quant: { type: 'object', required: true, description: '{sample:{transcript:{counts,effLength}}}.' }, tx2gene: { type: 'object', required: true, description: '{transcript: gene} map.' } },
  },
  {
    name: 'rnaseq_deseq', category: 'RNA-seq pipeline',
    description: 'Phase 4b flagship: DESeq2-style differential expression (median-of-ratios size factors → mean-dispersion trend + shrinkage → per-gene NB-GLM Wald test → BH FDR → normal-prior log2FC shrinkage → VST/PCA). Emits the full figure set (PCA, MA, volcano, dispersion, p-value hist, sample-distance & top-gene heatmaps, size factors, library sizes), result tables, and a report/document/article via the outcome bundle. Requires statsmodels.',
    handler: (i) => runPythonScript('server/rnaseq_pipeline.py', { ...i, task: 'rnaseq_deseq' }),
    parameters: { counts: { type: 'object', required: true, description: 'Gene count matrix {gene:[counts…]} or genes×samples array.' }, conditions: { type: 'array', required: true, description: 'Group label per sample (two groups).' }, samples: { type: 'array', description: 'Sample IDs aligned to columns.' }, reference: { type: 'string', description: 'Reference/control group level.' }, alpha: { type: 'number', description: 'FDR threshold (default 0.05).' }, minBaseMean: { type: 'number', description: 'Independent-filtering min base mean (default 1).' }, outputFormat: { type: 'string', description: "'report' | 'document' (adds .docx) | 'article' (full sections)." }, outputDir: { type: 'string', description: 'If set, write the full figures/tables/report bundle.' } },
  },
  // --- Epitranscriptomics ---
  {
    name: 'm6a_drach_scan', category: 'Epitranscriptomics',
    description: 'Scan an RNA/DNA transcript for the m6A DRACH consensus motif (D-R-A-C-H) and report candidate methylated-adenosine positions. Deterministic sequence analysis — no read data, no invented confidence scores. Emits an outcome bundle when outputDir is set.',
    handler: (i) => runPythonScript('server/epitranscriptomics.py', { ...i, task: 'm6a_drach_scan' }),
    parameters: { sequence: { type: 'string', required: true, description: 'RNA or DNA transcript sequence (A/C/G/U/T).' }, outputDir: { type: 'string', description: 'If set, write figures/tables/report bundle.' } },
  },
  // --- Single-cell RNA velocity (dynamo/scVelo steady-state; numpy) ---
  {
    name: 'velocity_estimate', category: 'Single-cell dynamics',
    description: 'RNA velocity: per-gene steady-state degradation rate gamma (through-origin u~s fit) and velocity v = u - gamma*s, with per-gene R^2. numpy only (dynamo/scVelo-style). Emits an outcome bundle (phase portrait + per-gene table) when outputDir is set.',
    handler: (i) => runPythonScript('server/rna_velocity.py', { ...i, task: 'velocity_estimate' }),
    parameters: { unspliced: { type: 'object', required: true, description: 'Unspliced counts {gene:[cells]} or genes×cells matrix.' }, spliced: { type: 'object', required: true, description: 'Spliced counts, same shape.' }, mode: { type: 'string', description: "'steady_state' (extreme-quantile) or 'deterministic' (all cells)." }, quantile: { type: 'number', description: 'Top-spliced fraction for steady-state fit (default 0.05).' }, geneNames: { type: 'array', description: 'Gene names if matrices are given.' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  {
    name: 'velocity_stream_projection', category: 'Single-cell dynamics',
    description: 'Project high-dimensional RNA velocities onto a 2-D embedding via a cosine-correlation transition kernel (scVelo-style), yielding per-cell velocity arrows. numpy only.',
    handler: (i) => runPythonScript('server/rna_velocity.py', { ...i, task: 'velocity_stream_projection' }),
    parameters: { expression: { type: 'object', required: true, description: 'Spliced/smoothed expression {gene:[cells]} or genes×cells.' }, velocity: { type: 'object', required: true, description: 'Velocity matrix, same shape (from velocity_estimate).' }, embedding: { type: 'array', required: true, description: 'cells×2 embedding coordinates.' }, nNeighbors: { type: 'number', description: 'kNN in expression space (default 10).' }, sigma: { type: 'number', description: 'Kernel bandwidth (default 0.05).' }, outputDir: { type: 'string', description: 'If set, write outcome bundle.' } },
  },
  // --- Spatial transcriptomics deconvolution / mapping (Tangram goal; scipy/POT) ---
  {
    name: 'nnls_deconvolution', category: 'Spatial transcriptomics',
    description: 'Per-spot cell-type proportions by non-negative least squares of the spot profile onto reference cell-type signatures (SPOTlight/NNLS-style). Requires scipy.',
    handler: (i) => runPythonScript('server/spatial_deconvolution.py', { ...i, task: 'nnls_deconvolution' }),
    parameters: { spots: { type: 'array', required: true, description: 'spots×genes expression matrix.' }, signatures: { type: 'array', required: true, description: 'cellTypes×genes reference signature matrix.' }, cellTypes: { type: 'array', description: 'Cell-type names aligned to signature rows.' }, spotIds: { type: 'array', description: 'Spot IDs aligned to rows.' } },
  },
  {
    name: 'ot_map_cells_to_spots', category: 'Spatial transcriptomics',
    description: 'Map single cells onto spatial spots via entropic optimal transport with a (1 − cosine similarity) cost on shared genes (deterministic Tangram-goal alternative). Returns a probabilistic cell×spot map. Requires POT.',
    handler: (i) => runPythonScript('server/spatial_deconvolution.py', { ...i, task: 'ot_map_cells_to_spots' }),
    parameters: { cells: { type: 'array', required: true, description: 'cells×genes expression matrix.' }, spots: { type: 'array', required: true, description: 'spots×genes expression matrix (shared genes).' }, reg: { type: 'number', description: 'Sinkhorn entropic regularization (default 0.05).' }, spotIds: { type: 'array', description: 'Spot IDs aligned to rows.' } },
  },
  // --- Trajectory inference / pseudotime (numpy/scipy) ---
  {
    name: 'diffusion_pseudotime', category: 'Trajectory inference',
    description: 'Diffusion-map pseudotime: Markov transition eigendecomposition → per-cell pseudotime distance from a root cell. numpy/scipy. Recovers a linear trajectory ordering (validated).',
    handler: (i) => runPythonScript('server/trajectory.py', { ...i, task: 'diffusion_pseudotime' }),
    parameters: { expression: { type: 'array', required: true, description: 'cells×genes matrix.' }, rootCell: { type: 'number', description: 'Root cell index (default 0).' }, sigma: { type: 'number', description: 'Gaussian kernel bandwidth (default = median distance).' }, nComps: { type: 'number', description: 'Diffusion components (default 10).' } },
  },
  {
    name: 'mst_pseudotime', category: 'Trajectory inference',
    description: 'Minimum-spanning-tree pseudotime: shortest-path distance along the MST of a cell graph from a root. numpy/scipy.',
    handler: (i) => runPythonScript('server/trajectory.py', { ...i, task: 'mst_pseudotime' }),
    parameters: { expression: { type: 'array', required: true, description: 'cells×genes matrix.' }, rootCell: { type: 'number', description: 'Root cell index (default 0).' }, nNeighbors: { type: 'number', description: 'kNN graph size; omit for a complete graph.' } },
  },
  // --- Gene regulatory network inference (scikit-learn) ---
  {
    name: 'genie3', category: 'Gene regulatory networks',
    description: 'GENIE3 tree-based GRN inference: per-target Random-Forest feature importance ranks candidate regulators. Requires scikit-learn. Validated: recovers the true regulator.',
    handler: (i) => runPythonScript('server/grn_inference.py', { ...i, task: 'genie3' }),
    parameters: { expression: { type: 'array', required: true, description: 'samples×genes matrix.' }, geneNames: { type: 'array', required: true, description: 'Gene names (one per column).' }, regulators: { type: 'array', description: 'Candidate regulator subset (default all).' }, topEdges: { type: 'number', description: 'Top edges to return (default 20).' }, nEstimators: { type: 'number', description: 'RF trees per target (default 100).' }, seed: { type: 'number', description: 'RNG seed.' } },
  },
  {
    name: 'aracne_mi', category: 'Gene regulatory networks',
    description: 'ARACNe mutual-information network with Data-Processing-Inequality pruning of indirect edges. Requires scikit-learn. Validated: prunes the X–Z indirect edge in an X→Y→Z chain.',
    handler: (i) => runPythonScript('server/grn_inference.py', { ...i, task: 'aracne_mi' }),
    parameters: { expression: { type: 'array', required: true, description: 'samples×genes matrix.' }, geneNames: { type: 'array', required: true, description: 'Gene names (one per column).' }, miThreshold: { type: 'number', description: 'Minimum MI to keep an edge (default 0).' }, seed: { type: 'number', description: 'RNG seed.' } },
  },
  // --- Multi-omics integration (numpy/scipy/scikit-learn) ---
  {
    name: 'snf', category: 'Multi-omics integration',
    description: 'Similarity Network Fusion (Wang 2014): fuse per-view sample-similarity networks + spectral clustering. Validated: recovers true clusters (ARI=1) from complementary noisy views.',
    handler: (i) => runPythonScript('server/multiomics_integration.py', { ...i, task: 'snf' }),
    parameters: { views: { type: 'array', required: true, description: 'List of sample×feature matrices (same samples).' }, nClusters: { type: 'number', required: true, description: 'Number of clusters.' }, K: { type: 'number', description: 'KNN neighbors (default min(20,n/2)).' }, t: { type: 'number', description: 'Fusion iterations (default 20).' }, mu: { type: 'number', description: 'Affinity scale (default 0.5).' } },
  },
  {
    name: 'cca', category: 'Multi-omics integration',
    description: 'Canonical Correlation Analysis between two omics views → canonical correlations. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/multiomics_integration.py', { ...i, task: 'cca' }),
    parameters: { X: { type: 'array', required: true, description: 'First view n×p.' }, Y: { type: 'array', required: true, description: 'Second view n×q (same n).' }, nComponents: { type: 'number', description: 'Canonical components.' } },
  },
  {
    name: 'joint_nmf', category: 'Multi-omics integration',
    description: 'Integrative NMF across feature-concatenated non-negative omics views → shared sample factors. Requires scikit-learn.',
    handler: (i) => runPythonScript('server/multiomics_integration.py', { ...i, task: 'joint_nmf' }),
    parameters: { views: { type: 'array', required: true, description: 'List of non-negative sample×feature matrices.' }, nComponents: { type: 'number', description: 'Factorization rank (default 2).' } },
  },
  // --- Statistical genetics: Mendelian randomization (numpy/scipy) ---
  {
    name: 'mr_ivw', category: 'Statistical genetics',
    description: 'Two-sample Mendelian randomization, inverse-variance weighted (fixed-effect, through origin) + Cochran Q heterogeneity. Validated: recovers a known causal effect from GWAS summary stats.',
    handler: (i) => runPythonScript('server/mendelian_randomization.py', { ...i, task: 'mr_ivw' }),
    parameters: { betaExposure: { type: 'array', required: true, description: 'Per-SNP SNP→exposure effects.' }, betaOutcome: { type: 'array', required: true, description: 'Per-SNP SNP→outcome effects.' }, seOutcome: { type: 'array', required: true, description: 'SEs of betaOutcome (weights=1/se²).' }, seExposure: { type: 'array', description: 'SEs of betaExposure (optional).' } },
  },
  {
    name: 'mr_egger', category: 'Statistical genetics',
    description: 'MR-Egger regression: pleiotropy-robust causal slope + Egger intercept (directional-pleiotropy test). Validated: detects an injected pleiotropic intercept IVW misses.',
    handler: (i) => runPythonScript('server/mendelian_randomization.py', { ...i, task: 'mr_egger' }),
    parameters: { betaExposure: { type: 'array', required: true, description: 'Per-SNP SNP→exposure effects.' }, betaOutcome: { type: 'array', required: true, description: 'Per-SNP SNP→outcome effects.' }, seOutcome: { type: 'array', required: true, description: 'SEs of betaOutcome (weights=1/se²).' }, seExposure: { type: 'array', description: 'SEs of betaExposure (optional).' } },
  },
  // --- RNA secondary structure (stdlib) ---
  {
    name: 'analyze_rna_secondary_structure_features', category: 'RNA structure',
    description: 'Parse RNA dot-bracket notation into base pairs and derive topological features (base pairs, stems, longest/average stem, paired %). Deterministic; the fabricated free-energy add-on from the source was dropped.',
    handler: (i) => runPythonScript('server/rna_structure_tools.py', { ...i, task: 'analyze_rna_secondary_structure_features' }),
    parameters: { dot_bracket_structure: { type: 'string', required: true, description: "Dot-bracket structure, e.g. '(((...)))'; supports (), [], {} and '.'." } },
  },
  // --- Enzyme kinetics & compartmental PK (scipy) ---
  {
    name: 'protease_kinetics', category: 'Pharmacokinetics',
    description: 'Michaelis-Menten enzymology from a fluorescence time-course matrix: initial velocities (polyfit) → Vmax, Km, kcat, catalytic efficiency (kcat/Km). Requires scipy. Validated: recovers Vmax=100/Km=10.',
    handler: (i) => runPythonScript('server/enzyme_pk_tools.py', { ...i, task: 'protease_kinetics' }),
    parameters: { time_points: { type: 'array', required: true, description: 'Shared time axis.' }, fluorescence_data: { type: 'array', required: true, description: '2D rows=substrate concentrations × cols=time.' }, substrate_concentrations: { type: 'array', required: true, description: 'One concentration per row (uM).' }, enzyme_concentration: { type: 'number', required: true, description: 'Total enzyme (uM).' }, initial_fraction: { type: 'number', description: 'Leading fraction for initial-velocity fit (default 0.2).' } },
  },
  {
    name: 'bi_exponential_pk', category: 'Pharmacokinetics',
    description: 'Two-phase (distribution + elimination) PK fit C(t)=A·e^(−αt)+B·e^(−βt) → half-lives. Requires scipy. Validated: recovers α/β and elimination t½.',
    handler: (i) => runPythonScript('server/enzyme_pk_tools.py', { ...i, task: 'bi_exponential_pk' }),
    parameters: { time: { type: 'array', required: true, description: 'Sampling times.' }, concentration: { type: 'array', required: true, description: 'Concentrations (same length).' } },
  },
  // --- Drug dissolution / release kinetics (scipy) ---
  {
    name: 'drug_release_kinetics', category: 'Pharmacology',
    description: 'Fit 4 dissolution models (zero-order, first-order, Higuchi, Korsmeyer-Peppas), pick best by R², derive t50 + transport mechanism. Requires scipy. Validated: t50=2.5 h on a zero-order profile.',
    handler: (i) => runPythonScript('server/dissolution_tools.py', { ...i, task: 'drug_release_kinetics' }),
    parameters: { time_points: { type: 'array', required: true, description: 'Times (h), ascending, ≥4.' }, concentration_data: { type: 'array', required: true, description: 'Cumulative release/amount (same length).' }, total_drug_loaded: { type: 'number', description: 'Total loaded; if set, normalize to % released.' }, drug_name: { type: 'string', description: 'Label (default Drug).' } },
  },
  // --- Systems biology dynamics (numpy/scipy ODE) ---
  {
    name: 'protein_dimerization_equilibrium', category: 'Systems biology',
    description: 'Monomer↔dimer (2M↔D) equilibrium: solve free monomer from mass balance → dimer conc + fraction dimerized. Supports a titration series. numpy. Validated: mass balance exact; strong/weak-binding limits.',
    handler: (i) => runPythonScript('server/systems_dynamics_tools.py', { ...i, task: 'protein_dimerization_equilibrium' }),
    parameters: { kd: { type: 'number', required: true, description: 'Dissociation constant (uM).' }, totalConcentration: { type: 'number', description: 'Total protein (uM), single-point mode.' }, totalConcentrations: { type: 'array', description: 'Titration series (uM); returns arrays.' } },
  },
  {
    name: 'simulate_gene_circuit_with_growth_feedback', category: 'Systems biology',
    description: 'ODE of gene expression with growth dilution: dP/dt = k_tx − (k_deg + growth)·P. Requires scipy. Validated: approaches analytic steady state k_tx/(k_deg+growth).',
    handler: (i) => runPythonScript('server/systems_dynamics_tools.py', { ...i, task: 'simulate_gene_circuit_with_growth_feedback' }),
    parameters: { k_transcription: { type: 'number', required: true, description: 'Transcription rate.' }, k_degradation: { type: 'number', required: true, description: 'Degradation rate.' }, growthRate: { type: 'number', required: true, description: 'Growth dilution rate.' }, initial: { type: 'number', description: 'Initial level (default 0).' }, tMax: { type: 'number', description: 'End time (default 50).' }, nPoints: { type: 'number', description: 'Output points (default 200).' } },
  },
  {
    name: 'simulate_protein_signaling_network', category: 'Systems biology',
    description: 'Linear activation cascade ODE (stage i activated by stage i−1) driven by a sustained stimulus. Requires scipy. Validated: each stage reaches its analytic cascade fixed point.',
    handler: (i) => runPythonScript('server/systems_dynamics_tools.py', { ...i, task: 'simulate_protein_signaling_network' }),
    parameters: { stimulus: { type: 'number', required: true, description: 'Sustained input to stage 0.' }, nStages: { type: 'number', description: 'Cascade length (default 3).' }, rates: { type: 'array', description: 'Per-stage activation rates.' }, deactivationRates: { type: 'array', description: 'Per-stage deactivation rates.' }, initial: { type: 'number', description: 'Initial level (default 0).' }, tMax: { type: 'number', description: 'End time (default 50).' }, nPoints: { type: 'number', description: 'Output points (default 200).' } },
  },
  // --- Biosignal / physiological waveform analysis (scipy.signal) ---
  {
    name: 'abr_waveform_p1_metrics', category: 'Biosignal analysis',
    description: 'Auditory brainstem response P1 metrics (dominant-peak latency + amplitude) via peak detection. Requires scipy. Validated: latency within 0.2 ms of a known peak.',
    handler: (i) => runPythonScript('server/biosignal_tools.py', { ...i, task: 'abr_waveform_p1_metrics' }),
    parameters: { signal: { type: 'array', required: true, description: 'ABR voltage samples.' }, samplingRateHz: { type: 'number', required: true, description: 'Sampling rate (Hz).' } },
  },
  {
    name: 'calcium_transient_dynamics', category: 'Biosignal analysis',
    description: 'Ca²⁺ transient metrics: baseline, peak amplitude, time-to-peak, exponential decay τ (curve fit) + R². Requires scipy. Validated: recovers τ=0.5 s.',
    handler: (i) => runPythonScript('server/biosignal_tools.py', { ...i, task: 'calcium_transient_dynamics' }),
    parameters: { time: { type: 'array', required: true, description: 'Time vector (s).' }, signal: { type: 'array', required: true, description: 'Fluorescence trace (F or dF/F).' } },
  },
  {
    name: 'hemodynamic_waveform', category: 'Biosignal analysis',
    description: 'Arterial pressure waveform metrics: systolic/diastolic/MAP + heart rate from beat detection. Requires scipy. Validated: HR=60, systolic=120, diastolic=40 on a synthetic waveform.',
    handler: (i) => runPythonScript('server/biosignal_tools.py', { ...i, task: 'hemodynamic_waveform' }),
    parameters: { signal: { type: 'array', required: true, description: 'Arterial pressure samples.' }, samplingRateHz: { type: 'number', required: true, description: 'Sampling rate (Hz).' } },
  },
  // --- Molecular biology / cloning (stdlib) ---
  {
    name: 'in_silico_pcr', category: 'Molecular biology',
    description: 'In-silico PCR: locate forward/reverse primers on template + reverse strand and return the amplicon + length. Honest error if a primer is absent.',
    handler: (i) => runPythonScript('server/molbio_tools.py', { ...i, task: 'in_silico_pcr' }),
    parameters: { template: { type: 'string', required: true, description: 'DNA template.' }, forwardPrimer: { type: 'string', required: true, description: 'Forward primer.' }, reversePrimer: { type: 'string', required: true, description: 'Reverse primer.' } },
  },
  {
    name: 'restriction_digest_fragments', category: 'Molecular biology',
    description: 'Restriction digest → fragment list (start/end/length/sequence). Built-in EcoRI/BamHI/HindIII or a custom recognitionSite; linear or circular.',
    handler: (i) => runPythonScript('server/molbio_tools.py', { ...i, task: 'restriction_digest_fragments' }),
    parameters: { sequence: { type: 'string', required: true, description: 'DNA sequence.' }, enzyme: { type: 'string', description: 'EcoRI | BamHI | HindIII.' }, recognitionSite: { type: 'string', description: 'Custom recognition site (e.g. GAATTC).' }, circular: { type: 'boolean', description: 'Treat as circular (default false).' }, cutOffset: { type: 'number', description: 'Cut offset for custom sites.' } },
  },
  {
    name: 'find_sequence_mutations', category: 'Molecular biology',
    description: 'Compare two equal-length (aligned) sequences → substitution list + percent identity. Honest error if lengths differ.',
    handler: (i) => runPythonScript('server/molbio_tools.py', { ...i, task: 'find_sequence_mutations' }),
    parameters: { reference: { type: 'string', required: true, description: 'Reference sequence.' }, query: { type: 'string', required: true, description: 'Query sequence (same length).' } },
  },
  {
    name: 'design_primer', category: 'Molecular biology',
    description: 'Design forward/reverse primers from a sequence, choosing lengths whose Tm is closest to target (Wallace / GC rule). Returns primers, Tm, GC%.',
    handler: (i) => runPythonScript('server/molbio_tools.py', { ...i, task: 'design_primer' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Template DNA.' }, targetTm: { type: 'number', description: 'Target Tm °C (default 60).' }, minLen: { type: 'number', description: 'Min primer length (default 18).' }, maxLen: { type: 'number', description: 'Max primer length (default 25).' } },
  },
  {
    name: 'primer_binding_scan', category: 'Molecular biology',
    description: 'Scan both template strands for primer binding sites allowing up to maxMismatches; returns sites (strand/start/mismatches) + count.',
    handler: (i) => runPythonScript('server/molbio_tools.py', { ...i, task: 'primer_binding_scan' }),
    parameters: { template: { type: 'string', required: true, description: 'DNA template.' }, primer: { type: 'string', required: true, description: 'Primer sequence.' }, maxMismatches: { type: 'number', description: 'Allowed mismatches (default 0).' } },
  },
  // --- CRISPR / cloning (stdlib) ---
  {
    name: 'crispr_cut_site', category: 'CRISPR / cloning',
    description: 'Locate a Cas9 protospacer + PAM on either strand and predict the blunt cut site (3 bp 5′ of the PAM). Honest error if no PAM-adjacent match.',
    handler: (i) => runPythonScript('server/crispr_cloning_tools.py', { ...i, task: 'crispr_cut_site' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Target DNA.' }, guide: { type: 'string', required: true, description: '20-nt spacer/guide.' }, pam: { type: 'string', description: "PAM IUPAC pattern (default 'NGG')." } },
  },
  {
    name: 'cas9_indel_spectrum', category: 'CRISPR / cloning',
    description: 'Classify edited amplicon reads vs reference (WT/insertion/deletion/substitution) → frameshift & editing fractions.',
    handler: (i) => runPythonScript('server/crispr_cloning_tools.py', { ...i, task: 'cas9_indel_spectrum' }),
    parameters: { reference: { type: 'string', required: true, description: 'Reference amplicon.' }, editedSequences: { type: 'array', required: true, description: 'List of edited read sequences.' } },
  },
  {
    name: 'golden_gate_assembly', category: 'CRISPR / cloning',
    description: 'Order and join overhang-compatible fragments (Golden Gate) → assembled construct + topology. Honest error on ambiguous/disconnected overhangs.',
    handler: (i) => runPythonScript('server/crispr_cloning_tools.py', { ...i, task: 'golden_gate_assembly' }),
    parameters: { fragments: { type: 'array', required: true, description: 'List of fragment DNA strings.' }, overhangLength: { type: 'number', description: 'Overhang length (default 4).' } },
  },
  {
    name: 'design_verification_primers', category: 'CRISPR / cloning',
    description: 'Design genotyping primers flanking an edit position so the amplicon spans it (Tm + amplicon coordinates).',
    handler: (i) => runPythonScript('server/crispr_cloning_tools.py', { ...i, task: 'design_verification_primers' }),
    parameters: { sequence: { type: 'string', required: true, description: 'Locus DNA.' }, editPosition: { type: 'number', required: true, description: '0-based edit index.' }, flank: { type: 'number', description: 'Flank size (default 200).' }, primerLen: { type: 'number', description: 'Primer length (default 20).' } },
  },
  // --- Preclinical pharmacology / toxicology assays (numpy/scipy) ---
  {
    name: 'xenograft_tgi', category: 'Preclinical pharmacology',
    description: 'Tumor Growth Inhibition (TGI%) from treated vs control volumes + fold-changes and a Welch t-test on final volumes (2D). Requires numpy/scipy.',
    handler: (i) => runPythonScript('server/pharmacology_assay_tools.py', { ...i, task: 'xenograft_tgi' }),
    parameters: { treatedVolumes: { type: 'array', required: true, description: '1D means or 2D animals×timepoints.' }, controlVolumes: { type: 'array', required: true, description: 'Same shape as treated.' }, days: { type: 'array', description: 'Timepoint labels.' } },
  },
  {
    name: 'atp_luminescence_viability', category: 'Preclinical pharmacology',
    description: 'Cell-viability % from ATP luminescence, normalized to vehicle control (blank-corrected). Requires numpy.',
    handler: (i) => runPythonScript('server/pharmacology_assay_tools.py', { ...i, task: 'atp_luminescence_viability' }),
    parameters: { luminescence: { type: 'array', required: true, description: 'Treated-well RLU readings.' }, vehicleControl: { type: 'number', required: true, description: 'Vehicle RLU (number or array).' }, blank: { type: 'number', description: 'Background blank (default 0).' } },
  },
  {
    name: 'vcog_ctcae_grade', category: 'Preclinical pharmacology',
    description: 'Map a lab value to a VCOG-CTCAE v1.1 adverse-event grade (neutropenia/thrombocytopenia/anemia/ALT). Honest error on unknown parameter.',
    handler: (i) => runPythonScript('server/pharmacology_assay_tools.py', { ...i, task: 'vcog_ctcae_grade' }),
    parameters: { parameter: { type: 'string', required: true, description: 'neutropenia|thrombocytopenia|anemia|alt_increase.' }, value: { type: 'number', required: true, description: 'Measured value.' }, lowerLimitNormal: { type: 'number', description: 'Override LLN.' }, upperLimitNormal: { type: 'number', description: 'Override ULN.' } },
  },
  {
    name: 'alpha_particle_dosimetry', category: 'Preclinical pharmacology',
    description: 'MIRD absorbed dose (Gy) from cumulated activity, energy per decay, organ mass and absorbed fraction. Requires numpy.',
    handler: (i) => runPythonScript('server/pharmacology_assay_tools.py', { ...i, task: 'alpha_particle_dosimetry' }),
    parameters: { cumulatedActivity_Bq_s: { type: 'number', required: true, description: 'Time-integrated activity (Bq·s).' }, energyPerDecay_MeV: { type: 'number', required: true, description: 'Energy per decay (MeV).' }, organMass_kg: { type: 'number', required: true, description: 'Organ mass (kg).' }, absorbedFraction: { type: 'number', description: 'Absorbed fraction (default 1.0).' } },
  },
  // --- Omics association / structure / barcoding (numpy/scipy/statsmodels) ---
  {
    name: 'methylome_wide_association', category: 'Epigenomics',
    description: 'Methylome-wide association: per-CpG OLS of methylation ~ phenotype (+covariates) with BH FDR. Requires statsmodels. Validated: recovers a spiked associated site.',
    handler: (i) => runPythonScript('server/omics_assoc_tools.py', { ...i, task: 'methylome_wide_association' }),
    parameters: { methylation: { type: 'array', required: true, description: 'samples×sites beta/M-values.' }, phenotype: { type: 'array', required: true, description: 'Phenotype per sample.' }, siteIds: { type: 'array', description: 'CpG site IDs.' }, covariates: { type: 'array', description: 'samples×k covariate matrix.' } },
  },
  {
    name: 'compare_protein_structures', category: 'Structural biology',
    description: 'RMSD between two structures with Kabsch superposition (numpy). Accepts matched CA coordinate arrays or PDB text (biopython). Validated: rigid transform → RMSD≈0 after superposition.',
    handler: (i) => runPythonScript('server/omics_assoc_tools.py', { ...i, task: 'compare_protein_structures' }),
    parameters: { coordsA: { type: 'array', description: 'N×3 CA coordinates.' }, coordsB: { type: 'array', description: 'N×3 CA coordinates (matched order).' }, pdbA: { type: 'string', description: 'PDB text (alt to coordsA).' }, pdbB: { type: 'string', description: 'PDB text (alt to coordsB).' } },
  },
  {
    name: 'barcode_sequencing', category: 'Sequencing utilities',
    description: 'Demultiplex reads to barcodes (best match within maxMismatches) → per-barcode counts, unassigned, Shannon diversity.',
    handler: (i) => runPythonScript('server/omics_assoc_tools.py', { ...i, task: 'barcode_sequencing' }),
    parameters: { reads: { type: 'array', required: true, description: 'DNA read strings.' }, barcodes: { type: 'object', required: true, description: '{name: sequence} or list of sequences.' }, maxMismatches: { type: 'number', description: 'Allowed mismatches (default 0).' }, barcodeStart: { type: 'number', description: 'Barcode start offset (default 0).' } },
  },
  // --- Bioimage analysis (OpenCV) ---
  {
    name: 'pixel_distribution', category: 'Bioimage analysis',
    description: 'Grayscale intensity statistics + histogram (mean/std/median/percentiles). Requires numpy.',
    handler: (i) => runPythonScript('server/image_tools.py', { ...i, task: 'pixel_distribution' }),
    parameters: { image: { type: 'array', required: true, description: '2D grayscale image (or path).' }, bins: { type: 'number', description: 'Histogram bins (default 256).' } },
  },
  {
    name: 'count_colonies', category: 'Bioimage analysis',
    description: 'Count bacterial colonies / blobs via Otsu threshold + connected components (area-filtered). Requires OpenCV. Validated: 5 drawn circles → 5.',
    handler: (i) => runPythonScript('server/image_tools.py', { ...i, task: 'count_colonies' }),
    parameters: { image: { type: 'array', required: true, description: '2D grayscale (bright blobs on dark bg).' }, invert: { type: 'boolean', description: 'Invert dark-on-light input.' }, minArea: { type: 'number', description: 'Min component area px (default 5).' } },
  },
  {
    name: 'optical_flow_deformation', category: 'Bioimage analysis',
    description: 'Dense Farneback optical flow between two frames → mean flow (x,y), magnitude, divergence, curl. Requires OpenCV. Validated: 3-px shift recovered.',
    handler: (i) => runPythonScript('server/image_tools.py', { ...i, task: 'optical_flow_deformation' }),
    parameters: { frame1: { type: 'array', required: true, description: 'First frame (2D).' }, frame2: { type: 'array', required: true, description: 'Second frame (2D, same shape).' } },
  },
  {
    name: 'ciliary_beat_frequency', category: 'Bioimage analysis',
    description: 'Ciliary/oscillation beat frequency: FFT of the per-frame mean intensity time series. Requires numpy. Validated: recovers 5 Hz.',
    handler: (i) => runPythonScript('server/image_tools.py', { ...i, task: 'ciliary_beat_frequency' }),
    parameters: { frames: { type: 'array', required: true, description: '3D image sequence [t][h][w].' }, samplingRateHz: { type: 'number', required: true, description: 'Acquisition rate (fps).' } },
  },
  // --- Cell motility (trajectory-based; numpy/scikit-learn) ---
  {
    name: 'cell_motility_metrics', category: 'Cell motility',
    description: 'Per-track motility: path length, net displacement, mean speed, directionality ratio, MSD(lag1) + population means. numpy.',
    handler: (i) => runPythonScript('server/cell_motility_tools.py', { ...i, task: 'cell_motility_metrics' }),
    parameters: { tracks: { type: 'array', required: true, description: 'List of trajectories (each a list of [x,y]).' }, dt: { type: 'number', description: 'Time per step (default 1).' }, pixelSize: { type: 'number', description: 'Units per pixel (default 1).' } },
  },
  {
    name: 'cluster_motility_patterns', category: 'Cell motility',
    description: 'Cluster cell tracks by [speed, directionality, displacement] with standardized KMeans. Requires scikit-learn. Validated: separates fast-straight vs slow-random (ARI=1).',
    handler: (i) => runPythonScript('server/cell_motility_tools.py', { ...i, task: 'cluster_motility_patterns' }),
    parameters: { tracks: { type: 'array', required: true, description: 'List of trajectories.' }, nClusters: { type: 'number', description: 'KMeans clusters (default 2).' }, dt: { type: 'number', description: 'Time per step (default 1).' }, pixelSize: { type: 'number', description: 'Units per pixel (default 1).' } },
  },
  // --- Quantitative proteomics (numpy/scipy/scikit-learn) ---
  {
    name: 'maxlfq_quantify', category: 'Proteomics',
    description: 'MaxLFQ-style label-free protein quantification: pairwise median peptide log2-ratios → least-squares abundance profile anchored to summed intensity. Requires numpy. Validated: recovers a known 1:2:4 sample ratio; samples with no linking peptide → null (never imputed).',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'maxlfq_quantify' }),
    parameters: { peptides: { type: 'object', required: true, description: '{proteinId: [[pep1_s1, pep1_s2, ...], ...]} peptide intensities per protein (peptides×samples).' }, sampleNames: { type: 'array', description: 'Sample column names.' }, treatZeroAsMissing: { type: 'boolean', description: 'Treat 0 intensity as missing (default true).' } },
  },
  {
    name: 'normalize_intensities', category: 'Proteomics',
    description: 'Median or quantile normalization of a samples×features intensity matrix (log or linear space). Requires numpy. Validated: median-norm equalizes sample medians; quantile-norm makes all samples share one distribution.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'normalize_intensities' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples×features intensity matrix.' }, method: { type: 'string', description: "'median' or 'quantile' (default median)." }, logSpace: { type: 'boolean', description: 'Operate in log2 space (default false).' }, treatZeroAsMissing: { type: 'boolean', description: 'Treat 0 as missing (default false).' } },
  },
  {
    name: 'impute_missing', category: 'Proteomics',
    description: 'Missing-value imputation: deterministic k-NN (scikit-learn), column-min fraction, or seeded MinProb (down-shifted normal). Requires numpy (+scikit-learn for knn). Validated: fills exactly the missing cells, leaves observed unchanged, MinProb seed-reproducible.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'impute_missing' }),
    parameters: { matrix: { type: 'array', required: true, description: 'samples×features matrix (missing = null/0).' }, method: { type: 'string', description: "'knn', 'min', or 'minprob' (default knn)." }, k: { type: 'number', description: 'k for knn (default 5).' }, fraction: { type: 'number', description: 'Fraction of column min for min (default 1.0).' }, shift: { type: 'number', description: 'MinProb down-shift in SDs (default 1.8).' }, width: { type: 'number', description: 'MinProb width in SDs (default 0.3).' }, seed: { type: 'number', description: 'MinProb RNG seed (default 0).' }, treatZeroAsMissing: { type: 'boolean', description: 'Treat 0 as missing (default true).' } },
  },
  {
    name: 'differential_abundance', category: 'Proteomics',
    description: 'Two-group Welch t-test per protein on log2 intensities + BH FDR + log2FC (B vs A). Requires numpy+scipy. Validated: recovers a spiked 4× up-regulated protein (log2FC≈+2, padj<0.05); flat protein not significant.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'differential_abundance' }),
    parameters: { groupA: { type: 'object', required: true, description: '{proteinId: [intensities...]} condition A.' }, groupB: { type: 'object', required: true, description: '{proteinId: [intensities...]} condition B.' }, alreadyLog2: { type: 'boolean', description: 'Inputs are already log2 (default false).' } },
  },
  {
    name: 'tmt_protein_rollup', category: 'Proteomics',
    description: 'TMT/iTRAQ reporter-ion PSM→protein rollup (median or sum) with optional per-channel median normalization. Requires numpy. Validated: recovers known per-channel medians; channel-norm equalizes channel medians.',
    handler: (i) => runPythonScript('server/proteomics_tools.py', { ...i, task: 'tmt_protein_rollup' }),
    parameters: { psms: { type: 'object', required: true, description: '{proteinId: [[ch1, ch2, ...], ...]} reporter intensities (PSMs×channels).' }, method: { type: 'string', description: "'median' or 'sum' (default median)." }, normalizeChannels: { type: 'boolean', description: 'Median-normalize channels (default true).' }, channelNames: { type: 'array', description: 'Channel/tag names.' } },
  },
  // --- Spatial-transcriptomics neighborhood analysis (numpy/scipy; squidpy-style) ---
  {
    name: 'neighborhood_enrichment', category: 'Spatial transcriptomics',
    description: 'Cell-type neighborhood enrichment: kNN spatial graph + seeded label-permutation z-scores per type pair (squidpy-style). Requires numpy+scipy. Validated: segregated single-type blobs → within-type z≫0, cross-type z<0; checkerboard → cross-type z>0.',
    handler: (i) => runPythonScript('server/spatial_neighborhood.py', { ...i, task: 'neighborhood_enrichment' }),
    parameters: { coordinates: { type: 'array', required: true, description: 'N×2 or N×3 cell positions.' }, labels: { type: 'array', required: true, description: 'Cell-type label per cell (length N).' }, k: { type: 'number', description: 'kNN neighbors (default 6).' }, nPermutations: { type: 'number', description: 'Null permutations (default 1000).' }, seed: { type: 'number', description: 'RNG seed (default 0).' } },
  },
  {
    name: 'cooccurrence', category: 'Spatial transcriptomics',
    description: 'Cell-type co-occurrence across distance bins: P(type=t within distance d of a center type) / P(type=t). Requires numpy+scipy. Validated: clustered same-type ratio>1 at short distance; segregated cross-type ratio<1.',
    handler: (i) => runPythonScript('server/spatial_neighborhood.py', { ...i, task: 'cooccurrence' }),
    parameters: { coordinates: { type: 'array', required: true, description: 'N×2 or N×3 cell positions.' }, labels: { type: 'array', required: true, description: 'Cell-type label per cell.' }, nBins: { type: 'number', description: 'Distance bins (default 10).' }, maxDistance: { type: 'number', description: 'Max distance (default half the coordinate diagonal).' }, distanceBins: { type: 'array', description: 'Explicit strictly-increasing bin edges (alt to nBins).' } },
  },
  {
    name: 'infiltration_score', category: 'Spatial transcriptomics',
    description: 'Fraction of target cells within a radius of any source cell (e.g. immune infiltration into tumor) + mean source contacts per target. Requires numpy+scipy. Validated: recovers a known 5/10 = 0.5 fraction.',
    handler: (i) => runPythonScript('server/spatial_neighborhood.py', { ...i, task: 'infiltration_score' }),
    parameters: { coordinates: { type: 'array', required: true, description: 'N×2 or N×3 cell positions.' }, labels: { type: 'array', required: true, description: 'Cell-type label per cell.' }, source: { type: 'string', required: true, description: 'Source cell-type label.' }, target: { type: 'string', required: true, description: 'Target cell-type label.' }, radius: { type: 'number', required: true, description: 'Infiltration/contact radius.' } },
  },
  {
    name: 'neighbor_composition', category: 'Spatial transcriptomics',
    description: 'Per-cell-type average neighbor-type composition over each cell\'s k nearest neighbors (homotypic vs heterotypic neighborhoods). Requires numpy+scipy. Validated: segregated field → self-fraction ~1.0.',
    handler: (i) => runPythonScript('server/spatial_neighborhood.py', { ...i, task: 'neighbor_composition' }),
    parameters: { coordinates: { type: 'array', required: true, description: 'N×2 or N×3 cell positions.' }, labels: { type: 'array', required: true, description: 'Cell-type label per cell.' }, k: { type: 'number', description: 'kNN neighbors (default 6).' } },
  },
  // --- ADMET / med-chem drug-likeness (RDKit; real descriptors, no fabricated predictions) ---
  {
    name: 'admet_profile', category: 'Drug discovery',
    description: '13-descriptor ADMET/physicochemical panel from SMILES (MW, logP, TPSA, HBD/HBA, rotatable bonds, aromatic rings, FractionCsp3, molar refractivity, heavy atoms, formal charge, rings, QED) via RDKit. Batch-safe; invalid SMILES → honest error entry. Validated: aspirin QED=0.55, TPSA=63.6.',
    handler: (i) => runPythonScript('server/admet_tools.py', { ...i, task: 'admet_profile' }),
    parameters: { smiles: { type: 'string', description: 'Single molecule SMILES.' }, smilesList: { type: 'array', description: 'Batch of SMILES strings (alt to smiles).' } },
  },
  {
    name: 'druglikeness_rules', category: 'Drug discovery',
    description: 'Apply Lipinski Ro5, Veber, Ghose, Egan and Muegge drug-likeness rule sets to a SMILES; per-rule pass/fail + exact violated criteria (RDKit). Validated: aspirin passes Lipinski with 0 violations.',
    handler: (i) => runPythonScript('server/admet_tools.py', { ...i, task: 'druglikeness_rules' }),
    parameters: { smiles: { type: 'string', required: true, description: 'Molecule SMILES.' } },
  },
  {
    name: 'synthetic_accessibility', category: 'Drug discovery',
    description: 'Ertl synthetic-accessibility (SA) score (1=easy … 10=hard) via the RDKit SA_Score contrib + interpretation. Real published algorithm. Validated: ethanol SA≈1.98 (easy).',
    handler: (i) => runPythonScript('server/admet_tools.py', { ...i, task: 'synthetic_accessibility' }),
    parameters: { smiles: { type: 'string', required: true, description: 'Molecule SMILES.' } },
  },
  {
    name: 'structural_alerts', category: 'Drug discovery',
    description: 'PAINS + BRENK + NIH structural-alert / toxicophore screen via RDKit FilterCatalog → matched alerts with descriptions. Validated: catechol flags ≥1 alert; clean molecules return 0.',
    handler: (i) => runPythonScript('server/admet_tools.py', { ...i, task: 'structural_alerts' }),
    parameters: { smiles: { type: 'string', required: true, description: 'Molecule SMILES.' } },
  },
  // --- Drug repurposing (CMap signature + chemical similarity; real algorithms) ---
  {
    name: 'connectivity_score', category: 'Drug repurposing',
    description: 'CMap connectivity score (Lamb et al. 2006): weighted-KS enrichment of query up/down gene sets against a drug reference signature. Negative = drug REVERSES the disease signature (repurposing candidate). Requires numpy. Validated: sign flips +0.965/−0.965 on mimic vs reversed signatures.',
    handler: (i) => runPythonScript('server/drug_repurposing.py', { ...i, task: 'connectivity_score' }),
    parameters: { upGenes: { type: 'array', required: true, description: 'Query up-regulated gene IDs.' }, downGenes: { type: 'array', required: true, description: 'Query down-regulated gene IDs.' }, referenceSignature: { type: 'object', required: true, description: '{gene: score} drug differential-expression signature.' } },
  },
  {
    name: 'signature_reversal_screen', category: 'Drug repurposing',
    description: 'Rank a library of drug signatures by how strongly they reverse a disease signature (reversalScore = −Spearman ρ over shared genes). Requires numpy+scipy. Validated: exact-negation drug ranks #1 (reversalScore≈+1), identical drug last (≈−1).',
    handler: (i) => runPythonScript('server/drug_repurposing.py', { ...i, task: 'signature_reversal_screen' }),
    parameters: { diseaseSignature: { type: 'object', required: true, description: '{gene: log2fc} disease signature.' }, drugSignatures: { type: 'object', required: true, description: '{drugName: {gene: log2fc}} library.' } },
  },
  {
    name: 'target_based_repurposing', category: 'Drug repurposing',
    description: 'Guilt-by-association repurposing: rank a library by ECFP4 Tanimoto to a query drug; echoes each hit\'s known indication/target (never invents them). Requires RDKit. Validated: self-match Tanimoto=1.0 ranks first; dissimilar molecules excluded.',
    handler: (i) => runPythonScript('server/drug_repurposing.py', { ...i, task: 'target_based_repurposing' }),
    parameters: { querySmiles: { type: 'string', required: true, description: 'Query drug SMILES.' }, library: { type: 'array', required: true, description: '[{name, smiles, indication?, target?}] reference drugs.' }, threshold: { type: 'number', description: 'Min Tanimoto (default 0.3).' }, topN: { type: 'number', description: 'Max hits (default 10).' } },
  },
  // --- Ligand-based virtual screening (RDKit; real fingerprints/scaffolds) ---
  {
    name: 'similarity_screen', category: 'Drug discovery',
    description: 'Ligand-based virtual screen: ECFP Morgan Tanimoto of a query vs a compound library → ranked hits above threshold (RDKit). Validated: aspirin self-match Tanimoto=1.0; dissimilar decane excluded at 0.3.',
    handler: (i) => runPythonScript('server/chem_screening.py', { ...i, task: 'similarity_screen' }),
    parameters: { querySmiles: { type: 'string', required: true, description: 'Query SMILES.' }, library: { type: 'array', required: true, description: '[{name, smiles}] compound library.' }, threshold: { type: 'number', description: 'Min Tanimoto (default 0.3).' }, topN: { type: 'number', description: 'Max hits (default 20).' }, radius: { type: 'number', description: 'Morgan radius (default 2).' }, nBits: { type: 'number', description: 'Fingerprint bits (default 2048).' } },
  },
  {
    name: 'pharmacophore_profile', category: 'Drug discovery',
    description: 'RDKit pharmacophore feature profile (Donor/Acceptor/Aromatic/Hydrophobe/PosIonizable/NegIonizable counts + feature list). Validated: phenol → ≥1 aromatic and ≥1 donor.',
    handler: (i) => runPythonScript('server/chem_screening.py', { ...i, task: 'pharmacophore_profile' }),
    parameters: { smiles: { type: 'string', description: 'Single molecule SMILES.' }, smilesList: { type: 'array', description: 'Batch of SMILES (alt to smiles).' } },
  },
  {
    name: 'scaffold_clustering', category: 'Drug discovery',
    description: 'Cluster a compound library by shared Bemis–Murcko scaffold (RDKit) → scaffold groups sorted by size. Validated: benzene/toluene/phenol collapse to one c1ccccc1 cluster of size 3.',
    handler: (i) => runPythonScript('server/chem_screening.py', { ...i, task: 'scaffold_clustering' }),
    parameters: { molecules: { type: 'array', required: true, description: '[{name, smiles}] library.' } },
  },
  {
    name: 'diversity_selection', category: 'Drug discovery',
    description: 'MaxMin diversity picker over Morgan fingerprints → a maximally diverse subset + mean pairwise Tanimoto (RDKit, seeded/deterministic). Validated: fixed seed reproduces the same selection.',
    handler: (i) => runPythonScript('server/chem_screening.py', { ...i, task: 'diversity_selection' }),
    parameters: { molecules: { type: 'array', required: true, description: '[{name, smiles}] library.' }, nPick: { type: 'number', description: 'How many to select (default 5).' }, seed: { type: 'number', description: 'RNG seed (default 42).' } },
  },
  // --- Bayesian optimal experimental design / active learning (numpy/scipy) ---
  {
    name: 'bayesian_optimal_design', category: 'Experimental design',
    description: 'Rank candidate next-experiments by expected information gain (EIG) under a Bayesian linear model — the decision layer of a closed-loop self-driving lab. Selects WHICH experiment to run, never predicts its outcome. Requires numpy. Validated: an unexplored design direction beats a redundant one.',
    handler: (i) => runPythonScript('server/experimental_design.py', { ...i, task: 'bayesian_optimal_design' }),
    parameters: { candidatePool: { type: 'array', required: true, description: 'Candidates × features design rows.' }, designMatrix: { type: 'array', description: 'Already-run experiments (experiments × features).' }, noiseVariance: { type: 'number', description: 'Observation noise σ² (default 1.0).' }, priorPrecision: { type: 'number', description: 'Gaussian prior precision τ (default 1.0).' } },
  },
  {
    name: 'sequential_active_learning', category: 'Experimental design',
    description: 'Greedily choose a batch of experiments by iterative max-EIG, updating the posterior after each pick (Sherman–Morrison). Requires numpy. Validated: per-step EIG diminishes, cumulative EIG is monotone.',
    handler: (i) => runPythonScript('server/experimental_design.py', { ...i, task: 'sequential_active_learning' }),
    parameters: { candidatePool: { type: 'array', required: true, description: 'Candidates × features.' }, nBatch: { type: 'number', description: 'How many to select (default min(3, pool)).' }, designMatrix: { type: 'array', description: 'Already-run experiments.' }, noiseVariance: { type: 'number', description: 'Observation noise σ² (default 1.0).' }, priorPrecision: { type: 'number', description: 'Prior precision τ (default 1.0).' } },
  },
  {
    name: 'd_optimal_selection', category: 'Experimental design',
    description: 'Greedy D-optimal subset selection: choose k rows maximizing log det of the information matrix (minimizes the parameter confidence-ellipsoid volume). Requires numpy. Validated: log det increases; picks the orthogonal (independent) direction.',
    handler: (i) => runPythonScript('server/experimental_design.py', { ...i, task: 'd_optimal_selection' }),
    parameters: { candidatePool: { type: 'array', required: true, description: 'Candidates × features.' }, k: { type: 'number', required: true, description: 'Number of experiments to select.' }, noiseVariance: { type: 'number', description: 'Observation noise σ² (default 1.0).' }, priorPrecision: { type: 'number', description: 'Prior precision τ (default 1.0).' } },
  },
  {
    name: 'space_filling_design', category: 'Experimental design',
    description: 'Maximin Latin-Hypercube space-filling design over given bounds for initial screening (scipy.stats.qmc), seeded/reproducible. Requires scipy. Validated: points in-bounds, reproducible, positive spread.',
    handler: (i) => runPythonScript('server/experimental_design.py', { ...i, task: 'space_filling_design' }),
    parameters: { nPoints: { type: 'number', required: true, description: 'Number of design points.' }, bounds: { type: 'array', required: true, description: '[[low, high], ...] per design dimension.' }, seed: { type: 'number', description: 'RNG seed (default 42).' } },
  },
  // --- Active-learning loop (stateful self-driving-lab brain; numpy) ---
  {
    name: 'propose_next_experiment', category: 'Active learning',
    description: 'Fit a Bayesian linear model on measured experiments (X, y), report coefficient estimates + uncertainty, and select the next experiment by expected information gain. HALTS for a real measurement — never fabricates an assay outcome. Requires numpy. Validated: recovers true coefficients; awaitingMeasurement=true.',
    handler: (i) => runPythonScript('server/active_learning_loop.py', { ...i, task: 'propose_next_experiment' }),
    parameters: { designMatrix: { type: 'array', required: true, description: 'Experiments × features already run.' }, observations: { type: 'array', required: true, description: 'Real measured response per experiment.' }, candidatePool: { type: 'array', required: true, description: 'Candidate next-experiments × features.' }, noiseVariance: { type: 'number', description: 'Observation noise σ² (default 1.0).' }, priorPrecision: { type: 'number', description: 'Prior precision τ (default 1.0).' } },
  },
  {
    name: 'assimilate_measurement', category: 'Active learning',
    description: 'Append a newly MEASURED (x, y) to the data, refit the Bayesian model, and report realized information gain + posterior-variance reduction. The measured response must be real (caller-supplied). Requires numpy. Validated: fresh direction yields more info gain than a redundant one; trace shrinks.',
    handler: (i) => runPythonScript('server/active_learning_loop.py', { ...i, task: 'assimilate_measurement' }),
    parameters: { designMatrix: { type: 'array', required: true, description: 'Prior experiments × features.' }, observations: { type: 'array', required: true, description: 'Prior measured responses.' }, newDesign: { type: 'array', required: true, description: 'Design row of the experiment just run.' }, newObservation: { type: 'number', required: true, description: 'Its REAL measured response.' }, noiseVariance: { type: 'number', description: 'Observation noise σ² (default 1.0).' }, priorPrecision: { type: 'number', description: 'Prior precision τ (default 1.0).' } },
  },
  {
    name: 'loop_convergence', category: 'Active learning',
    description: 'Stop criterion for the active-learning loop: report whether the maximum predictive uncertainty over a candidate pool has fallen below a tolerance. Requires numpy. Validated: converged under a large tolerance, not converged under a tiny one.',
    handler: (i) => runPythonScript('server/active_learning_loop.py', { ...i, task: 'loop_convergence' }),
    parameters: { designMatrix: { type: 'array', required: true, description: 'Experiments × features.' }, observations: { type: 'array', required: true, description: 'Measured responses.' }, candidatePool: { type: 'array', required: true, description: 'Candidates × features.' }, tolerance: { type: 'number', required: true, description: 'Max acceptable predictive std.' }, noiseVariance: { type: 'number', description: 'Observation noise σ² (default 1.0).' }, priorPrecision: { type: 'number', description: 'Prior precision τ (default 1.0).' } },
  },
  // --- Federated meta-analysis (privacy-preserving cross-site stats; numpy/scipy) ---
  {
    name: 'federated_ttest', category: 'Federated analysis',
    description: 'Pooled two-group Welch t-test from per-site (n, mean, variance) sufficient statistics — raw rows never leave a site. Requires numpy+scipy. Validated: result EXACTLY equals the t-test on the concatenated raw data (to 1e-9).',
    handler: (i) => runPythonScript('server/federated_meta.py', { ...i, task: 'federated_ttest' }),
    parameters: { sites: { type: 'array', required: true, description: '[{nA, meanA, varA, nB, meanB, varB}, ...] per-site sufficient statistics.' } },
  },
  {
    name: 'stouffer_meta', category: 'Federated analysis',
    description: 'Combine per-site z-scores into one Z + p (weighted Stouffer method). Requires numpy+scipy. Validated: four z=2 → combined Z=4.0.',
    handler: (i) => runPythonScript('server/federated_meta.py', { ...i, task: 'stouffer_meta' }),
    parameters: { zScores: { type: 'array', required: true, description: 'Per-site z-scores.' }, weights: { type: 'array', description: 'Optional positive per-site weights (e.g. √n).' } },
  },
  {
    name: 'fisher_meta', category: 'Federated analysis',
    description: "Combine per-site p-values via Fisher's method (−2·Σ ln p ~ χ²₂ₖ). Requires numpy+scipy. Validated: matches scipy.combine_pvalues exactly.",
    handler: (i) => runPythonScript('server/federated_meta.py', { ...i, task: 'fisher_meta' }),
    parameters: { pValues: { type: 'array', required: true, description: 'Per-site p-values in (0, 1].' } },
  },
  {
    name: 'random_effects_meta', category: 'Federated analysis',
    description: 'DerSimonian–Laird random-effects meta-analysis of per-site effect sizes + standard errors → pooled effect, 95% CI, Q, I², τ². Requires numpy+scipy. Validated: identical sites → τ²=0/I²=0/pooled=effect with a tighter CI.',
    handler: (i) => runPythonScript('server/federated_meta.py', { ...i, task: 'random_effects_meta' }),
    parameters: { effects: { type: 'array', required: true, description: 'Per-site effect sizes.' }, standardErrors: { type: 'array', required: true, description: 'Per-site standard errors (>0).' } },
  },
  // --- QSAR / QSPR modeling (RDKit descriptors + scikit-learn; real trained models) ---
  {
    name: 'descriptor_matrix', category: 'QSAR modeling',
    description: 'Compute the RDKit physicochemical descriptor table (MW, logP, TPSA, HBD/HBA, rotatable bonds, aromatic rings, FractionCsp3, MR, heavy atoms, rings) for a set of SMILES. Requires RDKit. Validated: MW descriptor matches RDKit; invalid SMILES flagged.',
    handler: (i) => runPythonScript('server/qsar_modeling.py', { ...i, task: 'descriptor_matrix' }),
    parameters: { molecules: { type: 'array', required: true, description: '[{smiles}] molecules.' } },
  },
  {
    name: 'qsar_cross_validate', category: 'QSAR modeling',
    description: 'k-fold cross-validated QSAR/QSPR performance (R²/RMSE/MAE) of a ridge or random-forest model on labeled molecules — the honest reliability estimate. Requires RDKit+scikit-learn. Validated: learnable endpoint → R²>0.9; pure noise → R²<0.5.',
    handler: (i) => runPythonScript('server/qsar_modeling.py', { ...i, task: 'qsar_cross_validate' }),
    parameters: { molecules: { type: 'array', required: true, description: '[{smiles, y}] with REAL measured endpoint y.' }, model: { type: 'string', description: "'ridge' or 'rf' (default ridge)." }, cvFolds: { type: 'number', description: 'CV folds (default 5).' } },
  },
  {
    name: 'qsar_predict', category: 'QSAR modeling',
    description: 'Fit a QSAR model on labeled molecules and predict unlabeled ones, reporting the training CV R² alongside. Predictions come ONLY from the fitted model on real data — no value is fabricated. Requires RDKit+scikit-learn. Validated: returns a real number per molecule.',
    handler: (i) => runPythonScript('server/qsar_modeling.py', { ...i, task: 'qsar_predict' }),
    parameters: { trainMolecules: { type: 'array', required: true, description: '[{smiles, y}] labeled training set.' }, predictMolecules: { type: 'array', required: true, description: '[{smiles}] molecules to predict.' }, model: { type: 'string', description: "'ridge' or 'rf' (default ridge)." } },
  },
  {
    name: 'applicability_domain', category: 'QSAR modeling',
    description: 'Leverage-based applicability-domain check: flag test molecules outside the training descriptor space (predictions there are extrapolations). Requires RDKit. Validated: an out-of-space molecule is flagged out-of-domain.',
    handler: (i) => runPythonScript('server/qsar_modeling.py', { ...i, task: 'applicability_domain' }),
    parameters: { trainMolecules: { type: 'array', required: true, description: '[{smiles}] training set.' }, testMolecules: { type: 'array', required: true, description: '[{smiles}] molecules to check.' } },
  },
  // --- Structured knowledge-base logic compiler (Z3; formal proofs, no LLM extraction) ---
  {
    name: 'compile_constraints', category: 'Knowledge logic',
    description: 'Compile explicit structured relationships (activates/inhibits/requires, optional conditions) into Z3 constraints and check the knowledge base is internally satisfiable. Requires z3-solver. No NL extraction — verdicts are formal SMT proofs. Validated: self-consistent KB → SAT.',
    handler: (i) => runPythonScript('server/knowledge_logic.py', { ...i, task: 'compile_constraints' }),
    parameters: { relationships: { type: 'array', required: true, description: '[{source, relation, target, when?, id?}] structured relationships.' } },
  },
  {
    name: 'check_consistency', category: 'Knowledge logic',
    description: 'Is (knowledge base ∧ observed node states) satisfiable? Returns a satisfying assignment when SAT or the exact minimal UNSAT core when not (Z3). Requires z3-solver. Validated: A→B with A=1,B=0 → INCONSISTENT with a core naming that rule + observation.',
    handler: (i) => runPythonScript('server/knowledge_logic.py', { ...i, task: 'check_consistency' }),
    parameters: { relationships: { type: 'array', required: true, description: 'Structured relationships (knowledge base).' }, observations: { type: 'object', required: true, description: 'Measured node states, {node: 0/1} or [{node, state}].' } },
  },
  {
    name: 'detect_novel_discovery', category: 'Knowledge logic',
    description: 'Treat experimental observations as trusted and find the minimal set of literature relationships they contradict (Z3 UNSAT core) — candidate novel findings, not errors. Proves removing exactly that set restores consistency. Requires z3-solver. Validated: A→B→C with A=1,B=1,C=0 flags exactly "B activates C".',
    handler: (i) => runPythonScript('server/knowledge_logic.py', { ...i, task: 'detect_novel_discovery' }),
    parameters: { relationships: { type: 'array', required: true, description: 'Literature knowledge base.' }, observations: { type: 'object', required: true, description: 'Trusted experimental node states.' } },
  },
];

const BY_NAME = new Map(TOOL_REGISTRY.map((t) => [t.name, t]));
// Also allow calling tools by their raw engine command for flexibility.
const BY_COMMAND = new Map(TOOL_REGISTRY.map((t) => [t.engineCommand, t]));

export function getTool(name: string): ToolSpec | undefined {
  return BY_NAME.get(name) || BY_COMMAND.get(name);
}

export interface ToolInvocation {
  tool: string;
  ok: boolean;
  result?: any;
  error?: string;
}

/** Invoke a registry tool with real computation. Unknown tools return an honest error, never a crash. */
export async function invokeTool(name: string, input: any): Promise<ToolInvocation> {
  const spec = getTool(name);
  if (!spec) {
    return { tool: name, ok: false, error: `Unknown tool '${name}'. Known tools: ${TOOL_REGISTRY.map((t) => t.name).join(', ')}.` };
  }
  const missing = Object.entries(spec.parameters)
    .filter(([, p]) => p.required)
    .map(([k]) => k)
    .filter((k) => input?.[k] === undefined || input?.[k] === null);
  if (missing.length) {
    return { tool: spec.name, ok: false, error: `Missing required parameter(s): ${missing.join(', ')}.` };
  }
  try {
    // JS-native tools (external DB clients) run their handler; engine tools spawn Python.
    if (spec.handler) {
      const result = await spec.handler(input);
      // External DB results carry an explicit status; an unavailable/not_found
      // upstream is an honest tool failure, not a crash.
      if (result && typeof result === 'object' && 'status' in result && result.status !== 'success') {
        return { tool: spec.name, ok: false, error: String(result.error || result.status), result };
      }
      return { tool: spec.name, ok: true, result };
    }
    if (!spec.engineCommand) {
      return { tool: spec.name, ok: false, error: `Tool '${spec.name}' has no engine command or handler.` };
    }
    const result = await runEngine(spec.engineCommand, input);
    if (result && typeof result === 'object' && 'error' in result && result.error) {
      return { tool: spec.name, ok: false, error: String(result.error), result };
    }
    return { tool: spec.name, ok: true, result };
  } catch (err: any) {
    return { tool: spec.name, ok: false, error: err?.message || String(err) };
  }
}

/** Tool schemas in a shape suitable for LLM function-calling / planning prompts. */
export function toolSchemasForLLM() {
  return TOOL_REGISTRY.map((t) => ({
    name: t.name,
    category: t.category,
    description: t.description,
    parameters: t.parameters,
  }));
}
