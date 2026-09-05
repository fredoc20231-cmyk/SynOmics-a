# SynOmics — Deployment & Infrastructure

Production-deployment scaffolding for the Advanced Bioinformatics Platform
("Aistudio Core"). This document is written under the repository's **ZERO
HALLUCINATION MANDATE**: it states plainly what has been verified in this
environment and what has **not**, and never claims a path works that has not
been run.

---

## Architecture at a glance

```
                 ┌────────────────────────────────────────────┐
   Internet ───▶ │ Cloud Armor WAF  ──▶  Cloud Run (dispatcher)│
                 │   (OWASP, rate-limit)   dist/server.mjs (ESM)│
                 └───────────────┬────────────────────────────┘
                                 │ Serverless VPC connector (gRPC, internal)
                 ┌───────────────▼────────────────────────────┐
                 │           GKE private cluster                │
                 │  cpu-optimized pool   → bioconda-engine img  │
                 │  high-memory pool     → bioconda-engine img  │
                 │  GPU (A100/L4) pool   → gpu-esm-openmm img    │
                 │  gVisor sandbox pool  → gvisor-sandbox img    │
                 │            Filestore NFS scratch (RWX PVC)    │
                 └──────────────────────────────────────────────┘
     CMEK/KMS (HSM) · GCS tiered buckets · BigQuery · Secret Manager
```

The single-container path (repo-root `Dockerfile`) runs the whole app in one
image and is the simplest way to deploy. The multi-image + GKE path below is the
scale-out target that unlocks the infra-gated tools (external bioinformatics
binaries, GPU folding/MD/docking) documented as "not built — needs infra" in
`CLAUDE.md`.

---

## Files

| Path | Purpose |
| --- | --- |
| `Dockerfile` (repo root) | Canonical single-container build (`dist/server.mjs`). |
| `docker/Dockerfile.api-dispatcher` | Same app as a named image for the multi-image set. |
| `docker/Dockerfile.base-bioconda` | CPU/high-mem worker: engine + samtools/bwa/bedtools/fastp/scanpy/cobra/opentrons. |
| `docker/Dockerfile.gpu-esm-openmm` | GPU worker: CUDA + torch + fair-esm + OpenMM + AutoDock Vina (Module E target). |
| `docker/Dockerfile.sandbox-gvisor` | Hardened micro-sandbox image for untrusted code. |
| `infra/terraform/` | GCP IaC: networking + Cloud Armor, KMS/CMEK, GKE, tiered storage, Cloud Run. |
| `k8s/helm/synomics-chart/` | Helm chart: worker tiers, gVisor RuntimeClass, Filestore PVC, gRPC services. |
| `cloudbuild.yaml` | Cloud Build pipeline: **real** test gates → image build/push → Helm → Cloud Run. |

---

## Verification status (honest)

**Verified in this environment:**
- `npm run lint` (tsc --noEmit) and `npm run build` succeed; the build emits
  `dist/server.mjs` — the api-dispatcher image CMD points at that exact file.
- The bioconda image's default `CMD ["align_sequences"]` runs a **real** built-in
  pairwise alignment (verified output: 80-col alignment, 98.75% identity) — not a
  fabricated or non-existent command.
- `cloudbuild.yaml` and the non-templated Helm YAML (`Chart.yaml`, `values.yaml`,
  `runtimeclass-gvisor.yaml`) parse as valid YAML.
- Terraform module wiring was hand-reviewed: every module output referenced by the
  root config exists, and the KMS↔storage dependency is broken by `data` sources
  (no resource-level cycle).

**NOT verified here (requires a real GCP project / cluster / GPU — none available
in this sandbox):**
- `terraform init/plan/apply` — needs GCP credentials and provider downloads.
  `terraform` is not installed here; the config is syntax-reviewed only.
- `helm lint`/`helm install` — `helm` is not installed here. The templated
  manifests (`deployments-bio-workers.yaml`, `service-mesh-grpc.yaml`,
  `persistent-volume-claims.yaml`) are not validated against the k8s schema.
- `docker build` of the bioconda/GPU images — these pull large conda/CUDA/torch
  layers and external binaries over the network and were not built here.
- Cloud Build, Cloud Run deploy, and image pushes — never executed.

Do not represent any of the un-verified items as working until they have actually
been run against a real target.

---

## Known gaps (do not overclaim)

1. **No long-running gRPC worker daemon yet.** The Helm worker Deployments expose
   port `50051` and assume a persistent server, but the engine
   (`server/synomics_engine.py`) is currently a **one-shot CLI** (reads argv +
   JSON stdin, prints, exits — see `server/engine_client.ts`). As-is, a worker pod
   would run its default command once and exit. A gRPC/HTTP server wrapper around
   the engine must be built before the GKE worker tier is functional. The chart is
   forward-looking scaffold, not a turnkey deploy.
2. **GPU engine commands are not implemented.** ESMFold/AlphaFold/OpenMM/Vina
   pipelines (Module E in `CLAUDE.md`) do not exist in the engine yet. The GPU
   image installs the toolchain and its default command is an honest readiness
   probe (`torch.cuda.is_available()` etc.), not a fake analysis result.
3. **Hardcoded internal gRPC IP.** `10.102.15.50` is shared between the Terraform
   `internal_grpc_service_ip` output and the Helm gRPC ClusterIP. It is a
   placeholder that must be reconciled with real cluster addressing.
4. **Cloud Run auth.** The Terraform Cloud Run service is internal-ingress behind
   Cloud Armor; the `cloudbuild.yaml` gcloud step deploys without
   `--allow-unauthenticated`. Decide the intended auth posture before exposing the
   service publicly.

---

## Quick start (single container — the verified path)

```bash
npm ci
npm run build
docker build -t synomics:local .        # repo-root Dockerfile
docker run -p 8080:8080 synomics:local  # GET /api/health -> 200
```

## Scale-out path (GKE + Terraform — requires a real GCP project)

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # set project_id, region, CIDRs
terraform init      # downloads providers (needs network + GCP creds)
terraform plan
terraform apply
# then, once a gRPC worker daemon exists (see Known gaps #1):
helm upgrade --install synomics k8s/helm/synomics-chart \
  --set global.projectID=$PROJECT_ID
```

## Test gates

The canonical CI is `.github/workflows/ci.yml` (93 real, ground-truth-asserting
suites + `tsc` type-check + `ruff` lint gate). `cloudbuild.yaml` re-runs the fast
subset (`engine_smoke`, `agent_smoke`, `server_integration`) before building
images — real pass/fail, no fabricated verification step.

---

## Deployment readiness (this build — verified locally)

Ran here and green (single-container path is fully exercisable in this sandbox):

- `npm run lint` (`tsc --noEmit`) — clean.
- `npm run build` — emits `dist/server.mjs` (the runtime CMD target).
- `ruff check server tests` — clean.
- `python tests/engine_smoke.py` — 23/23 (real engine computations).
- `npx tsx tests/agent_smoke.ts` — 12/12 (real agent tool-use loop).
- `npx tsx tests/server_integration.ts` — 12/12 (boots `dist/server.mjs`, real
  HTTP stack incl. `GET /api/health` → 200).
- All 93 CI suites reference existing test files; the 12 most-recent capability
  waves (bioimage, cell-motility, proteomics, spatial-neighborhood, ADMET,
  drug-repurposing, chem-screening, experimental-design, active-learning,
  federated-meta, QSAR, knowledge-logic) each pass against known ground truth.
- Tool registry loads **277** typed tools; every route dispatches to a real
  Python module (`/api/synomics/*` + `/api/biomni/*` aliases).

**Deploy this now via the single-container path** (repo-root `Dockerfile`,
`INSTALL_SCIENCE_STACK=true` for the full module set). The GKE/GPU scale-out path
remains scaffold — see *Known gaps* — and unlocks only the honestly infra-gated
tools (external aligner binaries, GPU folding/MD/docking), which the app already
surfaces as explicit "requires backend" states rather than fabricating output.
Every Python dependency the 277 tools need is pinned in `requirements.txt`
(numpy, scipy, scikit-learn, statsmodels, networkx, POT, z3-solver, rdkit,
tensorly, opencv-python-headless, h5py, gseapy, biopython, Cython, jinja2,
python-docx, reportlab) — no undeclared imports.
