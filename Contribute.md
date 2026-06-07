# Contributing Guide  MergeDeployGraduate

## Team Members
| Member | Role | Responsibilities |
|---|---|---|
| Musaddiq | Pipeline A + API + GCP | LR models, FastAPI, GCP deployment |
| Nishanth | Docker + CI/CD + Cloud Run | Docker builds, GitHub Actions, Cloud Run |
| Israail | Pipeline B +CI | Ensemble models, Gradio/Streamlit |
| Lohith | Testing + UI | pytest, pre-commit, GitHub Actions CI |

---

## CI/CD Requirements

### Before Pushing
All code must pass these checks locally:

```bash
# Run linter
ruff check src/

# Run formatter
ruff format src/

# Run type checker
mypy src/

# Run tests
pytest tests/ -v

# Run pre-commit hooks
pre-commit run --all-files
```

### GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Every push/PR | Linting, formatting, type checking, tests |
| `cml.yml` | Every push/PR | Trains model, posts metrics as PR comment |
| `docker-build.yml` | Push to main | Builds and pushes Docker image to Docker Hub |

All workflows must pass before merging to `main`.

---

## Testing Requirements for PRs

### Required Tests
- Unit tests for any new functions in `src/`
- Integration tests for pipeline changes
- Minimum 80% code coverage for new code

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/mlops_frauddetection --cov-report=term-missing

# Run specific test file
pytest tests/test_features.py -v
```

### Test Structure
```
tests/
 test_features.py      # Feature engineering tests
 test_model.py         # Model training tests
 test_api.py           # FastAPI endpoint tests
 conftest.py           # Shared fixtures
```

---

## Deployment Process

### Local Development
```bash
# Activate venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run FastAPI locally
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run training
python -m mlops_frauddetection.train_model training.pipeline=lr
```

### GCP Deployment (Musaddiq)
```bash
# Push to Artifact Registry
bash scripts/push_to_artifact_registry.sh

# Submit training job
bash scripts/gcp_train.sh

# Deploy Cloud Functions
bash scripts/deploy_cloud_function.sh
```

### Docker (Nishanth)
```bash
# Build image
docker build -f dockerfiles/Dockerfile -t fraud-detection-mlops .

# Push to Docker Hub
docker push nshastry00/fraud-detection-mlops:latest
```

---

## Branch Strategy

```
main            protected, requires PR review
 musaddiq/*  Musaddiq's feature branches
 nishanth/*  Nishanth's feature branches
 rghazzal/*   Israail's feature branches
 lohith/*    Lohith's feature branches
```

### Naming Convention
```
musaddiq/phase3-deployment
nishanth/docker-ci-update
rghazzal/rghazzal-CI-testing
lohith/pytest-coverage
```

---

## PR Guidelines

1. Create branch from latest `main`
2. Make changes
3. Run all CI checks locally
4. Push branch
5. Create PR with clear title and description
6. Request review from at least one teammate
7. Wait for CI to pass
8. Merge after approval

### PR Title Format
```
[Phase X] Brief description of changes
```

### PR Description Template
```markdown
## What changed
- Brief description of changes

## Why
- Reason for the change
```

---

## Secret Management

Never commit credentials. All secrets go in GitHub Secrets:

| Secret | Used by | Purpose |
|---|---|---|
| `DOCKER_USERNAME` | docker-build.yml | Docker Hub login |
| `DOCKER_PASSWORD` | docker-build.yml | Docker Hub login |
| `GDRIVE_CREDENTIALS` | ci.yml, cml.yml | DVC data pull |
| `GCP_CREDENTIALS` | docker-build.yml | GCP Artifact Registry |
| `GCP_PROJECT_ID` | docker-build.yml | GCP project |
| `GCP_REGION` | docker-build.yml | GCP region |

---

## DVC Data Management

```bash
# Pull data
dvc pull

# Add new model files
dvc add models/new_model.joblib

# Push to remote
dvc push

# Commit DVC pointer
git add models/new_model.joblib.dvc
git commit -m "Add new model DVC pointer"
```

DVC remote: Google Drive (configured in `.dvc/config`)