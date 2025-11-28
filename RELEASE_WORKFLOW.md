# Release Workflow: v0.2.2

## Overview
This workflow will:
1. Commit current changes to `dev`
2. Merge `dev` into `main`
3. Update version to 0.2.2 in `pyproject.toml`
4. Tag the release as `v0.2.2`
5. Reset `dev` to point to the same commit as `main`

## Step-by-Step Commands

### Step 1: Commit current changes to dev
```bash
# Add all modified files
git add .

# Commit with a descriptive message
git commit -m "Update documentation: clarify x_data shape handling, improve docstrings, fix consistency issues"
```

### Step 2: Switch to main and merge dev
```bash
# Switch to main branch
git checkout main

# Merge dev into main (creates a merge commit)
git merge dev -m "Merge dev into main for v0.2.2 release"
```

### Step 3: Update version in pyproject.toml
```bash
# Update version from 0.2.1 to 0.2.2
# (This will be done via file edit)
```

### Step 4: Commit version change and create tag
```bash
# Add the updated pyproject.toml
git add pyproject.toml

# Commit the version change
git commit -m "Bump version to 0.2.2"

# Create the version tag
git tag -a v0.2.2 -m "Release v0.2.2: Documentation updates and consistency improvements"
```

### Step 5: Reset dev to match main
```bash
# Switch back to dev branch
git checkout dev

# Reset dev to point to the same commit as main
# This makes dev and main identical
git reset --hard main
```

## Verification Commands

After completing the workflow, verify with:
```bash
# Check that dev and main point to the same commit
git log --oneline --graph --all -5

# Verify the tag exists
git tag -l "v0.2.2"

# Check current branch
git branch --show-current
```

## Notes
- The `git reset --hard main` in step 5 will make `dev` identical to `main`
- If you have uncommitted changes when switching branches, git will warn you
- The tag `v0.2.2` will be created on the `main` branch
- If you want to push these changes, you'll need to push both branches and the tag:
  ```bash
  git push origin main
  git push origin dev
  git push origin v0.2.2
  ```

