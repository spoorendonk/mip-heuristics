Before starting any work, check the current state and present options. Follow these steps exactly:

## Step 1: Project health check

1. If `.gitmodules` mentions `.dev-standards` but the `.dev-standards/` directory is empty (submodule not initialized), stop and ask the user. Suggest running `git submodule update --init`. Don't continue until this is resolved.

2. If CLAUDE.md is missing or minimal (fewer than 5 lines or missing build/test commands), stop and suggest running `/init` to auto-detect project details. Don't continue until this is resolved.

3. Check if copied files are out of sync:
   - Compare each file in `.claude/commands/` against the matching file in `.dev-standards/commands/`
   - Compare `.git/hooks/pre-push` against `.dev-standards/hooks/pre-push.sh`
   - Compare `.git/hooks/commit-msg` against `.dev-standards/hooks/commit-msg.sh`

   Use `diff -q` for each pair. If any differ, list the stale files and suggest running `.dev-standards/setup.sh` to update them. Don't auto-fix — setup.sh has interactive diff/overwrite prompts.

## Step 2: Check current state

1. Run `git status` and `git branch --show-current` to see where we are.
2. Run `git branch` to see all local branches.
3. For each branch other than main, check if it's stale:
   - Last commit older than 1 week, OR
   - Merge-base with main is 25+ commits behind main HEAD
   - Flag stale branches explicitly.

## Step 3: Present options

Report findings, then present choices:

- If on a feature branch: ask whether to continue here, or switch to main for new work.
- If other local branches exist: flag them (especially stale ones) and ask if any should be cleaned up with `git branch -d`.
- If on main: run `git pull` to get latest, ready for new work.

## Step 4: Branch selection

Only after the user chooses:

- To continue existing work: stay on or check out that branch.
- To start fresh: ensure we're on main, then create a new branch with format `feat/<short-description>` or `fix/<short-description>`.

For non-trivial new work, suggest entering plan mode before implementing.

Do NOT automatically check out branches, create new ones, or delete branches without asking first.
