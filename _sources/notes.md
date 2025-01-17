# Comp 605 Notes for Spring 2025


- Participation (can include engagement in class, attendance, use of office hours, etc) (5%)
- Assignment 1 (5%): due Friday, February 7, by midnight (AOE): Git (expand on more advanced features for former COMP526 students)
- Assignment 2 (10%): due Friday, February 21, by midnight (AOE): Experiments in Vectorization (Jed)
- Assignment 3 (10%): due Friday, March 07, by midnight (AOE): Using Makefiles and Parallel Sorting, performance and Scaling (Jed)
- Midterm Part #1: due Friday, March 14, by midnight (AOE): Project Proposal
- Midterm Part #2: due Friday, March 28, by midnight (AOE): Open Issue
- Assignment 4 (10%): due Friday, April 11, by midnight (AOE): reduction using MPI (perhaps in Julia)
- Assignment 5 (10%): due Friday, April 25, by midnight (AOE): GPU (perhaps in Julia - HW 2 from Jeremy)

- Midterm Project: (%15) Proposal + Open Issue
- Final Project: (%35)

## Git assignment:

1. Make two commits, with the following commit messages "Commit 1" and "Commit 2", respectively.
2. Create a new branch, called newImage (which now refers to Commit 2) and commit again with the following commit message "Commit 3". Oh no! The main branch moved but the newImage branch didn't! That's because we weren't "on" the new branch, which is why the asterisk (*) was on main.
3. Let's tell git we want to checkout the branch with `git checkout newImage`
This will put us on the new branch before committing our changes:
`git commit`.
4. git merge: Make a new branch called `bugFix`. Checkout the `bugFix` branch with `git checkout bugFix`. Commit once. Go back to main with `git checkout`. Commit another time.
Merge the branch `bugFix` into `main` with `git merge`. (solution: `git checkout -b bugFix`; `git commit`; `git checkout main`; `git commit`; `git merge bugFix`). Inspect the sequence with `git log` or `git reflog`
5. git rebase: Checkout a new branch named `bugFix`; Commit once; Go back to `main` and commit again; Check out `bugFix` again and rebase onto `main`.

EC (for all):

HEAD is the symbolic name for the currently checked out commit -- it's essentially what commit you're working on top of.

HEAD always points to the most recent commit which is reflected in the working tree. Most git commands which make changes to the working tree will start by changing HEAD.

Normally HEAD points to a branch name (like bugFix). When you commit, the status of bugFix is altered and this change is visible through HEAD.

"Detaching HEAD" just means attaching it to a commit instead of a branch.

Or better, reset/revert:

`git reset` reverses changes by moving a branch reference backwards in time to an older commit. In this sense you can think of it as "rewriting history;" git reset will move a branch backwards as if the commit had never been made in the first place.

Let's see what that looks like: `git reset HEAD~1`.

While resetting works great for local branches on your own machine, its method of "rewriting history" doesn't work for remote branches that others are using.

In order to reverse changes and share those reversed changes with others, we need to use git revert. Let's see it in action:
`git revert HEAD`.
1. Start from a branch called `local`. Commit. Erase the last commit by using `reset` (solution: Use `git reset HEAD~1`). Checkout a branch called `remote` and erase the last commit there (solution: `git revert HEAD`).
