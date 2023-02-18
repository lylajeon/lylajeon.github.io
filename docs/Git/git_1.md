---
title : How to use Git (1)
layout: default
parent: Git
nav_order: 1
---
date: 2022-10-11

This is the content I learned from Git lecture done by egoing. ([seomal link](https://seomal.com/))



A directory (folder) consists of working directory, stage area, and repository. 



Initialize git repository: `git init`

Add changes to stage area : `git add (file name)`

Store changes in stage area : `git commit -m (commit message)`



Commit ID has the information of parent commit and current commit. 

Parent commit id points to where HEAD points.



When we want to move where HEAD points, which means we want to change the current working directory to the past state: `git checkout (commit ID)` can be used. 

main branch points to last work we did. 

`git log --oneline --all` : shows all the logs regardless of where HEAD pointer points to. 

When we want to return to last work we did, `git checkout main` can be used. Then, HEAD points to main.

This state refers to *attached head state*. When head doesn't point to main, this state refers to *detached head state*.



When `git checkout main` is executed in detached HEAD state, the commits made when the directory was in detached HEAD state is gone away. 

In this situation, `git reflog ` can be used to figure out what the commit id was, when in detached HEAD state. 



Useful Git plugin in VScode:

- Git Graph
- Git Lens



In detached head state, a new branch can be made. Let's call this branch exp. When all the experiments are  done with exp branch, the main branch merges exp branch. 



When we want to move HEAD, `git checkout (commit id)` is used. 

When we want to move branch, `git reset --(option)` is used. 



How to make new branch : `git branch (branch name)`.

To change where HEAD points to new exp branch, `git checkout exp`.



To merge exp branch from main branch, go to main branch and then merge the branch. 

`git checkout main`, `git merge exp`