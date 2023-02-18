---
title: How to use Git (2)
layout: default
parent: Git
nav_order: 2
---
date: 2022-10-18

This is what I learned from second lecture of egoing's Git lecture. 

[Github repo](https://github.com/lylajeon/connect-10-18)



`git diff` : to show unstaged changes 



Tracted files are files that `git add ` is done at least once. 

Untracted files are files that are not tracted. This can contain information that are sensitive. 

`git commit -a`: -a option to auto add changes in tracted files 



Meaning of `git add`

1. makes it wait for commit
2. makes untracted files to tracted files
3. informs Git that conflict is solved 



`git add .` is a bad habit. 

`.gitignore` file is used to ignore untracted files.



`git checkout` moves HEAD. `git reset` moves branch where HEAD points to (when in attached head state). If in detached head state, this operation equals to git checkout. 

Commit has parent node as what HEAD currently pointed to. 



`git branch exp` : makes exp branch

`git checkout -b exp` : makes exp branch and checkout to exp branch



` git log --oneline --all --graph` : --graph option is used to show git log like a graph.

If --all option is not used, it shows only the parent nodes of the head. If --all option is used, all parent nodes of all references are shown.



Alias can be set by using commands like `git config --global alias.l "log --oneline --all --graph"`.



When branch merges are done, changes from common ancestor node (which is called *base*) are added.

When same line is changed differently in each branches, **merge conflict** has occurred.

When merge conflict is solved manually, `git add` is used to inform Git that merge conflict is solved.



---

GitHub

There are local repository and remote repository. One of the most commonly used remote repository company is GitHub.



`git remote add origin (github address)` : adds a new remote called origin

origin/main branch keeps track of what is pushed until now. If main branch is ahead of where origin/main is, we need to push.

---



When commit message is changed by --amend, new commit which is copy of the original commit is made and the message is changed to the new version, so the new commit which head points to has a different commit id. 

 