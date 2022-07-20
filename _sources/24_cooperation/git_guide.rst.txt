Git Usage
~~~~~~~~~~~~~~~~

In this section, we will introduce a list of commonly used knowledge in actual development.


1. Basic Concepts and Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Concept** of original code, workspace, stage, local repository, remote repository, and commands for **interconversion between them**:

.. image:: ./images/git_command1.png
    :scale: 33%
    :align: center

Here are some commands that are not shown in the figure above, but are also very commonly used:

**git stash command**\: If you are developing branch A, and suddenly there is a job that requires to be switched to branch B, and branch A has not yet reached the point where a commit can be submitted. At this time you can use \ ``git stash``\ to firstly save the changes to branch A (If you want to leave some information like a commit, you can use \ ``git stash save "STASH-MESSAGE"``\ ).

Then you can switch to branch B via ``git checkout xxx`` command (an error will be reported if \ ``git stash``\ or \ ``git commit``\ are not executed). After the development on branch B is completed, you can switch back to branch A, and then use \ ``git stash pop``\ to restore the temporary content.

Stash is a stack. If you need to pop a non-top element, you can use \ ``git stash list``\ to view all stash records, and then use the command \ ``git stash pop stash@{0}``\ , where 0 can be replaced with any existing stash record number.


.. image:: ./images/git_command2_stash.png
    :scale: 33%
    :align: center

**git log command**\: Can display the commit information, please refer to \ `tutorial <https://www.yiibai.com/git/git_log.html>`__

**git cherry-pick command**\: You can apply a commit to another branch. The difference from ``git merge`` is that git merge merges the entire branch into other branches. While git cherry-pick will only apply a commit to other branches, please refer to \ `tutorial <https://ruanyifeng.com/blog/2020/04/git-cherry-pick.html>`_


2. Git Commit Specification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have some rules for git commit:

1. As far as possible, each independent function corresponds to a commit

2. | Template: feature/fix/polish/test/style(commiter_name or project_name):
      commit message
   | Example: fix(zlx): add tb logger in naive buffer



3. Take an Actual Development As an Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take an actual development process as an example to explain the git commands that need to be used.

1. Clone the repository: \ ``git clone REPO-URL``\ ;

2. Checkout to your own development branch: \ ``git checkout -b YOUR-BRANCH-NAME``

3. Develop your branch. After one part is completed, include the content you want to submit with command \ ``git add``\ (here you can use \ ``git status``\ to view all the changes, if you want to include all the changes file, you can use the \ ``git add -u``\ command); then submit a commit with \ ``git commit -m COMMIT-MESSAGE``\ .

4. Push the local repository to the remote repository: \ ``git push origin YOUR-BRANCH-NAME``\ . If it is the first push, it will show that the remote repo has no branch associated with it, and you can modify the command according to the hint, usually \ ``git push --set-upstream origin YOUR-BRANCH-NAME``\ . If someone else has modified the code and pushed it to the remote repository before you submit this modification, it will prompt that there is a conflict. You need to use the \ ``git pull origin YOUR-BRANCH-NAME --rebase``\ command to pull the latest code. And resolve the conflict between the latest code and your own code (if any) before you can push.

5. Branch rebase command: \ ``git rebase BRANCH-NAME``\ . In our development process, it must be guaranteed that the branch can be merged into main (or source branch) without conflicts. Therefore, the rebase command is often used in the following scenarios: Commiter A and B checkout branches C and D respectively from the main branch for development, then committer A completed the C branch and merged it into the main branch. At the end of the development, committer B must make sure the latest main branch is pulled to the local, then use \ ``git rebase main``\ and resolve all conflicts before the development of this branch is over.


4. .gitignore File
^^^^^^^^^^^^^^^^^^^^^

In our local development path, there are many files that we do not want to submit to the remote repo, such as the local configuration information of the project, pycache, log files, checkpoints, and so on. At this time, using the .gitignore file can ignore these files by string matching.

.gitignore file is commonly as follows:


.. code:: 

   # Ignore the specified file
   HelloWrold.class
   # Ignore the specified folder
   pkg/
   __pycache__/
   # * is a wildcard and can match any string
   # Ignore all .jpg files
   *.jpg
   # Ignore folders with 'ignore' at the end of the name
   *ignore/
   # Ignore folders with 'ignore' in the name
   *ignore*/

If you really want to submit some files (such as images) that are blocked by the .gitignore rule, you can use \ ``git push -f PATH``\ to upload. But please make sure that this file is necessary.

