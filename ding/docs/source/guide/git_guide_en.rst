Git Collaboration Guide
=================================
This document aims to provide collaborative development standardization suggestions on git for developers with understandings on the basic usage of git. If you are not familiar with git, you can **in order** read `Liao Xue Feng's git Tutorials <https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/>`_ and `Pro Git v2 <https://git-scm.com/book/zh/v2>`_ . Liao's tutorial is suitable for one-hour-getting-started, and pro git is in-depth enough as the official recommended tutorial for git.

Collaboration Processg
--------------------------
1. Init your code repo. There are mainly two cases: creating a repository from scratch (e.g. for making your own wheels), or planning to modify existing code.

  - Create a new code repository: It is generally recommended to put the new code repository in their respective team (Group), such as ``XLab/Cell``, so that members of the same group have direct accesing & editing permissions. If it is developed and used by yourself, or is not ready to be released to the team, create the code repository under your own user name. For example, ``niuyazhe/my_test_project``. All new code repositories **must** be set to ``private``.
  - Modify existing codes: For modification, it is recommended to fork a new code repository with owner's username (fork is recommended for cross-group repositories), modify the above, and submit ``merge request`` on the original repository. Or you can create your own dev branch under the original repo (a new branch is recommended for the same repo group), and then submit the changes through the ``merge request`` as well. **It is forbidden** to update directly by people other than the project maintainer. The ``master`` branch is generally **forbidden** to use ``git push -f`` to forcibly overwrite and push local changes to the remote branch.
  
.. tip::
  
    When using the fork method, in order to keep the fork code base synchronized with the original code base, the original code base can be added locally to ``git remote``  and use a branch to keep track of the master branch of the original code base. Examples are as follows:

    .. code:: bash

        # after forking `group_foo/proj_bar.git` to `user_baz/proj_bar.git`
        git clone <forked_path>
        git remote add base <origin_path>  # base is user-defined name referred to origin remote
        git fetch base
        git checkout -b base-master base/master  # use `base-master` to track origin repository

1. For Local repo development, as creating the repo and cloning to local, you can start. Following are some suggestions:

   - For any major code changes or feature updates, you should use ``GitLab Issue`` to describe the specific work to be done or discuss related possible resolution with colleagues and open at the same time a new ``merge request`` with ``WIP(Work In Progress)``. Follow up your progress, and other colleagues can also comment, discuss or add code reviews to the MR.
   - If the repo is to fix a bug, it is recommended to describe the performance and triggering conditions of the bug on the ``Issue`` page to prevent other colleagues from stepping on the pit, and label this issue ``Bug Report``.
   - If you plan to add a new feature to repo, **before starting to implement**, please chat with the repo manager about your intention and reason for adding this feature as well as the implementation method you intend to use, and then make sure that adding this feature is necessary and feasible. After that, please still raise an issue in the original code repository to describe the feature to add and the specific ``TODO`` list and label the issue accordingly, for example, use ``Algorithm`` for algorithm updates.
   - In progress of the corresponding development tasks to each issue, you can commit locally according to your own work schedule but try to ensure that each commit corresponds to only one type of related code changes, or you can push half of these commits into your forked repo to avoid loss of work. However, before completing resolving a issue (or a TODO), **do not push the incomplete commit to the original repo base**.


2. After making local changes, the developer should sync with the upstream code, integrate your own local commits, and resolve conflicts. Simple conflicts can be resolved directly on the gitlab ``merge request`` page. For more complex cases, you should manually merge after the local fetched target branch changed. For further complicated situations, you can consider using ``git rebase``, (please refer to `Pro Git-Branch-Rebase <https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5 %8F%98%E5%9F%BA>`_ for details. 

3. After the conflict is resolved, the developer should descibe corresponding code changes in the MR or Issue, and later manager or a designated colleague will review the changes. After the code is merged successfully, close the corresponding GitLab Issue.

Git Standards
--------------

This section lists some points that should be paid special attention to when using git:

  - The code of the merge request submitted to the master branch must be able to **run correctly**. Specifically, developers should perform style-checks and unit tests for the repo. It is recommended to include automatic testing and continuous integration (CI) to ensure stability for the master branch;
  - When editing files in the git repo, please pay special attention to the **Line Breaks**. By default, Win sys use ``CRLF`` for line breaks, and Unix use ``LF`` for line breaks. We force all files adding to git repository to **use LF line breaks** to avoid potential sys-level bugs. With Win sys, you can use ``dos2unix`` to perform `batch processing <https://stackoverflow.com/questions/11929461/how-can-i-run-dos2unix-on-an-entire-directory>`_ the entire folder's Line breaks;
  - After the dev completed, archived branches should be deleted in remote. This is mainly to avoid the code base from branching confusion or cloning to be overly time-consuming;
  - Note to write appropriate ``.gitignore`` and do not add any files irrelevant to the implementation of the project into git, e.g. ``.DS_Store`` folder under macOS, IDE configuration folders with ``.vscode``, ``.idea``, running log files, etc.;
  - Unless necessary, **Never embed large-capacity binary files in a git repo**, such as model files, binary lib, etc. These files may cause a variety of problems. If you want to import these, use the ``git lfs``

