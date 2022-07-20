Github Usage
~~~~~~~~~~~~~~~~~~~

This section will introduce issues & pull requests in Github.

1. Create from template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both issue and PR have their own templates.

Issue needs to choose a category, then make sure you have read the documentation and PRevious issues and PRs, and specify the version number, operating system, and then formally describe the issue. See below for details:

-  I have marked all applicable categories:

   -  exception-raising bug

   -  RL algorithm bug

   -  system worker bug

   -  system utils bug

   -  code design/refactor

   -  documentation request

   -  new feature request

-  I have visited the
   `readme <https://github.com/opendilab/DI-engine/blob/github-dev/README.md>`_
   and `doc <https://opendilab.github.io/DI-engine/>`_ 

-  I have searched through the `issue
   tracker <https://github.com/opendilab/DI-engine/issues>`_ and `PR
   tracker <https://github.com/opendilab/DI-engine/pulls>`_ 

-  I have mentioned version numbers, operating system and environment,
   where applicable: 

   .. code:: python

      import ding, torch, sys
      print(ding.__version__, torch.__version__, sys.version, sys.platform)


In the PR template, Description is used to describe the role and function of the current PR, Related Issue is used to list related issues, and TODO is used to list the work that has not yet been completed.

In addition, there is a Check List to ensure that source branch is merged, all conflicts are resolved, code style checks passed, and all tests passed. See below for details:


   -  merge the latest version source branch/repo, and resolve all the conflicts

   -  pass style check

   -  pass all the tests

For the naming convention of PR, please refer to the git commit chapter. Also, if the current PR is still under development, you can add  ``WIP:`` at the beginning of the PR name, it is the abbreviation of Work In Progress.


2. Set up Label and Milestone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each issue and PR needs to be marked with a label, and indicate the relevant important milestone, which is used to track the medium and long-term goals corresponding to each specific task. The two need to be specified here:

.. image:: ./images/github_label_milestone.png
    :scale: 25%
    :align: center

DI-engine repo currently supports labels:

.. image:: ./images/github_label1.png
    :scale: 33%
    :align: center

.. image:: ./images/github_label2.png
    :scale: 33%
    :align: center

DI-engine repo currently supports milestones:

.. image:: ./images/github_milestone.png
    :scale: 33%
    :align: center


3. PR's Github Actions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./images/github_actions_all.png
    :scale: 25%
    :align: center

GitHub actions is a continuous integration for automating various tasks. It is mainly used in DI-engine for various tests (algorithm tests, platform tests, style tests, unit tests, etc.), and only when a PR passes all necessary tests, it can be merged. If any actions are not passed, the PR will be displayed as shown below:

.. image:: ./images/github_actions_all.png
    :scale: 25%
    :align: center

At this point, you need to click Details to view the specific failure reasons. If you can pass the tests locally, but CI fails, try rerun:

.. image:: ./images/github_actions_rerun.png
    :scale: 25%
    :align: center


.. note::
    
    If you want to know more, you can reference \ `教程 <http://www.ruanyifeng.com/blog/2019/09/getting-started-with-github-actions.html>`__\


4. PR's Code Review
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PR review requires these five perspectives: code style, algorithm principle, computational efficiency, interface ease of use, and compatibility. Any questions can be commented. It is recommended to take a certain time every day to go through the PRs on github to know what the community is doing, and what can be learned or improved from each other.

If you need to review other people's PRs, there are generally two ways to comment:

One is to comment directly in the conversation of PR, usually for the whole, as shown below:

.. image:: ./images/github_review11.png
    :scale: 33%
    :align: center

.. image:: ./images/github_review12.png
    :scale: 33%
    :align: center

The second is to comment on a specific line or piece of code, you can click the plus sign in Files Changed to create a new comment, as shown below:


.. image:: ./images/github_review2.png
    :scale: 33%
    :align: center


.. note::
    
    Generally speaking, the workflow of a PR is as follows:

    1. Discuss in the Discussion. Someone summarizes and raises the issue, then the developer needs to develop for the issue

    2. Submit Pull Request on Github

    3. Code Development

    4. Assign someone to do code review, fix problems raised by others, finish all the development work

    5. Merge the latest source branch (usually main) and resolve the conflict, make sure to pass github CI, and finally wait to be merged
