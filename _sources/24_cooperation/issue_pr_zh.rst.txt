Github 的使用
~~~~~~~~~~~~~~~~~~~

本节将对 Github 中的 issue & pull request 进行介绍。

1. 从模板创建
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

issue 和 pr 都有自己的模板。

issue 需要选择类别，确保已经阅读过文档和之前的 issue 和 pr，并指明版本号、操作系统，然后才是正式描述这个 issue。详细见下：

-  I have marked all applicable categories (我已经标注了所有适用类别):

   -  exception-raising bug (引发异常的错误)

   -  RL algorithm bug (RL算法错误)

   -  system worker bug (系统组件错误)

   -  system utils bug (系统工具错误)

   -  code design/refactor (代码设计/重构)

   -  documentation request (文档请求)

   -  new feature request (新功能请求)

-  I have visited the
   `readme <https://github.com/opendilab/DI-engine/blob/github-dev/README.md>`_
   and `doc <https://opendilab.github.io/DI-engine/>`_ (我已经阅读了 readme 和文档)

-  I have searched through the `issue
   tracker <https://github.com/opendilab/DI-engine/issues>`_ and `pr
   tracker <https://github.com/opendilab/DI-engine/pulls>`_ (我已经浏览了所有 issue 和 pr)

-  I have mentioned version numbers, operating system and environment,
   where applicable: (我描述了版本号、操作系统和环境，它们可以从下述代码得到)

   .. code:: python

      import ding, torch, sys
      print(ding.__version__, torch.__version__, sys.version, sys.platform)


pr 的模板中，Description 用于描述当前 pr 的作用和功能，Related Issue 用于列出相关的 issue，TODO 用于列出目前还没有完成的工作。

此外，还有一个 Check List 用于确保融合了源分支并解决冲突，通过代码风格检查，通过所有测试。详细见下：

   -  merge the latest version source branch/repo, and resolve all the conflicts (合并最新版本的源分支/仓库，并解决所有冲突)

   -  pass style check (通行代码风格检查)

   -  pass all the tests (通过所有测试)


pr 的命名规范可以参考 git commit 章节。此外，如果当前 pr 仍在开发中，可以在 pr 名字的开头加上\ ``WIP:``\ 标记，它是 Work
In Progess 的缩写。

2. 设置 Label 和 Milestone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每个 issue 和 pr 都需要被打上标签label，并注明相关的重要时间节点 milestone，milestone 的意义是追踪每个具体任务对应的中长期目标，二者需要在界面的这个位置进行指定：

.. image:: ./images/github_label_milestone.png
    :scale: 25%
    :align: center

DI-engine repo 中目前支持以下 label：

.. image:: ./images/github_label1.png
    :scale: 33%
    :align: center

.. image:: ./images/github_label2.png
    :scale: 33%
    :align: center

目前的 milestone 支持：

.. image:: ./images/github_milestone.png
    :scale: 33%
    :align: center


3. PR 的 Github Actions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./images/github_actions_all.png
    :scale: 25%
    :align: center

GitHub actions 是一种持续式集成，用于自动化完成各种任务。DI-engine 中主要使用 actions 进行各种测试（算法测试、平台测试、风格测试、单元测试等），只有当一个 pr 通过了所有必要的测试，它才可以被 merge。假如有 actions 没有通过，pr 会显示如下图：

.. image:: ./images/github_actions_all.png
    :scale: 25%
    :align: center

此时就需要点击 Details 进入查看具体失败原因。如果本地可以通过测试，但 CI 不通过，可以尝试 rerun：

.. image:: ./images/github_actions_rerun.png
    :scale: 25%
    :align: center


.. note::
    
    如果还想进一步了解，可以移步\ `教程 <http://www.ruanyifeng.com/blog/2019/09/getting-started-with-github-actions.html>`__\


4. PR 的 Code Review
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PR review 需要从以下五个角度去看：代码风格，算法原理，计算效率，接口易用性，兼容性。任何问题都可以提 comment。推荐每天抽出一定时间看看 github 上的 PR 看看整个开发社区在做什么，有什么可以学习或者互相提升的地方。

如果需要 review 别人的 PR，一般有两种评论的方式：

一是直接在 pr 的 conversation 中评论，通常是针对整体进行评论，如下图：

.. image:: ./images/github_review11.png
    :scale: 33%
    :align: center

.. image:: ./images/github_review12.png
    :scale: 33%
    :align: center

二是针对具体某行或某段代码进行评论，可以在 Files Changed 中点击加号新建评论，如下图：

.. image:: ./images/github_review2.png
    :scale: 33%
    :align: center


.. note::
    
    一般来讲，一个 PR 的工作流程如下：

    1. 在 discussion 中进行讨论，某人总结并提了 issue，开发者现在需要针对 issue 进行开发

    2. 在 github 提 Pull Request

    3. 代码开发

    4. 分配给某人进行 code review，解决他人提出的问题，完成所有的开发工作

    5. merge 最新源分支（一般是 main）并解决冲突，保证通过 github CI，最终等待被合并
