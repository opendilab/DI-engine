Git 协作指南
=================================
这份文档旨在给了解 git 基本使用方法的小伙伴们一些规范 git 协作开发流程的建议。如果你还不熟悉 git ，可以 **依次** 阅读 `廖雪峰的 git 教程 <https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/>`_ 和 `Pro Git v2 <https://git-scm.com/book/zh/v2>`_ 。廖的教程方便一小时入门，pro git 讲得足够深入，是 git 官方的推荐教程。

协作流程
-------------
1.初始化你的代码仓库。这里主要分两种情况：从新开始创建一个仓库（比如自己造轮子），或是打算修改现有代码。

  - 创建新代码仓库：一般建议把新的代码库放到各自的小组 (Group) 下，例如 ``XLab/Cell`` ，这样的话，同组成员会直接有访问和编辑的权限。如果是自己开发使用，或还没有准备好发布到小组内，则把代码仓库创建在自己的用户名下。例如 ``niuyazhe/my_test_project`` 。所有新代码仓库 **务必** 设置为 ``private`` 。

  - 修改现有代码：对现有代码的修改，建议fork出一个新代码仓库至自己的用户名下（跨组仓库推荐用fork），在上面做修改，最后再在原仓库上提 ``merge request`` 。或者也可以在原repo下创建一个自己的开发分支（同组仓库推荐用新分支），然后同样通过 ``merge request`` 来提交改动， **禁止** 项目维护者之外的人直接更新 ``master`` 分支，一般情况下也 **禁止** 使用 ``git push -f`` 强行覆盖推送本地改动到远程分支。

.. tip::

    在使用fork方式时，为保持fork代码库和原代码库同步，可以在本地将原代码库添加至 ``git remote`` ， 并单独用一个 branch 保持追踪原始代码库的 master 分支，实例如下：

    .. code:: bash

        # after forking `group_foo/proj_bar.git` to `user_baz/proj_bar.git`
        git clone <forked_path>
        git remote add base <origin_path>  # base is user-defined name referred to origin remote
        git fetch base
        git checkout -b base-master base/master  # use `base-master` to track origin repository

2. 本地代码仓库开发。在创建好代码仓库并 clone 至本地之后，即可开始代码开发。这里有下面几个建议：

   - 任何一个较大的代码改动或功能更新，需要通过 GitLab Issue 来描述具体要做的工作或跟同事讨论相关解决方案，同时新开一个 ``WIP(Work In Progress)`` 的 ``merge request`` 跟进自己的进展，其他同事也可以在该MR下comment，进行讨论或code review。
   - 如果是修复 bug, 建议现在 issue 页面描述 bug 的表现及触发场景，以防其他同事踩雷，且为这个issue 打上 ``Bug Report`` label。
   - 如果是打算添加新功能，那么 **在开始写代码之前** ，请先线下和代码库的负责人聊一下你的意图、添加这个功能的原因和打算使用的实现方式，确定加这个功能是必须且可行的。之后，请仍然在原代码仓库提一个 issue，描述你准备加入的功能和具体的 TODO 列表，且为这个issue打上相应label，比如算法更新使用 ``Algorithm`` 。 
   - 在进行每一个 issue 对应的开发任务时，可以在本地按照自己的工作节奏 commit ，尽量保证每个commit只对应一种相关的代码改动，也可以把这些做到一半的 commits 推送进自己的 fork 仓库内以防止工作丢失。但是，在完成发起的 issue 的目标（或某一条 TODO）前， **不要把不完整的 commit 推送至原始仓库** 。


3. 完成本地开发之后, 同步上游代码，整合自己的本地 commits，处理 conflicts。简单在conflicts可以直接在 ``merge request`` 对应的gitlab网页上直接进行操作，较为复杂的需要在本地fetch目标分支的改动之后，手动进行merge，一些较复杂的情形可以使用 ``git rebase`` 来完成，（具体细节可以参考 `Pro Git-分支-变基 <https://git-scm.com/book/zh/v2/Git-%E5%88%86%E6%94%AF-%E5%8F%98%E5%9F%BA>`_ 一节

4. 冲突解决后，在MR或Issue里描述代码改动对应的工作内容，由项目负责人或指定相关同事进行 code review。代码合并成功后，关闭相应的 GitLab Issue。

git规范
--------------

这一节主要列一些在使用 git 时，应该特别注意的点:

  - 提交至 master 分支的 merge request，其代码必须是确保能够 **正确运行** 的。具体的开发者需要做好本地的代码风格检查和单元测试，推荐引入自动测试和持续集成(CI)来确保 master 分支的稳定性；
  - 在编辑 git repository 中的文件时，请特别留意 **换行符** 。默认情况下，Windows 系统使用 ``CRLF`` 换行，Unix 系统使用 ``LF`` 换行。我们强制要求所有签入 git repository 文件 **必须使用 LF 换行** ，以避免潜在的系统级 bug。Windows下可以使用 ``dos2unix`` `批量处理 <https://stackoverflow.com/questions/11929461/how-can-i-run-dos2unix-on-an-entire-directory>`_ 整个文件夹内的换行;
  - 开发完成后无用的分支请在远程仓库中删除。这样主要是为了防止公共代码库分支混乱、clone 耗时；
  - 注意编写合适的 ``.gitignore`` ，不要把任何与项目实现无关的文件签入 git，例如 macOS 下的 ``.DS_Store`` 文件夹，IDE 配置文件夹 ``.vscode`` 、 ``.idea`` ，运行日志文件等；
  - 除非必要， **绝对不要把大容量二进制文件嵌入git repo中** ，例如训练的模型文件、二进制库文件等，这些文件可能会带来多种多样的问题。 如果要引入这些文件，可使用 ``gif lfs``
