Code Comment
===================

.. toctree::
    :maxdepth: 3

Code Comment Template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have the following comment format:

  1. File head 
      - Goal: Copyright, license and the file's main function.
  2. Line comment 
      - Goal: Explain the implementation or emphasize the code detail.
  3. Class and function comment
      - Goal: Explain the main function, input and output type and effect.
      - Format: Overview, Arguments, Returns, Shapes, Notes, etc.
        (If it doesn't have the corresponding attribute, you can omit it)
  4. TODO
      - Goal: Set a todo task.
      - Format: `# TODO(assignee name) task statement`.
  5. ISSUE
      - Goal: Issue some doubt about the code.
      - Format: `# ISSUE(questioner name) issue statement`.

.. note::
    According to the formatted class and function comment, we can automatically generate the corresponding doc,
    a example result is `link <../package_ref/model/alphastar.html>`_

We recommend that you should use English in code line comment. And you should obey these rules:

    1. Capitalize the first letter in a sentence.
    2. Use period "." to correctly end a sentence, which is widely adopted in English grammar. Do not always use comma "," to separate different sentences, which is quite common in Chinese grammar.

You can know about the specific format from the following code example.

.. code::

    class BaseLearner(object):
        r"""
        Overview:
            Base class for model learning.
        Interface:
            __init__, train, start, setup_dataloader, close, call_hook, register_hook, save_checkpoint
        Property:
            learn_info, priority_info, last_iter, name, rank, policy
            tick_time, monitor, log_buffer, logger, tb_logger, load_path
        """

        _name = "BaseLearner"  # override this variable for sub-class learner

        def __init__(self, cfg: EasyDict) -> None:
            """
            Overview:
                Init method. Load config and use ``self._cfg`` setting to build common learner components,
                e.g. logger helper, checkpoint helper, hooks.
                Policy is not initialized here, but set afterwards through policy setter.
            Arguments:
                - cfg (:obj:`EasyDict`): Learner config, you can view `cfg <../../../configuration/index.html>`_ for ref.
            Notes:
                If you want to debug in sync CUDA mode, please add the following code at the beginning of ``__init__``.

                .. code:: python

                    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
            """
            self._cfg = deep_merge_dicts(base_learner_default_config, cfg)
            self._learner_uid = get_task_uid()  # ISSUE(Tony) Is _learner_uid used? Can it be deleted?
            self._load_path = self._cfg.load_path
            self._use_cuda = self._cfg.get('use_cuda', False)
            self._use_distributed = self._cfg.use_distributed

            # Learner rank. Used to discriminate which GPU it uses.
            self._rank = get_rank()
            self._device = 'cuda:{}'.format(self._rank % 8) if self._use_cuda else 'cpu'

            # Logger (Monitor is initialized in policy setter)
            # Only rank == 0 learner needs monitor and tb_logger, others only need text_logger to display terminal output.
            self._timer = EasyTimer()
            rank0 = True if self._rank == 0 else False
            self._logger, self._tb_logger = build_logger('./log/learner', 'learner', rank0)
            self._log_buffer = build_log_buffer()

            # Checkpoint helper. Used to save model checkpoint.
            self._checkpointer_manager = build_checkpoint_helper(self._cfg)
            # Learner hook. Used to do specific things at specific time point. Will be set in ``_setup_hook``
            self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
            # Priority info. Used to update replay buffer according to data's priority.
            self._priority_info = None
            # Last iteration. Used to record current iter.
            self._last_iter = CountVar(init_val=0)
            self.info(pretty_print({
                "config": self._cfg,
            }, direct_print=False))

            # Setup wrapper and hook.
            self._setup_wrapper()
            self._setup_hook()

        def _time_wrapper(self, fn: Callable, name: str) -> Callable:
            """
            Overview:
                Wrap a function and measure the time it used.
            Arguments:
                - fn (:obj:`Callable`): Function to be time_wrapped.
                - name (:obj:`str`): Name to be registered in ``_log_buffer``.
            Returns:
                - wrapper (:obj:`Callable`): The wrapper to acquire a function's time.
            """

            def wrapper(*args, **kwargs) -> Any:
                with self._wrapper_timer:
                    ret = fn(*args, **kwargs)
                self._log_buffer[name] = self._wrapper_timer.value
                return ret

            return wrapper
        
        # TODO(Lisa) Not finished.


        learner_mapping = {}


        def register_learner(name: str, learner: type) -> None:
            """
            Overview:
                Add a new Learner class with its name to dict learner_mapping, any subclass derived from BaseLearner must
                use this function to register in DI-engine system before instantiate.
            Arguments:
                - name (:obj:`str`): Name of the new Learner.
                - learner (:obj:`type`): The new Learner class, should be subclass of ``BaseLearner``.
            """
            assert isinstance(name, str)
            assert issubclass(learner, BaseLearner)
            learner_mapping[name] = learner

.. note::
    Sometimes, we add line comment behind the corresponding code line, but sometimes it will surpass the max-line-length(default 120),
    and you can solve this problem by adding **# noqa** at the end of this line, which is the ignore sign of the codestyle checker.
    And of course you can insert the line comment in front of the code line as a new line.