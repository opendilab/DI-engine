Step 0. clean old version
rm ~/.local/lib/python3.6/site-packages/hpc_*.so
rm ~/.local/lib/python3.6/site-packages/hpc_rl* -rf
rm ~/.local/lib/python3.6/site-packages/di_hpc_rl* -rf

Step 1.
pip install di_hpc_rll-0.0.1-cp36-cp36m-linux_x86_64.whl --user
ls ~/.local/lib/python3.6/site-packages/di_hpc_rl*
ls ~/.local/lib/python3.6/site-packages/hpc_rl*

Step 2.
python3 tests/test_gae.py