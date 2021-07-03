Step 0. clean old version
rm ~/.local/lib/python3.6/site-packages/hpc_*.so
rm ~/.local/lib/python3.6/site-packages/hpc_rl* -rf

Step 1.
pip install hpc_rll-0.0.1-cp36-cp36m-linux_x86_64.whl --user
ls ~/.local/lib/python3.6/site-packages/hpc_rl

Step 2.
cd tests
python3 test_gae.py
