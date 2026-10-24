$env:UV_CACHE_DIR='G:\robotTest\pipcache\uv'
$py='G:\robotTest\env_isaac6\Scripts\python.exe'; $L='G:\robotTest\IsaacLab6\source'
uv pip install --python $py --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match -e "$L\isaaclab" -e "$L\isaaclab_physx" -e "$L\isaaclab_assets" -e "$L\isaaclab_tasks" -e "$L\isaaclab_rl[rsl-rl]"
"=== LAB FINITO exit $LASTEXITCODE"
uv pip install --python $py --reinstall-package torch --reinstall-package torchvision torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
"=== TORCH FINITO exit $LASTEXITCODE"
& $py -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
